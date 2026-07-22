from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from api.image_inputs import (
    MAX_IMAGE_MODEL_CHARS,
    MAX_IMAGE_PROMPT_CHARS,
    ImageSource,
    close_image_sources,
    parse_image_edit_request,
    read_image_sources_sync,
)
from api.support import require_identity, resolve_image_base_url
from services.content_filter import check_request, request_shape, request_text
from services.editable_file_task_service import (
    MAX_EDITABLE_BASE64_CHARS,
    MAX_EDITABLE_IMAGE_COUNT,
    MAX_EDITABLE_PROMPT_CHARS,
    MAX_EDITABLE_TASK_ID_CHARS,
    editable_file_task_service,
)
from services.log_service import LoggedCall
from services.protocol import (
    anthropic_v1_messages,
    openai_v1_chat_complete,
    openai_v1_image_edit,
    openai_v1_image_generations,
    openai_v1_models,
    openai_v1_response,
    openai_search,
)
from services.image_task_runtime import ImageTaskDeadline, ImageTaskRuntimeError, image_task_runtime
from services.image_io_runtime import image_io_runtime
from utils.helper import (
    count_input_image_references,
    has_response_image_generation_tool,
    is_image_chat_request,
    parse_image_count,
)


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=MAX_IMAGE_PROMPT_CHARS)
    model: str = Field(default="gpt-image-2", max_length=MAX_IMAGE_MODEL_CHARS)
    n: int = Field(default=1, ge=1, le=4)
    size: str | None = None
    quality: str = "auto"
    response_format: str = "b64_json"
    history_disabled: bool = True
    stream: bool | None = None
    image: str | list[str] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    prompt: str | None = None
    n: int | None = None
    stream: bool | None = None
    modalities: list[str] | None = None
    messages: list[dict[str, object]] | None = None


class ResponseCreateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    input: object | None = None
    tools: list[dict[str, object]] | None = None
    tool_choice: object | None = None
    stream: bool | None = None


class AnthropicMessageRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    messages: list[dict[str, object]] | None = None
    system: object | None = None
    stream: bool | None = None


class SearchRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


EditableBase64Image = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_EDITABLE_BASE64_CHARS),
]


class EditableFileTaskRequest(BaseModel):
    prompt: str = Field(default="", max_length=MAX_EDITABLE_PROMPT_CHARS)
    base64_images: list[EditableBase64Image] = Field(
        default_factory=list,
        max_length=MAX_EDITABLE_IMAGE_COUNT,
    )
    client_task_id: str | None = Field(default=None, max_length=MAX_EDITABLE_TASK_ID_CHARS)


async def filter_or_log(call: LoggedCall, text: str) -> None:
    try:
        await run_in_threadpool(check_request, text)
    except HTTPException as exc:
        call.log("调用失败", status="failed", error=str(exc.detail))
        raise


def _checked_image_handler(handler, payload: dict[str, object]):
    """在图片专用线程内完成审核和生成，避免审核网络调用占满公共线程池。"""

    from services.image_task_runtime import image_deadline_from_payload

    check_request(
        request_text(payload.get("prompt"), payload.get("messages"), payload.get("input")),
        deadline=image_deadline_from_payload(payload),
    )
    return handler(payload)


def _new_request_image_deadline(request: Request) -> ImageTaskDeadline:
    """复用中间件记录的请求起点；独立路由测试没有中间件时自动从当前时刻开始。"""

    state = request.scope.get("state")
    started_at = state.get("image_request_started_at") if isinstance(state, dict) else None
    return image_task_runtime.new_deadline(started_at=started_at if isinstance(started_at, (int, float)) else None)


def _editable_runtime_error_response(exc: ImageTaskRuntimeError) -> JSONResponse:
    """后台任务只同步返回入队结果；队列满等运行时错误使用统一 OpenAI 错误结构。"""

    headers = {"Retry-After": "5"} if exc.status_code == 429 else None
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_openai_error(),
        headers=headers,
    )


def _checked_image_edit_handler(
    payload: dict[str, object],
    image_sources: list[ImageSource],
    mask_sources: list[ImageSource],
    deadline: ImageTaskDeadline,
):
    """把上传读取、URL 下载、内容审核和图片编辑纳入同一总截止时间。"""

    if len(image_sources) > 4 or len(mask_sources) > 4:
        raise HTTPException(status_code=400, detail={"error": "图片和蒙版各最多支持 4 张"})
    combined = read_image_sources_sync(image_sources + mask_sources, deadline, max_count=8)
    image_count = len(image_sources)
    payload["images"] = combined[:image_count]
    if mask_sources:
        payload["mask"] = combined[image_count:]
    return _checked_image_handler(openai_v1_image_edit.handle, payload)


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/v1/models")
    async def list_models(authorization: str | None = Header(default=None)):
        require_identity(authorization)
        try:
            return await run_in_threadpool(openai_v1_models.list_models)
        except Exception as exc:
            raise HTTPException(status_code=502, detail={"error": str(exc)}) from exc

    @router.post("/v1/images/generations")
    async def generate_images(
            body: ImageGenerationRequest,
            request: Request,
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        deadline = _new_request_image_deadline(request)
        payload = body.model_dump(mode="python")
        payload["base_url"] = resolve_image_base_url(request)
        payload["_image_task_deadline"] = deadline
        has_image = bool(body.image)
        input_image_count = len(body.image) if isinstance(body.image, list) else int(bool(body.image))
        if input_image_count > 4:
            raise HTTPException(status_code=400, detail={"error": "一次请求最多支持 4 张输入图片"})
        call = LoggedCall(identity, "/v1/images/generations", body.model, "图生图" if has_image else "文生图", request_text=body.prompt)
        return await call.run_image(
            _checked_image_handler,
            openai_v1_image_generations.handle,
            payload,
            stream=bool(body.stream),
            deadline=deadline,
            weight=max(body.n, input_image_count),
        )

    @router.post("/v1/images/edits")
    async def edit_images(
            request: Request,
            authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        deadline = _new_request_image_deadline(request)
        payload, image_sources, mask_sources = await parse_image_edit_request(request)
        prompt = str(payload["prompt"])
        model = str(payload["model"])
        call = LoggedCall(identity, "/v1/images/edits", model, "图生图", request_text=prompt)
        payload["base_url"] = resolve_image_base_url(request)
        payload["_image_task_deadline"] = deadline
        return await call.run_image(
            _checked_image_edit_handler,
            payload,
            image_sources,
            mask_sources,
            deadline,
            stream=bool(payload.get("stream")),
            deadline=deadline,
            weight=max(
                int(payload.get("n") or 1),
                len(image_sources) + len(mask_sources),
            ),
            cleanup=lambda: close_image_sources(image_sources + mask_sources),
        )

    @router.post("/v1/chat/completions")
    async def create_chat_completion(
        body: ChatCompletionRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        payload = body.model_dump(mode="python")
        model = str(payload.get("model") or "auto")
        request_preview = request_text(payload.get("prompt"), payload.get("messages"))
        call = LoggedCall(
            identity,
            "/v1/chat/completions",
            model,
            "图片生成" if is_image_chat_request(payload) else "文本生成",
            request_text=request_preview,
            request_shape=request_shape(payload.get("messages")),
        )
        if is_image_chat_request(payload):
            deadline = _new_request_image_deadline(request)
            payload["_image_task_deadline"] = deadline
            return await call.run_image(
                _checked_image_handler,
                openai_v1_chat_complete.handle,
                payload,
                stream=bool(payload.get("stream")),
                deadline=deadline,
                weight=parse_image_count(payload.get("n")),
            )
        vision_weight = count_input_image_references(payload.get("messages"))
        if vision_weight:
            if vision_weight > 4:
                raise HTTPException(status_code=400, detail={"error": "一次请求最多支持 4 张输入图片"})
            deadline = _new_request_image_deadline(request)
            payload["_image_task_deadline"] = deadline
            return await call.run_image(
                _checked_image_handler,
                openai_v1_chat_complete.handle,
                payload,
                stream=bool(payload.get("stream")),
                deadline=deadline,
                weight=vision_weight,
            )
        await filter_or_log(call, request_preview)
        return await call.run(openai_v1_chat_complete.handle, payload)

    @router.post("/v1/responses")
    async def create_response(
        body: ResponseCreateRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        payload = body.model_dump(mode="python")
        model = str(payload.get("model") or "auto")
        request_preview = request_text(payload.get("input"), payload.get("instructions"))
        call = LoggedCall(
            identity,
            "/v1/responses",
            model,
            "Responses",
            request_text=request_preview,
            request_shape=request_shape(payload.get("input")),
        )
        if has_response_image_generation_tool(payload):
            deadline = _new_request_image_deadline(request)
            payload["_image_task_deadline"] = deadline
            return await call.run_image(
                _checked_image_handler,
                openai_v1_response.handle,
                payload,
                stream=bool(payload.get("stream")),
                deadline=deadline,
            )
        vision_weight = count_input_image_references(payload.get("input"))
        if vision_weight:
            if vision_weight > 4:
                raise HTTPException(status_code=400, detail={"error": "一次请求最多支持 4 张输入图片"})
            deadline = _new_request_image_deadline(request)
            payload["_image_task_deadline"] = deadline
            return await call.run_image(
                _checked_image_handler,
                openai_v1_response.handle,
                payload,
                stream=bool(payload.get("stream")),
                deadline=deadline,
                weight=vision_weight,
            )
        await filter_or_log(call, request_preview)
        return await call.run(openai_v1_response.handle, payload)

    @router.post("/v1/messages")
    async def create_message(
            body: AnthropicMessageRequest,
            request: Request,
            authorization: str | None = Header(default=None),
            x_api_key: str | None = Header(default=None, alias="x-api-key"),
            anthropic_version: str | None = Header(default=None, alias="anthropic-version"),
    ):
        identity = require_identity(authorization or (f"Bearer {x_api_key}" if x_api_key else None))
        payload = body.model_dump(mode="python")
        model = str(payload.get("model") or "auto")
        request_preview = request_text(payload.get("system"), payload.get("messages"), payload.get("tools"))
        call = LoggedCall(identity, "/v1/messages", model, "Messages", request_text=request_preview)
        vision_weight = count_input_image_references(payload.get("messages"))
        if vision_weight:
            if vision_weight > 4:
                raise HTTPException(status_code=400, detail={"error": "一次请求最多支持 4 张输入图片"})
            # Anthropic 视觉输入同样包含远程下载，必须隔离；输出仍保持 Anthropic SSE。
            deadline = _new_request_image_deadline(request)
            payload["_image_task_deadline"] = deadline
            return await call.run_image(
                _checked_image_handler,
                anthropic_v1_messages.handle,
                payload,
                stream=bool(payload.get("stream")),
                deadline=deadline,
                weight=vision_weight,
                sse="anthropic",
            )
        await filter_or_log(call, request_preview)
        return await call.run(anthropic_v1_messages.handle, payload, sse="anthropic")

    @router.post("/v1/search")
    async def search(body: SearchRequest, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        call = LoggedCall(identity, "/v1/search", openai_search.MODEL, "搜索", request_text=body.prompt)
        await filter_or_log(call, body.prompt)
        return await call.run(openai_search.handle, body.model_dump(mode="python"))

    @router.get("/v1/editable-file-tasks")
    async def list_editable_file_tasks(ids: str = "", authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        task_ids = [item.strip() for item in ids.split(",") if item.strip()]
        try:
            return await image_io_runtime.run(
                editable_file_task_service.list_tasks,
                identity,
                task_ids,
                timeout_secs=2,
                operation_name="可编辑文件任务查询",
            )
        except ImageTaskRuntimeError as exc:
            return _editable_runtime_error_response(exc)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc

    @router.get("/files/{file_path:path}")
    async def download_editable_file(file_path: str):
        try:
            path = await image_io_runtime.run(
                editable_file_task_service.public_file_path,
                file_path,
                timeout_secs=5,
                operation_name="可编辑文件下载",
            )
        except Exception as exc:
            raise HTTPException(status_code=404, detail={"error": "file not found"}) from exc
        return FileResponse(path, filename=path.name)

    @router.post("/v1/ppt/generations")
    async def create_ppt_task(body: EditableFileTaskRequest, request: Request, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        deadline = _new_request_image_deadline(request)
        try:
            return await image_io_runtime.run(
                editable_file_task_service.submit_ppt,
                identity,
                client_task_id=body.client_task_id or "",
                prompt=body.prompt,
                base64_images=body.base64_images,
                base_url=resolve_image_base_url(request),
                deadline=deadline,
                timeout_secs=min(6.0, deadline.remaining() + 0.5),
                operation_name="PPT 任务提交",
            )
        except HTTPException as exc:
            if exc.status_code == 504:
                # 外层提交桥超时后通知仍在专用线程中的校验逻辑停止，防止迟到入队。
                deadline.cancel("PPT 任务提交超时")
            raise
        except ImageTaskRuntimeError as exc:
            return _editable_runtime_error_response(exc)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc

    @router.post("/v1/psd/generations")
    async def create_psd_task(body: EditableFileTaskRequest, request: Request, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        deadline = _new_request_image_deadline(request)
        try:
            return await image_io_runtime.run(
                editable_file_task_service.submit_psd,
                identity,
                client_task_id=body.client_task_id or "",
                prompt=body.prompt,
                base64_images=body.base64_images,
                base_url=resolve_image_base_url(request),
                deadline=deadline,
                timeout_secs=min(6.0, deadline.remaining() + 0.5),
                operation_name="PSD 任务提交",
            )
        except HTTPException as exc:
            if exc.status_code == 504:
                deadline.cancel("PSD 任务提交超时")
            raise
        except ImageTaskRuntimeError as exc:
            return _editable_runtime_error_response(exc)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc

    return router

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.image_inputs import (
    MAX_CLIENT_TASK_ID_CHARS,
    MAX_IMAGE_MODEL_CHARS,
    MAX_IMAGE_PROMPT_CHARS,
    ImageSource,
    close_image_sources,
    parse_image_edit_request,
    read_image_sources_sync,
)
from api.support import require_identity, resolve_image_base_url
from services.image_task_runtime import ImageTaskDeadline, ImageTaskRuntimeError, image_task_runtime
from services.image_task_service import image_task_service
from services.image_io_runtime import image_io_runtime


class ImageGenerationTaskRequest(BaseModel):
    client_task_id: str = Field(..., min_length=1, max_length=MAX_CLIENT_TASK_ID_CHARS)
    prompt: str = Field(..., min_length=1, max_length=MAX_IMAGE_PROMPT_CHARS)
    model: str = Field(default="gpt-image-2", max_length=MAX_IMAGE_MODEL_CHARS)
    size: str | None = None
    quality: str = "auto"


class ResumePollRequest(BaseModel):
    extra_timeout_secs: float = Field(default=30.0, ge=5.0, le=120.0)


def _parse_task_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _runtime_error_response(exc: ImageTaskRuntimeError) -> JSONResponse:
    headers = {"Retry-After": "5"} if exc.status_code == 429 else None
    return JSONResponse(status_code=exc.status_code, content=exc.to_openai_error(), headers=headers)


def _new_request_image_deadline(request: Request) -> ImageTaskDeadline:
    """让后台任务提交也从 HTTP 请求进入服务时开始计算总时限。"""

    state = request.scope.get("state")
    started_at = state.get("image_request_started_at") if isinstance(state, dict) else None
    return image_task_runtime.new_deadline(started_at=started_at if isinstance(started_at, (int, float)) else None)


def _read_edit_inputs(
    image_sources: list[ImageSource],
    mask_sources: list[ImageSource],
    deadline: ImageTaskDeadline,
):
    """一次读取图片和蒙版，让 50MB 总上限覆盖两组输入。"""

    if len(image_sources) > 4 or len(mask_sources) > 4:
        raise HTTPException(status_code=400, detail={"error": "图片和蒙版各最多支持 4 张"})
    combined = read_image_sources_sync(image_sources + mask_sources, deadline, max_count=8)
    image_count = len(image_sources)
    return combined[:image_count], combined[image_count:] or None


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/image-tasks")
    async def list_image_tasks(
        ids: str = Query(default=""),
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        return await image_io_runtime.run(
            image_task_service.list_tasks,
            identity,
            _parse_task_ids(ids),
        )

    @router.post("/api/image-tasks/generations")
    async def create_generation_task(
        body: ImageGenerationTaskRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        deadline = _new_request_image_deadline(request)
        try:
            return await image_io_runtime.run(
                image_task_service.submit_generation,
                identity,
                client_task_id=body.client_task_id,
                prompt=body.prompt,
                model=body.model,
                size=body.size,
                quality=body.quality,
                base_url=resolve_image_base_url(request),
                deadline=deadline,
            )
        except ImageTaskRuntimeError as exc:
            return _runtime_error_response(exc)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc

    @router.post("/api/image-tasks/edits")
    async def create_edit_task(
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        deadline = _new_request_image_deadline(request)
        payload, image_sources, mask_sources = await parse_image_edit_request(request)
        try:
            client_task_id = str(payload.get("client_task_id") or "").strip()
            if not client_task_id:
                raise HTTPException(status_code=400, detail={"error": "client_task_id is required"})
            prompt = str(payload["prompt"])
            model = str(payload["model"])
            try:
                return await image_io_runtime.run(
                    image_task_service.submit_edit_with_loader,
                    identity,
                    client_task_id=client_task_id,
                    prompt=prompt,
                    model=model,
                    size=payload["size"],
                    quality=payload["quality"],
                    base_url=resolve_image_base_url(request),
                    input_loader=lambda active_deadline: _read_edit_inputs(
                        image_sources,
                        mask_sources,
                        active_deadline,
                    ),
                    deadline=deadline,
                    weight=max(1, len(image_sources) + len(mask_sources)),
                    timeout_secs=deadline.remaining() + 0.5,
                )
            except ImageTaskRuntimeError as exc:
                return _runtime_error_response(exc)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
        finally:
            # submit 立即拒绝、排队超时和 loader 中途失败都要关闭上传临时文件。
            # loader 成功时内部已关闭，再次关闭是安全的幂等操作。
            close_image_sources(image_sources + mask_sources)

    @router.post("/api/image-tasks/{task_id}/resume-poll")
    async def resume_image_poll(
        task_id: str,
        body: ResumePollRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        try:
            return await image_io_runtime.run(
                image_task_service.resume_poll,
                identity,
                task_id,
                body.extra_timeout_secs,
            )
        except ImageTaskRuntimeError as exc:
            return _runtime_error_response(exc)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc

    return router

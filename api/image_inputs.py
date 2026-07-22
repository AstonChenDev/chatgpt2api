from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, TypeGuard
from urllib.parse import unquote, unquote_to_bytes, urlparse

from curl_cffi import requests
from fastapi import HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from services.deadline_http import DeadlineSession
from services.image_task_runtime import ImageTaskDeadline, ImageTaskRuntimeError
from services.proxy_service import proxy_settings

ImageInput = tuple[bytes, str, str]


@dataclass(frozen=True)
class Base64ImageSource:
    """尚未解码的 base64 图片；真正解码必须在图片专用工作线程内进行。"""

    encoded: str
    filename: str
    mime_type: str


ImageSource = str | UploadFile | ImageInput | Base64ImageSource

# 单张限制防止解码/合成时瞬时内存成倍放大；总量限制防止一次请求堆入多张大图。
MAX_IMAGE_COUNT = 4
MAX_IMAGE_REFERENCE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_TOTAL_BYTES = 50 * 1024 * 1024
MAX_BASE64_IMAGE_CHARS = ((MAX_IMAGE_REFERENCE_BYTES + 2) // 3) * 4
MAX_IMAGE_PROMPT_CHARS = 32_000
MAX_CLIENT_TASK_ID_CHARS = 128
MAX_IMAGE_MODEL_CHARS = 128
IMAGE_REFERENCE_FIELDS = {"image", "image[]", "images", "images[]", "image_url", "image_url[]"}
MASK_REFERENCE_FIELDS = {"mask", "mask[]"}


def _clean(value: object, default: str = "") -> str:
    """清理字符串：转换为字符串并去掉首尾空白。"""
    text = str(value if value is not None else default).strip()
    return text or default


def _is_upload(value: object) -> TypeGuard[UploadFile]:
    """识别上传文件：兼容 Starlette 表单返回的 UploadFile。"""
    return isinstance(value, UploadFile)


def _parse_bool(value: object) -> bool | None:
    """解析布尔字段：兼容 JSON 布尔值和表单字符串。"""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = _clean(value).lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    raise HTTPException(status_code=400, detail={"error": "stream must be a boolean"})


def _parse_count(value: object) -> int:
    """解析生成数量：保持图片接口的 1 到 4 限制。"""
    try:
        count = int(value or 1)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail={"error": "n must be an integer"}) from exc
    if count < 1 or count > 4:
        raise HTTPException(status_code=400, detail={"error": "n must be between 1 and 4"})
    return count


def _payload_from_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """构造图片编辑载荷：从表单或 JSON 字段提取通用参数。"""
    prompt = _clean(fields.get("prompt"))
    if not prompt:
        raise HTTPException(status_code=400, detail={"error": "prompt is required"})
    if len(prompt) > MAX_IMAGE_PROMPT_CHARS:
        raise HTTPException(status_code=400, detail={"error": "prompt 最多支持 32000 个字符"})
    model = _clean(fields.get("model"), "gpt-image-2")
    if len(model) > MAX_IMAGE_MODEL_CHARS:
        raise HTTPException(status_code=400, detail={"error": "model 最多支持 128 个字符"})
    payload = {
        "prompt": prompt,
        "model": model,
        "n": _parse_count(fields.get("n")),
        "size": _clean(fields.get("size")) or None,
        "quality": _clean(fields.get("quality"), "auto"),
        "response_format": _clean(fields.get("response_format"), "b64_json"),
        "stream": _parse_bool(fields.get("stream")),
    }
    if "client_task_id" in fields:
        client_task_id = _clean(fields.get("client_task_id"))
        if len(client_task_id) > MAX_CLIENT_TASK_ID_CHARS:
            raise HTTPException(status_code=400, detail={"error": "client_task_id 最多支持 128 个字符"})
        payload["client_task_id"] = client_task_id
    return payload


def _json_reference_value(value: object) -> object:
    """解析表单图片引用：支持把 images 字段写成 JSON 字符串。"""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _base64_image_source(value: object, filename: str, mime_type: str) -> Base64ImageSource:
    """只做廉价长度预检，避免在事件循环中分配几十 MB 的解码结果。"""

    encoded = str(value).strip()
    if len(encoded) > MAX_BASE64_IMAGE_CHARS:
        raise HTTPException(status_code=400, detail={"error": "单张图片不能超过 20MB"})
    return Base64ImageSource(encoded=encoded, filename=filename, mime_type=mime_type)


def _decode_base64_image(value: object, filename: str, mime_type: str) -> ImageInput:
    try:
        data = base64.b64decode(str(value).strip(), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail={"error": "invalid base64 image data"}) from exc
    if not data:
        raise HTTPException(status_code=400, detail={"error": "image file is empty"})
    if len(data) > MAX_IMAGE_REFERENCE_BYTES:
        raise HTTPException(status_code=400, detail={"error": "单张图片不能超过 20MB"})
    return data, filename, mime_type


def _source_from_object(value: dict[str, Any]) -> list[ImageSource]:
    """提取图片引用对象：支持 image_url 或 url，明确拒绝 file_id。"""
    has_url = "image_url" in value or "url" in value
    if value.get("file_id"):
        raise HTTPException(
            status_code=400,
            detail={"error": "file_id image references are not supported; use image_url instead"},
        )
    inline = value.get("b64_json") or value.get("base64")
    if inline:
        filename = _clean(value.get("filename") or value.get("file_name"), "image.png")
        mime_type = _clean(value.get("mime_type") or value.get("mimeType"), "image/png")
        return [_base64_image_source(inline, filename, mime_type)]
    if not has_url:
        raise HTTPException(status_code=400, detail={"error": "image reference must include image_url"})
    image_url = value.get("image_url", value.get("url"))
    if isinstance(image_url, dict):
        image_url = image_url.get("url")
    return _sources_from_value(image_url)


def _sources_from_value(value: object) -> list[ImageSource]:
    """展开图片引用：把字符串、数组和对象统一成图片来源列表。"""
    value = _json_reference_value(value)
    if _is_upload(value):
        return [value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.lower().startswith(("data:", "http://", "https://")):
            return [text]
        return [_base64_image_source(text, "image.png", "image/png")]
    if isinstance(value, list):
        sources: list[ImageSource] = []
        for item in value:
            sources.extend(_sources_from_value(item))
        return sources
    if isinstance(value, dict):
        return _source_from_object(value)
    if value is None:
        return []
    raise HTTPException(status_code=400, detail={"error": "invalid image reference"})


def _json_image_sources(body: dict[str, Any]) -> list[ImageSource]:
    """读取 JSON 图片引用：优先支持官方 images 数组字段。"""
    sources: list[ImageSource] = []
    for key in ("images", "image", "image_url"):
        if key in body:
            sources.extend(_sources_from_value(body.get(key)))
    return sources


def _json_mask_sources(body: dict[str, Any]) -> list[ImageSource]:
    """读取 JSON mask 引用。"""
    mask = body.get("mask")
    if mask is not None:
        return _sources_from_value(mask)
    return []


async def parse_image_edit_request(request: Request) -> tuple[dict[str, Any], list[ImageSource], list[ImageSource]]:
    """解析图片编辑请求：同时支持 multipart 上传和官方 JSON 图片 URL。
    
    返回 (payload, image_sources, mask_sources)
    """
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type == "application/json":
        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail={"error": "invalid JSON body"}) from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail={"error": "JSON body must be an object"})
        return _payload_from_fields(body), _json_image_sources(body), _json_mask_sources(body)

    form = await request.form()
    try:
        fields: dict[str, Any] = {}
        for key in ("client_task_id", "prompt", "model", "n", "size", "quality", "response_format", "stream"):
            value = form.get(key)
            if isinstance(value, str):
                fields[key] = value
        sources: list[ImageSource] = []
        mask_sources: list[ImageSource] = []
        for key, value in form.multi_items():
            if key in IMAGE_REFERENCE_FIELDS:
                sources.extend(_sources_from_value(value))
            elif key in MASK_REFERENCE_FIELDS:
                mask_sources.extend(_sources_from_value(value))
        return _payload_from_fields(fields), sources, mask_sources
    except BaseException:
        # 成功时 UploadFile 由图片工作线程接管；解析失败时没有工作线程，必须在
        # 此处关闭表单临时文件，防止恶意请求耗尽文件描述符。
        await form.close()
        raise


def _extension_from_mime(mime_type: str) -> str:
    """推导图片扩展名：把 MIME 类型转换为常见文件后缀。"""
    subtype = mime_type.split("/", 1)[1].split("+", 1)[0] if "/" in mime_type else "png"
    if subtype == "jpeg":
        return "jpg"
    return re.sub(r"[^a-z0-9]+", "", subtype.lower()) or "png"


def _safe_filename(name: str, mime_type: str, fallback: str) -> str:
    """生成安全文件名：清理 URL 文件名并补齐扩展名。"""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not cleaned:
        cleaned = fallback
    if "." not in cleaned:
        cleaned = f"{cleaned}.{_extension_from_mime(mime_type)}"
    return cleaned


def _decode_data_url(url: str) -> ImageInput:
    """解码 data URL：把内联图片转成标准图片输入元组。"""
    header, separator, payload = url.partition(",")
    if not separator:
        raise HTTPException(status_code=400, detail={"error": "invalid data image URL"})
    mime_type = header.split(";", 1)[0].removeprefix("data:") or "image/png"
    if not mime_type.startswith("image/"):
        raise HTTPException(status_code=400, detail={"error": "image_url must point to an image"})
    encoded_limit = MAX_BASE64_IMAGE_CHARS if ";base64" in header else MAX_IMAGE_REFERENCE_BYTES * 3
    if len(payload) > encoded_limit:
        raise HTTPException(status_code=400, detail={"error": "单张图片不能超过 20MB"})
    try:
        data = base64.b64decode(payload, validate=True) if ";base64" in header else unquote_to_bytes(payload)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail={"error": "invalid data image URL"}) from exc
    if not data:
        raise HTTPException(status_code=400, detail={"error": "image URL is empty"})
    if len(data) > MAX_IMAGE_REFERENCE_BYTES:
        raise HTTPException(status_code=400, detail={"error": "单张图片不能超过 20MB"})
    return data, f"image_url.{_extension_from_mime(mime_type)}", mime_type


def _response_mime_type(response: requests.Response, parsed_path: str) -> str:
    """识别下载图片类型：优先响应头，必要时按 URL 后缀推断。"""
    header_type = str(response.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
    guessed_type = mimetypes.guess_type(parsed_path)[0] or ""
    if header_type.startswith("image/"):
        return header_type
    if header_type and header_type not in {"application/octet-stream", "binary/octet-stream"}:
        raise HTTPException(status_code=400, detail={"error": "image_url must point to an image"})
    if guessed_type.startswith("image/"):
        return guessed_type
    if not header_type or header_type in {"application/octet-stream", "binary/octet-stream"}:
        return "image/png"
    raise HTTPException(status_code=400, detail={"error": "image_url must point to an image"})


def _filename_from_url(parsed_path: str, mime_type: str) -> str:
    """生成 URL 图片文件名：从链接路径提取名称并做安全化。"""
    raw_name = PurePosixPath(unquote(parsed_path)).name
    return _safe_filename(raw_name, mime_type, "image_url")


def _download_image_url(url: str, deadline: ImageTaskDeadline | None = None) -> ImageInput:
    """下载远程图片：把 http/https 图片链接转成标准图片输入元组。"""
    source = _clean(url)
    if source.startswith("data:"):
        return _decode_data_url(source)
    parsed = urlparse(source)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail={"error": "image_url must be an http or https URL"})
    session = DeadlineSession(image_deadline=deadline)
    response: requests.Response | None = None
    try:
        response = session.get(
            source,
            headers={"Accept": "image/*,*/*;q=0.8", "User-Agent": "chatgpt2api image fetcher"},
            timeout=60,
            stream=True,
            allow_redirects=True,
            **proxy_settings.build_session_kwargs(),
        )
    except ImageTaskRuntimeError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"error": f"image_url fetch failed: {exc}"}) from exc
    try:
        if not 200 <= response.status_code < 300:
            raise HTTPException(status_code=400, detail={"error": f"image_url fetch failed: HTTP {response.status_code}"})
        content_length = _clean(response.headers.get("content-length"))
        if content_length and content_length.isdigit() and int(content_length) > MAX_IMAGE_REFERENCE_BYTES:
            raise HTTPException(status_code=400, detail={"error": "单张图片不能超过 20MB"})
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if deadline is not None:
                deadline.check("下载输入图片")
            if not chunk:
                continue
            size += len(chunk)
            if size > MAX_IMAGE_REFERENCE_BYTES:
                raise HTTPException(status_code=400, detail={"error": "单张图片不能超过 20MB"})
            chunks.append(chunk)
        data = b"".join(chunks)
        if not data:
            raise HTTPException(status_code=400, detail={"error": "image_url returned empty content"})
        mime_type = _response_mime_type(response, parsed.path)
        return data, _filename_from_url(parsed.path, mime_type), mime_type
    finally:
        response.close()
        session.close()


def _validate_image_input(data: bytes, *, total_size: int) -> int:
    """校验单张与请求总大小，返回累加后的字节数。"""

    if not data:
        raise HTTPException(status_code=400, detail={"error": "image file is empty"})
    if len(data) > MAX_IMAGE_REFERENCE_BYTES:
        raise HTTPException(status_code=400, detail={"error": "单张图片不能超过 20MB"})
    next_total = total_size + len(data)
    if next_total > MAX_IMAGE_TOTAL_BYTES:
        raise HTTPException(status_code=400, detail={"error": "一次请求的图片总大小不能超过 50MB"})
    return next_total


def close_image_sources(sources: list[ImageSource]) -> None:
    """幂等关闭所有上传文件，覆盖排队拒绝、取消和中途校验失败路径。"""

    for source in sources:
        if not _is_upload(source):
            continue
        try:
            source.file.close()
        except Exception:
            pass


def read_image_sources_sync(
    sources: list[ImageSource],
    deadline: ImageTaskDeadline | None = None,
    max_count: int = MAX_IMAGE_COUNT,
) -> list[ImageInput]:
    """在图片专用线程读取上传/URL，避免占用 FastAPI 公共线程池。"""

    try:
        if len(sources) > max_count:
            raise HTTPException(status_code=400, detail={"error": f"一次请求最多支持 {max_count} 张图片"})
        images: list[ImageInput] = []
        total_size = 0
        for index, source in enumerate(sources, start=1):
            if deadline is not None:
                deadline.check("读取输入图片")
            if isinstance(source, tuple):
                total_size = _validate_image_input(source[0], total_size=total_size)
                images.append(source)
                continue
            if isinstance(source, Base64ImageSource):
                image = _decode_base64_image(source.encoded, source.filename, source.mime_type)
                total_size = _validate_image_input(image[0], total_size=total_size)
                images.append(image)
                continue
            if _is_upload(source):
                # 多读一个字节即可判定超限，避免把任意大文件完整读入内存。
                image_data = source.file.read(MAX_IMAGE_REFERENCE_BYTES + 1)
                total_size = _validate_image_input(image_data, total_size=total_size)
                images.append((image_data, source.filename or "image.png", source.content_type or "image/png"))
                continue
            image = _download_image_url(source, deadline)
            if source.strip().startswith("data:") and image[1].startswith("image_url."):
                # data URL 没有原始文件名，按请求顺序生成稳定且不重复的名称，方便
                # 多图编辑时定位上游日志；远程 URL 仍优先保留自身路径文件名。
                extension = image[1].rsplit(".", 1)[-1]
                image = (image[0], f"image_{index}.{extension}", image[2])
            total_size = _validate_image_input(image[0], total_size=total_size)
            images.append(image)
        if not images:
            raise HTTPException(
                status_code=400,
                detail={"error": "image file is required; alternatively provide image_url"},
            )
        return images
    finally:
        close_image_sources(sources)


async def read_image_sources(
    sources: list[ImageSource],
    deadline: ImageTaskDeadline | None = None,
) -> list[ImageInput]:
    """兼容旧调用；新图片 API 应把同步读取函数提交到图片专用执行器。"""

    return await run_in_threadpool(read_image_sources_sync, sources, deadline)

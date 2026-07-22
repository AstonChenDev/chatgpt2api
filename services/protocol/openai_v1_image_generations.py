from __future__ import annotations

import base64
import binascii
from typing import Any, Iterator

from fastapi import HTTPException

from services.image_task_runtime import image_deadline_from_payload
from services.protocol.conversation import (
    ConversationRequest,
    collect_image_outputs,
    count_text_tokens,
    stream_image_chunks,
    stream_image_outputs_with_pool,
)
from utils.image_tokens import count_image_output_items_tokens, image_usage


def _decode_images(raw: Any) -> list[str]:
    """Normalize image field into a list of image references.

    Each item may be a base64 string, a data-url, or an HTTP(S) URL.
    Data-url headers are stripped; URLs and plain base64 are passed through as-is.
    """
    if not raw:
        return []
    items = raw if isinstance(raw, list) else [raw]
    if len(items) > 4:
        raise HTTPException(status_code=400, detail={"error": "一次请求最多支持 4 张输入图片"})
    result = []
    total_decoded_bytes = 0
    for item in items:
        value = str(item or "").strip()
        if not value:
            continue
        # strip data-url header if present
        if value.startswith("data:") and "," in value:
            value = value.split(",", 1)[1]
        if not value.startswith(("http://", "https://")):
            # 先按编码长度估算，避免对异常超大字符串先分配完整解码缓冲区。
            if len(value) > ((20 * 1024 * 1024 + 2) // 3) * 4 + 4:
                raise HTTPException(status_code=400, detail={"error": "单张输入图片不能超过 20MB"})
            try:
                decoded = base64.b64decode(value, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise HTTPException(status_code=400, detail={"error": "输入图片不是有效的 base64"}) from exc
            if len(decoded) > 20 * 1024 * 1024:
                raise HTTPException(status_code=400, detail={"error": "单张输入图片不能超过 20MB"})
            total_decoded_bytes += len(decoded)
            if total_decoded_bytes > 50 * 1024 * 1024:
                raise HTTPException(status_code=400, detail={"error": "输入图片总大小不能超过 50MB"})
        result.append(value)
    return result


def handle(body: dict[str, Any]) -> dict[str, Any] | Iterator[dict[str, Any]]:
    prompt = str(body.get("prompt") or "")
    model = str(body.get("model") or "gpt-image-2")
    n = int(body.get("n") or 1)
    size = body.get("size")
    quality = str(body.get("quality") or "auto")
    response_format = str(body.get("response_format") or "b64_json")
    base_url = str(body.get("base_url") or "") or None
    progress_callback = body.get("progress_callback")
    images = _decode_images(body.get("image"))
    outputs = stream_image_outputs_with_pool(ConversationRequest(
        prompt=prompt,
        model=model,
        n=n,
        size=size,
        quality=quality,
        response_format=response_format,
        base_url=base_url,
        images=images or None,
        message_as_error=True,
        progress_callback=progress_callback,
        # 由 API 入口创建的同一截止线贯穿排队、生成、下载与存储。
        deadline=image_deadline_from_payload(body),
    ))
    if body.get("stream"):
        return stream_image_chunks(outputs)
    result = collect_image_outputs(outputs)
    result["usage"] = image_usage(
        input_text_tokens=count_text_tokens(prompt, model),
        output_tokens=count_image_output_items_tokens(result.get("data"), size, quality),
    )
    return result

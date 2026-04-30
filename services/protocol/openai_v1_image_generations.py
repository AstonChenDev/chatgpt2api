from __future__ import annotations

import base64
from typing import Any, Iterator

from services.protocol.conversation import (
    ConversationRequest,
    collect_image_outputs,
    count_text_tokens,
    stream_image_chunks,
    stream_image_outputs_with_pool,
)
from utils.image_tokens import count_image_output_items_tokens, image_usage


def _decode_images(raw: Any) -> list[str]:
    """Decode image field (single base64 string or list) into base64 strings."""
    if not raw:
        return []
    items = raw if isinstance(raw, list) else [raw]
    result = []
    for item in items:
        value = str(item or "").strip()
        if not value:
            continue
        # strip data-url header if present
        if value.startswith("data:") and "," in value:
            value = value.split(",", 1)[1]
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
    ))
    if body.get("stream"):
        return stream_image_chunks(outputs)
    result = collect_image_outputs(outputs)
    result["usage"] = image_usage(
        input_text_tokens=count_text_tokens(prompt, model),
        output_tokens=count_image_output_items_tokens(result.get("data"), size, quality),
    )
    return result

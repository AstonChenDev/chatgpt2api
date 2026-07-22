from __future__ import annotations

import asyncio
import json
import time
from typing import Any


IMAGE_BODY_PATHS = {
    "/v1/images/generations",
    "/v1/images/edits",
    "/v1/chat/completions",
    "/v1/responses",
    "/v1/messages",
    "/api/image-tasks/generations",
    "/api/image-tasks/edits",
    "/v1/ppt/generations",
    "/v1/psd/generations",
}
EDITABLE_FILE_BODY_PATHS = {
    "/v1/ppt/generations",
    "/v1/psd/generations",
}
MAX_IMAGE_REQUEST_BODY_BYTES = 75 * 1024 * 1024
MAX_EDITABLE_FILE_REQUEST_BODY_BYTES = 32 * 1024 * 1024
LARGE_BODY_THRESHOLD_BYTES = 1024 * 1024
MAX_CONCURRENT_LARGE_BODIES = 2
REQUEST_BODY_TIMEOUT_SECS = 120.0


class _BodyGuardError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class ImageRequestBodyGuardMiddleware:
    """在 FastAPI/Pydantic 分配大块内存前限制图片请求体。

    不能只检查 Content-Length：chunked 请求或伪造长度仍可能绕过。因此 receive
    每块都会累计实际字节数；超过 1MB 的请求还会持有独立许可直到响应结束，防止
    大量 base64/multipart 请求在图片运行时队列之前同时驻留内存。
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        self._large_body_slots = asyncio.Semaphore(MAX_CONCURRENT_LARGE_BODIES)

    async def __call__(self, scope: dict[str, Any], receive, send) -> None:
        path = str(scope.get("path") or "")
        method = str(scope.get("method") or "").upper()
        if scope.get("type") != "http" or method not in {"POST", "PUT", "PATCH"} or path not in IMAGE_BODY_PATHS:
            await self.app(scope, receive, send)
            return

        header_map = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers") or []
        }
        declared_length = self._parse_content_length(header_map.get("content-length"))
        body_limit = MAX_IMAGE_REQUEST_BODY_BYTES
        if path in EDITABLE_FILE_BODY_PATHS:
            body_limit = min(body_limit, MAX_EDITABLE_FILE_REQUEST_BODY_BYTES)
        # 提示使用产品约定值，不受单元测试临时缩小数值阈值影响。
        body_limit_label = "32MB" if path in EDITABLE_FILE_BODY_PATHS else "75MB"
        if declared_length is not None and declared_length > body_limit:
            await self._send_error(scope, receive, send, 413, f"图片请求体不能超过 {body_limit_label}")
            return

        acquired = False
        received = 0
        started_at = time.monotonic()
        response_started = False
        from services.config import config

        request_timeout_secs = min(
            REQUEST_BODY_TIMEOUT_SECS,
            float(config.get_image_task_runtime_settings()["total_timeout_secs"]),
        )
        # 后续图片 handler 复用这个单调时钟起点，让“总时限”覆盖上传和
        # FastAPI/Pydantic 解析请求体的时间，而不是进入业务函数后重新计时。
        scope.setdefault("state", {})["image_request_started_at"] = started_at

        async def acquire_large_slot() -> None:
            nonlocal acquired
            if not acquired:
                # 大请求不在中间件里排队数分钟：内存保护槽满时快速失败，让客户端重试。
                remaining = request_timeout_secs - (time.monotonic() - started_at)
                try:
                    await asyncio.wait_for(self._large_body_slots.acquire(), timeout=max(0.01, min(1.0, remaining)))
                except asyncio.TimeoutError as exc:
                    raise _BodyGuardError(429, "大图片请求正在处理中，请稍后重试") from exc
                acquired = True

        if declared_length is not None and declared_length > LARGE_BODY_THRESHOLD_BYTES:
            try:
                await acquire_large_slot()
            except _BodyGuardError as exc:
                await self._send_error(scope, receive, send, exc.status_code, str(exc))
                return

        async def guarded_receive():
            nonlocal received
            remaining = request_timeout_secs - (time.monotonic() - started_at)
            if remaining <= 0:
                raise _BodyGuardError(408, "图片请求体上传超时")
            try:
                message = await asyncio.wait_for(receive(), timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise _BodyGuardError(408, "图片请求体上传超时") from exc
            if message.get("type") == "http.request":
                received += len(message.get("body") or b"")
                if received > body_limit:
                    raise _BodyGuardError(413, f"图片请求体不能超过 {body_limit_label}")
                if received > LARGE_BODY_THRESHOLD_BYTES:
                    await acquire_large_slot()
            return message

        async def guarded_send(message):
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, guarded_receive, guarded_send)
        except _BodyGuardError as exc:
            if not response_started:
                await self._send_error(scope, receive, send, exc.status_code, str(exc))
        finally:
            if acquired:
                self._large_body_slots.release()

    @staticmethod
    def _parse_content_length(value: str | None) -> int | None:
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return max(0, parsed)

    @staticmethod
    async def _send_error(scope, receive, send, status_code: int, message: str) -> None:
        error_type = "invalid_request_error"
        error_code = "request_body_too_large"
        if status_code == 408:
            error_type = "request_timeout"
            error_code = "request_body_timeout"
        elif status_code == 429:
            error_type = "rate_limit_error"
            error_code = "image_body_busy"
        payload = json.dumps(
            {"error": {"message": message, "type": error_type, "code": error_code}},
            ensure_ascii=False,
        ).encode("utf-8")
        headers = [(b"content-type", b"application/json; charset=utf-8"), (b"content-length", str(len(payload)).encode())]
        if status_code == 429:
            headers.append((b"retry-after", b"5"))
        await send({"type": "http.response.start", "status": status_code, "headers": headers})
        await send({"type": "http.response.body", "body": payload})

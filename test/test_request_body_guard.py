from __future__ import annotations

import asyncio
import json
import time
import unittest
from collections.abc import Awaitable, Callable
from typing import Any
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.request_body_guard as guard_module
import api.system as system_module


Receive = Callable[[], Awaitable[dict[str, Any]]]


def _scope(
    *,
    request_id: str,
    content_length: int | None,
    path: str = "/v1/images/edits",
) -> dict[str, Any]:
    headers = [(b"x-request-id", request_id.encode("ascii"))]
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode("ascii")))
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }


def _receive_from_chunks(*chunks: bytes) -> Receive:
    messages = [
        {
            "type": "http.request",
            "body": chunk,
            "more_body": index < len(chunks) - 1,
        }
        for index, chunk in enumerate(chunks)
    ]

    async def receive() -> dict[str, Any]:
        if messages:
            return messages.pop(0)
        return {"type": "http.disconnect"}

    return receive


def _send_into(messages: list[dict[str, Any]]):
    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    return send


def _response_status(messages: list[dict[str, Any]]) -> int:
    start = next(message for message in messages if message["type"] == "http.response.start")
    return int(start["status"])


def _response_headers(messages: list[dict[str, Any]]) -> dict[str, str]:
    start = next(message for message in messages if message["type"] == "http.response.start")
    return {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in start.get("headers") or []
    }


def _response_json(messages: list[dict[str, Any]]) -> dict[str, Any]:
    body = b"".join(
        message.get("body") or b""
        for message in messages
        if message["type"] == "http.response.body"
    )
    return json.loads(body)


class ImageRequestBodyGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_editable_file_routes_are_guarded_before_pydantic_parsing(self) -> None:
        app_called = False

        async def should_not_run(scope, receive, send) -> None:
            nonlocal app_called
            app_called = True

        with mock.patch.object(guard_module, "MAX_IMAGE_REQUEST_BODY_BYTES", 32):
            for path in ("/v1/ppt/generations", "/v1/psd/generations"):
                with self.subTest(path=path):
                    middleware = guard_module.ImageRequestBodyGuardMiddleware(should_not_run)
                    sent: list[dict[str, Any]] = []
                    await middleware(
                        _scope(request_id=path.rsplit("/", 2)[-2], content_length=33, path=path),
                        _receive_from_chunks(b""),
                        _send_into(sent),
                    )
                    self.assertEqual(_response_status(sent), 413)
                    error = _response_json(sent)["error"]
                    self.assertEqual(error["code"], "request_body_too_large")
                    self.assertIn("32MB", error["message"])

        self.assertFalse(app_called)

    async def test_records_request_start_for_end_to_end_image_deadline(self) -> None:
        captured_started_at: list[float] = []

        async def inspect_scope_app(scope, receive, send) -> None:
            captured_started_at.append(scope["state"]["image_request_started_at"])
            await send({"type": "http.response.start", "status": 204, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        before = time.monotonic()
        middleware = guard_module.ImageRequestBodyGuardMiddleware(inspect_scope_app)
        sent: list[dict[str, Any]] = []
        await middleware(
            _scope(request_id="deadline", content_length=0),
            _receive_from_chunks(b""),
            _send_into(sent),
        )

        self.assertEqual(_response_status(sent), 204)
        self.assertEqual(len(captured_started_at), 1)
        self.assertGreaterEqual(captured_started_at[0], before)
        self.assertLessEqual(captured_started_at[0], time.monotonic())

    async def test_actual_accumulated_body_over_limit_returns_413_even_with_false_or_missing_length(self) -> None:
        async def consume_body_app(scope, receive, send) -> None:
            while True:
                message = await receive()
                if message.get("type") != "http.request" or not message.get("more_body", False):
                    break
            await send({"type": "http.response.start", "status": 204, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        # 缩小测试阈值，但仍通过多个 ASGI 数据块累计，覆盖生产中的真实计数逻辑。
        with (
            mock.patch.object(guard_module, "MAX_IMAGE_REQUEST_BODY_BYTES", 32),
            mock.patch.object(guard_module, "LARGE_BODY_THRESHOLD_BYTES", 8),
        ):
            for label, declared_length in (("false", 1), ("missing", None)):
                with self.subTest(content_length=label):
                    middleware = guard_module.ImageRequestBodyGuardMiddleware(consume_body_app)
                    sent: list[dict[str, Any]] = []
                    await middleware(
                        _scope(request_id=label, content_length=declared_length),
                        _receive_from_chunks(b"a" * 20, b"b" * 20),
                        _send_into(sent),
                    )

                    self.assertEqual(_response_status(sent), 413)
                    error = _response_json(sent)["error"]
                    self.assertEqual(error["code"], "request_body_too_large")
                    self.assertIn("75MB", error["message"])
                    # 超限路径在获取过大请求许可后也必须归还，不能逐渐耗尽槽位。
                    self.assertEqual(middleware._large_body_slots._value, 2)

    async def test_third_large_request_fails_fast_and_released_slot_can_be_reused(self) -> None:
        entered = {name: asyncio.Event() for name in ("first", "second", "fourth")}
        release = {name: asyncio.Event() for name in ("first", "second", "fourth")}

        async def holding_app(scope, receive, send) -> None:
            request_id = next(
                value.decode("ascii")
                for key, value in scope["headers"]
                if key.lower() == b"x-request-id"
            )
            while True:
                message = await receive()
                if message.get("type") != "http.request" or not message.get("more_body", False):
                    break
            entered[request_id].set()
            await release[request_id].wait()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        async def call(middleware, request_id: str) -> list[dict[str, Any]]:
            sent: list[dict[str, Any]] = []
            await middleware(
                _scope(request_id=request_id, content_length=9),
                _receive_from_chunks(b"x" * 9),
                _send_into(sent),
            )
            return sent

        with mock.patch.object(guard_module, "LARGE_BODY_THRESHOLD_BYTES", 8):
            middleware = guard_module.ImageRequestBodyGuardMiddleware(holding_app)
            first_task = asyncio.create_task(call(middleware, "first"))
            second_task = asyncio.create_task(call(middleware, "second"))
            await asyncio.wait_for(
                asyncio.gather(entered["first"].wait(), entered["second"].wait()),
                timeout=1,
            )

            started_at = time.monotonic()
            third_response = await call(middleware, "third")
            elapsed = time.monotonic() - started_at

            self.assertEqual(_response_status(third_response), 429)
            self.assertEqual(_response_headers(third_response).get("retry-after"), "5")
            self.assertGreaterEqual(elapsed, 0.75)
            self.assertLess(elapsed, 2.0)

            # 只结束第一个请求；它的 finally 应立即归还一个槽位，让第四个请求进入。
            release["first"].set()
            first_response = await asyncio.wait_for(first_task, timeout=1)
            self.assertEqual(_response_status(first_response), 200)

            fourth_task = asyncio.create_task(call(middleware, "fourth"))
            await asyncio.wait_for(entered["fourth"].wait(), timeout=1)
            release["second"].set()
            release["fourth"].set()
            second_response, fourth_response = await asyncio.wait_for(
                asyncio.gather(second_task, fourth_task),
                timeout=1,
            )
            self.assertEqual(_response_status(second_response), 200)
            self.assertEqual(_response_status(fourth_response), 200)
            self.assertEqual(middleware._large_body_slots._value, 2)


class _FakeImageRuntime:
    def status(self) -> dict[str, Any]:
        return {
            "healthy": True,
            "active_jobs": 0,
            "queued_jobs": 0,
            "max_concurrency": 8,
            "max_queue_size": 16,
        }


class HealthzTests(unittest.TestCase):
    def test_healthz_returns_lightweight_runtime_json(self) -> None:
        app = FastAPI()
        app.include_router(system_module.create_router("9.9.9-test"))

        with mock.patch("services.image_task_runtime.image_task_runtime", _FakeImageRuntime()):
            response = TestClient(app).get("/healthz")

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["healthy"])
        self.assertEqual(payload["version"], "9.9.9-test")
        self.assertEqual(payload["image_runtime"]["max_concurrency"], 8)


if __name__ == "__main__":
    unittest.main()

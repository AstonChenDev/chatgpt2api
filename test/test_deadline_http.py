from __future__ import annotations

import contextlib
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from curl_cffi import CurlOpt, requests
from curl_cffi.requests.exceptions import RequestException

from services.deadline_http import DeadlineSession
from services.image_task_runtime import ImageTaskDeadline, ImageTaskDeadlineExceeded


class DeadlineSessionTests(unittest.TestCase):
    def test_numeric_timeout_is_limited_by_remaining_total_budget(self) -> None:
        deadline = ImageTaskDeadline(0.5)
        session = DeadlineSession(image_deadline=deadline)
        captured: dict[str, object] = {}
        sentinel = object()

        def fake_request(instance, method, url, **kwargs):
            captured.update(kwargs)
            return sentinel

        with mock.patch.object(requests.Session, "request", new=fake_request):
            response = session.request("GET", "https://example.test", timeout=30)

        self.assertIs(response, sentinel)
        self.assertGreater(captured["timeout"], 0)  # type: ignore[operator]
        self.assertLessEqual(captured["timeout"], 0.5)  # type: ignore[operator]

    def test_tuple_timeout_is_limited_component_by_component(self) -> None:
        deadline = ImageTaskDeadline(0.5)
        session = DeadlineSession(image_deadline=deadline)
        captured: dict[str, object] = {}

        def fake_request(instance, method, url, **kwargs):
            captured.update(kwargs)
            return object()

        with mock.patch.object(requests.Session, "request", new=fake_request):
            session.request("POST", "https://example.test", timeout=(10, 20))

        timeout = captured["timeout"]
        self.assertIsInstance(timeout, tuple)
        self.assertEqual(len(timeout), 2)  # type: ignore[arg-type]
        self.assertTrue(all(0 < item <= 0.5 for item in timeout))  # type: ignore[union-attr]

    def test_stream_request_applies_and_then_restores_hard_curl_timeout(self) -> None:
        deadline = ImageTaskDeadline(0.5)
        session = DeadlineSession(image_deadline=deadline)
        session.curl_options[CurlOpt.TIMEOUT_MS] = 9999
        captured: dict[str, object] = {}

        def fake_request(instance, method, url, **kwargs):
            captured["timeout"] = kwargs.get("timeout")
            captured["hard_timeout_ms"] = instance.curl_options.get(CurlOpt.TIMEOUT_MS)
            return object()

        with mock.patch.object(requests.Session, "request", new=fake_request):
            session.request("GET", "https://example.test", timeout=30, stream=True)

        self.assertGreaterEqual(captured["hard_timeout_ms"], 50)  # type: ignore[operator]
        self.assertLessEqual(captured["hard_timeout_ms"], 500)  # type: ignore[operator]
        self.assertLessEqual(captured["timeout"], 0.5)  # type: ignore[operator]
        self.assertEqual(session.curl_options[CurlOpt.TIMEOUT_MS], 9999)

    def test_stream_request_removes_temporary_option_when_none_existed(self) -> None:
        session = DeadlineSession(image_deadline=ImageTaskDeadline(0.5))
        session.curl_options.pop(CurlOpt.TIMEOUT_MS, None)

        def fake_request(instance, method, url, **kwargs):
            self.assertIn(CurlOpt.TIMEOUT_MS, instance.curl_options)
            return object()

        with mock.patch.object(requests.Session, "request", new=fake_request):
            session.request("GET", "https://example.test", stream=True)

        self.assertNotIn(CurlOpt.TIMEOUT_MS, session.curl_options)

    def test_expired_deadline_fails_before_network_call(self) -> None:
        expired = ImageTaskDeadline(0.05, started_at=time.monotonic() - 1)
        session = DeadlineSession(image_deadline=expired)

        with mock.patch.object(requests.Session, "request") as parent_request:
            with self.assertRaises(ImageTaskDeadlineExceeded):
                session.request("GET", "https://example.test")
        parent_request.assert_not_called()

    def test_slow_stream_cannot_run_past_end_to_end_deadline(self) -> None:
        stop_stream = threading.Event()

        class SlowStreamHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 固定方法名
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                while not stop_stream.is_set():
                    with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                        self.wfile.write(b"x")
                        self.wfile.flush()
                    time.sleep(0.03)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), SlowStreamHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        started_at = time.monotonic()
        try:
            session = DeadlineSession(image_deadline=ImageTaskDeadline(0.25))
            with self.assertRaises(RequestException):
                response = session.get(
                    f"http://127.0.0.1:{server.server_port}/",
                    stream=True,
                )
                for _chunk in response.iter_content():
                    pass
            elapsed = time.monotonic() - started_at
            self.assertGreaterEqual(elapsed, 0.15)
            self.assertLess(elapsed, 0.8)
        finally:
            stop_stream.set()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()

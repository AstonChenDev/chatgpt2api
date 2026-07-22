from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest import mock

from fastapi.responses import JSONResponse, StreamingResponse

from services.config import config
from services.image_task_runtime import (
    ImageTaskDeadline,
    ImageTaskDeadlineExceeded,
    ImageTaskOverloadedError,
    ImageTaskQueueTimeoutError,
    ImageTaskRuntime,
)
from services.image_task_service import ImageTaskService
from services.log_service import LoggedCall
from services.openai_backend_api import ImagePollTimeoutError
from services.protocol import conversation


OWNER = {"id": "owner-stability", "name": "Stability", "role": "admin"}


def _runtime_settings() -> dict[str, int | float]:
    """提供不经过生产配置下限裁剪的短测试配置。"""

    return {
        "total_timeout_secs": 5,
        "max_concurrency": 1,
        "max_queue_size": 2,
        "queue_timeout_secs": 1,
    }


def _logged_image_call() -> LoggedCall:
    return LoggedCall(
        identity=OWNER,
        endpoint="/v1/images/generations",
        model="gpt-image-2",
        summary="稳定性测试",
    )


def _json_response_body(response: JSONResponse) -> dict[str, object]:
    return json.loads(response.body.decode("utf-8"))


class LoggedImageCallRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_stream_immediate_overload_is_returned_as_429(self) -> None:
        runtime = mock.Mock()
        runtime.submit.side_effect = ImageTaskOverloadedError("测试队列已满")
        call = _logged_image_call()

        with (
            mock.patch("services.image_task_runtime.image_task_runtime", runtime),
            mock.patch.object(call, "log") as log,
        ):
            started_at = time.monotonic()
            response = await call.run_image(
                lambda: {"data": []},
                deadline=ImageTaskDeadline(5),
            )
            elapsed = time.monotonic() - started_at

        self.assertIsInstance(response, JSONResponse)
        self.assertEqual(response.status_code, 429)
        self.assertLess(elapsed, 0.5)
        self.assertEqual(_json_response_body(response)["error"]["code"], "image_queue_full")  # type: ignore[index]
        runtime.submit.assert_called_once()
        log.assert_called_once()

    async def test_non_stream_deadline_keeps_the_exact_worker_phase(self) -> None:
        runtime = ImageTaskRuntime(_runtime_settings)  # type: ignore[arg-type]
        deadline = ImageTaskDeadline(5)
        call = _logged_image_call()

        def fail_in_storage() -> dict[str, object]:
            raise ImageTaskDeadlineExceeded(deadline.timeout_secs, "COS 图片持久化")

        with (
            mock.patch("services.image_task_runtime.image_task_runtime", runtime),
            mock.patch.object(call, "log"),
        ):
            response = await call.run_image(fail_in_storage, deadline=deadline)

        self.assertIsInstance(response, JSONResponse)
        self.assertEqual(response.status_code, 504)
        payload = _json_response_body(response)
        self.assertEqual(payload["error"]["code"], "image_generation_timeout")  # type: ignore[index]
        self.assertIn("COS 图片持久化", payload["error"]["message"])  # type: ignore[index]
        self.assertNotIn("图片生成超过", payload["error"]["message"])  # type: ignore[index]

    async def test_stream_immediate_error_is_returned_before_http_200(self) -> None:
        runtime = ImageTaskRuntime(_runtime_settings)  # type: ignore[arg-type]
        deadline = ImageTaskDeadline(5)
        call = _logged_image_call()

        def fail_before_first_chunk():
            raise ImageTaskDeadlineExceeded(deadline.timeout_secs, "上游图片流启动")
            yield {}  # pragma: no cover - 保持该函数为惰性生成器

        with (
            mock.patch("services.image_task_runtime.image_task_runtime", runtime),
            mock.patch.object(call, "log"),
        ):
            started_at = time.monotonic()
            response = await call.run_image(
                fail_before_first_chunk,
                stream=True,
                deadline=deadline,
            )
            elapsed = time.monotonic() - started_at

        self.assertIsInstance(response, JSONResponse)
        self.assertEqual(response.status_code, 504)
        self.assertLess(elapsed, 1.0)
        payload = _json_response_body(response)
        self.assertEqual(payload["error"]["code"], "image_generation_timeout")  # type: ignore[index]
        self.assertIn("上游图片流启动", payload["error"]["message"])  # type: ignore[index]

    async def test_stream_preflight_returns_in_about_one_second_without_waiting_for_first_chunk(self) -> None:
        runtime = ImageTaskRuntime(_runtime_settings)  # type: ignore[arg-type]
        deadline = ImageTaskDeadline(5)
        release_first_chunk = threading.Event()
        call = _logged_image_call()

        def slow_first_chunk():
            if not release_first_chunk.wait(3):
                raise AssertionError("测试未及时解除首帧等待")
            yield {"object": "image.generation.chunk", "data": []}

        with (
            mock.patch("services.image_task_runtime.image_task_runtime", runtime),
            mock.patch.object(call, "log"),
        ):
            started_at = time.monotonic()
            response = await call.run_image(
                slow_first_chunk,
                stream=True,
                deadline=deadline,
            )
            elapsed = time.monotonic() - started_at

            self.assertIsInstance(response, StreamingResponse)
            # 首项仍被上游阻塞时，接口只等待约 1 秒做错误预检，不能等完整出图。
            self.assertGreater(elapsed, 0.7)
            self.assertLess(elapsed, 1.6)

            release_first_chunk.set()
            chunks: list[str] = []
            async for chunk in response.body_iterator:
                chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)

        body = "".join(chunks)
        self.assertIn(": stream-open", body)
        self.assertIn("image.generation.chunk", body)
        self.assertIn("data: [DONE]", body)

    async def test_disconnect_on_stream_open_cancels_job_and_releases_slot(self) -> None:
        runtime = ImageTaskRuntime(_runtime_settings)  # type: ignore[arg-type]
        deadline = ImageTaskDeadline(5)
        call = _logged_image_call()

        def waits_before_first_chunk():
            deadline.sleep(4, "等待上游图片首帧")
            yield {"object": "image.generation.chunk", "data": []}

        with (
            mock.patch("services.image_task_runtime.image_task_runtime", runtime),
            mock.patch.object(call, "log"),
        ):
            response = await call.run_image(
                waits_before_first_chunk,
                stream=True,
                deadline=deadline,
            )
            self.assertIsInstance(response, StreamingResponse)

            iterator = response.body_iterator
            first = await anext(iterator)
            first_text = first.decode("utf-8") if isinstance(first, bytes) else first
            self.assertIn(": stream-open", first_text)
            await iterator.aclose()

        stop_at = time.monotonic() + 1
        while runtime.status()["active_units"] and time.monotonic() < stop_at:
            await asyncio.sleep(0.01)
        self.assertTrue(deadline.cancelled)
        self.assertEqual(runtime.status()["active_units"], 0)


class _BackgroundJob:
    def __init__(self) -> None:
        self.future: Future[None] = Future()

    def add_done_callback(self, callback) -> None:
        self.future.add_done_callback(callback)


class BackgroundImageTaskRuntimeRegressionTests(unittest.TestCase):
    def _service(self, path: Path, handler) -> ImageTaskService:
        return ImageTaskService(
            path,
            generation_handler=handler,
            edit_handler=handler,
            retention_days_getter=lambda: 30,
        )

    def test_background_task_is_submitted_to_the_bounded_image_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            handler = mock.Mock(return_value={"data": [{"url": "https://example.test/image.png"}]})
            service = self._service(Path(tmp_dir) / "tasks.json", handler)
            deadline = ImageTaskDeadline(5)
            job = _BackgroundJob()
            runtime = mock.Mock()
            runtime.new_deadline.return_value = deadline
            runtime.submit.return_value = job

            with mock.patch("services.image_task_service.image_task_runtime", runtime):
                result = service.submit_generation(
                    OWNER,
                    client_task_id="runtime-dispatch",
                    prompt="draw a cat",
                    model="gpt-image-2",
                    size=None,
                )

            self.assertEqual(result["status"], "queued")
            handler.assert_not_called()
            runtime.submit.assert_called_once()
            submit_call = runtime.submit.call_args
            submitted_handler = submit_call.args[0]
            self.assertIs(submitted_handler.__self__, service)
            self.assertIs(submitted_handler.__func__, ImageTaskService._run_task)
            self.assertEqual(submit_call.args[2], "generate")
            self.assertIs(submit_call.args[3]["_image_task_deadline"], deadline)
            self.assertIs(submit_call.kwargs["deadline"], deadline)
            self.assertEqual(submit_call.kwargs["weight"], 1)
            self.assertEqual(submit_call.kwargs["name"], "background-image-generate")

    def test_queue_timeout_before_worker_start_is_persisted_as_task_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "tasks.json"
            handler = mock.Mock(return_value={"data": [{"url": "https://example.test/image.png"}]})
            service = self._service(path, handler)
            job = _BackgroundJob()
            runtime = mock.Mock()
            runtime.new_deadline.return_value = ImageTaskDeadline(5)
            runtime.submit.return_value = job

            with mock.patch("services.image_task_service.image_task_runtime", runtime):
                service.submit_generation(
                    OWNER,
                    client_task_id="queued-timeout",
                    prompt="draw a cat",
                    model="gpt-image-2",
                    size=None,
                )
                job.future.set_exception(ImageTaskQueueTimeoutError("测试图片排队超时"))

            handler.assert_not_called()
            task = service.list_tasks(OWNER, ["queued-timeout"])["items"][0]
            self.assertEqual(task["status"], "error")
            self.assertIn("测试图片排队超时", task["error"])

            saved_task = json.loads(path.read_text(encoding="utf-8"))["tasks"][0]
            self.assertEqual(saved_task["status"], "error")
            self.assertIn("测试图片排队超时", saved_task["error"])

    def test_immediate_queue_rejection_is_persisted_before_being_raised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "tasks.json"
            service = self._service(path, mock.Mock())
            runtime = mock.Mock()
            runtime.new_deadline.return_value = ImageTaskDeadline(5)
            runtime.submit.side_effect = ImageTaskOverloadedError("测试图片队列已满")

            with (
                mock.patch("services.image_task_service.image_task_runtime", runtime),
                self.assertRaises(ImageTaskOverloadedError),
            ):
                service.submit_generation(
                    OWNER,
                    client_task_id="queue-full",
                    prompt="draw a cat",
                    model="gpt-image-2",
                    size=None,
                )

            task = service.list_tasks(OWNER, ["queue-full"])["items"][0]
            self.assertEqual(task["status"], "error")
            self.assertIn("测试图片队列已满", task["error"])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["tasks"][0]["status"], "error")


class _PollingTimeoutBackend:
    def __init__(self) -> None:
        self.poll_timeouts: list[float] = []
        self.closed = False

    def resolve_conversation_image_urls(
        self,
        conversation_id: str,
        file_ids: list[str],
        sediment_ids: list[str],
        *,
        poll_timeout_secs: float,
    ) -> list[str]:
        self.poll_timeouts.append(poll_timeout_secs)
        error = ImagePollTimeoutError("测试原会话轮询超时")
        setattr(error, "conversation_id", conversation_id)
        raise error

    def close(self) -> None:
        self.closed = True


class ConversationDeadlineRegressionTests(unittest.TestCase):
    def test_existing_conversation_uses_remaining_total_budget_and_does_not_switch_account(self) -> None:
        deadline = ImageTaskDeadline(300)
        request = conversation.ConversationRequest(
            model="gpt-image-2",
            prompt="draw a cat",
            n=1,
            deadline=deadline,
        )
        backend = _PollingTimeoutBackend()
        poll_event = {
            "type": "conversation.done",
            "conversation_id": "conv-existing-task",
            "file_ids": [],
            "sediment_ids": [],
            "text": "",
            "turn_use_case": "image gen",
        }

        with (
            mock.patch.dict(config.data, {"image_poll_timeout_secs": 120}),
            mock.patch.object(
                conversation.account_service,
                "get_available_access_token",
                side_effect=["token-one", "token-two"],
            ) as get_token,
            mock.patch.object(
                conversation.account_service,
                "get_account",
                return_value={"email": "account@example.test"},
            ),
            mock.patch.object(conversation.account_service, "mark_image_result") as mark_result,
            mock.patch.object(conversation, "OpenAIBackendAPI", return_value=backend) as backend_factory,
            mock.patch.object(conversation, "conversation_events", return_value=[poll_event]),
            mock.patch.object(conversation, "_get_detailed_error_from_tasks", return_value=""),
            self.assertRaises(ImagePollTimeoutError) as raised,
        ):
            conversation._generate_single_image(request, 1, 1)

        self.assertEqual(getattr(raised.exception, "conversation_id", ""), "conv-existing-task")
        self.assertEqual(get_token.call_count, 1)
        self.assertEqual(backend_factory.call_count, 1)
        mark_result.assert_called_once_with("token-one", False)
        self.assertTrue(backend.closed)
        self.assertEqual(len(backend.poll_timeouts), 1)
        # 300 秒总预算会预留最多 15 秒用于下载/持久化，因此应得到约 285 秒，
        # 而不能再被旧的 image_poll_timeout_secs=120 提前截断。
        self.assertGreater(backend.poll_timeouts[0], 120)
        self.assertGreater(backend.poll_timeouts[0], 280)
        self.assertLessEqual(backend.poll_timeouts[0], 300)


if __name__ == "__main__":
    unittest.main()

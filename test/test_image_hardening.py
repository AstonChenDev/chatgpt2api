from __future__ import annotations

import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException
from starlette.datastructures import UploadFile

from services.image_io_runtime import ImageIORuntime
from services.image_task_runtime import ImageTaskOverloadedError


class VisionInputBudgetTests(unittest.TestCase):
    def test_budget_is_shared_across_all_historical_messages(self) -> None:
        from services.protocol.conversation import normalize_messages

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "data": bytes([index + 1]), "mime": "image/png"}],
            }
            for index in range(5)
        ]
        with self.assertRaises(HTTPException) as raised:
            normalize_messages(messages)
        self.assertEqual(400, raised.exception.status_code)
        self.assertIn("最多支持 4 张", str(raised.exception.detail))

    def test_response_image_extraction_stops_after_first_consumed_image(self) -> None:
        from services.protocol.openai_v1_response import extract_response_image

        content = [
            {"type": "input_image", "image_url": f"https://example.test/{index}.png"}
            for index in range(3)
        ]
        with mock.patch(
            "utils.helper._decode_message_image_url",
            side_effect=[(b"first", "image/png"), (b"second", "image/png")],
        ) as fetch:
            result = extract_response_image({"content": content})
        self.assertEqual((b"first", "image/png"), result)
        self.assertEqual(1, fetch.call_count)


class OutputBoundaryTests(unittest.TestCase):
    def test_oversized_base64_is_rejected_before_decode_or_storage(self) -> None:
        from services.protocol.conversation import (
            ImageGenerationError,
            MAX_OUTPUT_IMAGE_BASE64_CHARS,
            format_image_result,
        )

        with self.assertRaises(ImageGenerationError):
            format_image_result(
                [{"b64_json": "A" * (MAX_OUTPUT_IMAGE_BASE64_CHARS + 1)}],
                "prompt",
                "b64_json",
            )

    def test_image_poll_timeout_has_gateway_timeout_semantics(self) -> None:
        from services.openai_backend_api import ImagePollTimeoutError

        error = ImagePollTimeoutError("poll timeout")
        self.assertEqual(504, error.status_code)
        self.assertEqual("image_poll_timeout", error.to_openai_error()["error"]["code"])


class DiskSpaceGuardTests(unittest.TestCase):
    def test_atomic_write_refuses_to_cross_reserved_free_space(self) -> None:
        from services.disk_space_guard import InsufficientDiskSpaceError, atomic_write_bytes

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "image.png"
            with (
                mock.patch("services.disk_space_guard.config") as disk_config,
                mock.patch(
                    "services.disk_space_guard.shutil.disk_usage",
                    return_value=SimpleNamespace(free=500 * 1024 * 1024),
                ),
            ):
                disk_config.image_min_free_mb = 500
                with self.assertRaises(InsufficientDiskSpaceError):
                    atomic_write_bytes(target, b"payload")
            self.assertFalse(target.exists())
            self.assertEqual([], list(Path(directory).glob("*.part")))

    def test_local_image_save_fails_fast_without_leaving_partial_file(self) -> None:
        from services.image_storage_service import ImageStorageError, ImageStorageService

        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            images_dir = data_dir / "images"
            with (
                mock.patch("services.image_storage_service.config") as storage_config,
                mock.patch("services.disk_space_guard.config") as disk_config,
                mock.patch(
                    "services.disk_space_guard.shutil.disk_usage",
                    return_value=SimpleNamespace(free=500 * 1024 * 1024),
                ),
            ):
                storage_config.images_dir = images_dir
                storage_config.base_url = "http://app.test"
                storage_config.cleanup_old_images.return_value = 0
                storage_config.get_image_storage_settings.return_value = {
                    "enabled": False,
                    "mode": "local",
                }
                disk_config.image_min_free_mb = 500
                with self.assertRaises(ImageStorageError):
                    ImageStorageService(data_dir / "image_index.json").save(
                        b"not-important-for-this-boundary-test",
                        "http://app.test",
                    )
            self.assertEqual([], [path for path in images_dir.rglob("*") if path.is_file()])


class UploadCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_immediate_runtime_rejection_closes_upload(self) -> None:
        from services.log_service import LoggedCall

        upload = UploadFile(io.BytesIO(b"payload"), filename="input.png")
        call = LoggedCall({"id": "test"}, "/v1/images/edits", "gpt-image-2", "图生图")
        with (
            mock.patch(
                "services.image_task_runtime.image_task_runtime.submit",
                side_effect=ImageTaskOverloadedError(),
            ),
            mock.patch.object(call, "log"),
        ):
            await call.run_image(
                lambda: {},
                cleanup=upload.file.close,
            )
        self.assertTrue(upload.file.closed)


class ImageIORuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_timed_out_queued_operation_is_cancelled_before_execution(self) -> None:
        runtime = ImageIORuntime(max_workers=1, max_queue_size=1)
        release = __import__("threading").Event()
        second_called = __import__("threading").Event()

        first = asyncio.create_task(runtime.run(release.wait, 2.0, timeout_secs=1.0))
        await asyncio.sleep(0.03)
        with self.assertRaises(HTTPException) as raised:
            await runtime.run(second_called.set, timeout_secs=0.05)
        self.assertEqual(504, raised.exception.status_code)
        release.set()
        await first
        await asyncio.sleep(0.03)
        self.assertFalse(second_called.is_set())
        runtime._executor.shutdown(wait=True, cancel_futures=True)


class RuntimeWatchdogTests(unittest.TestCase):
    def test_persistently_unhealthy_component_requests_process_restart(self) -> None:
        from services.runtime_watchdog import _watchdog_loop

        class FastEvent:
            def wait(self, _timeout: float) -> bool:
                return False

        exit_codes: list[int] = []
        with mock.patch("services.runtime_watchdog.time.monotonic", side_effect=[0.0, 20.0]):
            _watchdog_loop(
                FastEvent(),  # type: ignore[arg-type]
                grace_secs=15.0,
                status_getter=lambda: {"image_runtime": {"healthy": False, "overdue_jobs": 1}},
                exit_process=exit_codes.append,
            )
        self.assertEqual([70], exit_codes)


class EditableArtifactDownloadTests(unittest.TestCase):
    class _Response:
        status_code = 200
        headers = {"Content-Type": "application/zip"}
        url = "https://example.test/result.zip"

        def __init__(self, chunks: list[bytes]) -> None:
            self.chunks = chunks
            self.closed = False

        @property
        def content(self):  # pragma: no cover - 访问即表示回归到整包内存读取
            raise AssertionError("response.content must not be used")

        def iter_content(self, chunk_size: int):
            self.requested_chunk_size = chunk_size
            yield from self.chunks

        def close(self) -> None:
            self.closed = True

    class _Session:
        def __init__(self, response) -> None:
            self.response = response

        def get(self, *_args, **kwargs):
            self.kwargs = kwargs
            return self.response

    def _backend(self, response):
        from services.image_task_runtime import ImageTaskDeadline
        from services.openai_backend_api import OpenAIBackendAPI

        backend = object.__new__(OpenAIBackendAPI)
        backend.image_deadline = ImageTaskDeadline(5)
        backend.session = self._Session(response)
        backend._resolve_editable_download_url = lambda *_args: "https://example.test/result.zip"
        backend._resolve_editable_output_name = lambda *_args: "result.zip"
        backend._unique_editable_path = lambda path: path
        return backend

    def test_artifact_is_streamed_to_atomic_part_file(self) -> None:
        from services.openai_backend_api import EditableFileArtifact

        response = self._Response([b"abc", b"def"])
        backend = self._backend(response)
        with tempfile.TemporaryDirectory() as directory:
            output = backend._download_editable_artifact(
                "conversation",
                EditableFileArtifact(file_id="file"),
                Path(directory),
                set(),
                (),
                ".zip",
                max_bytes=10,
            )
            self.assertEqual(b"abcdef", output.read_bytes())
            self.assertFalse(output.with_name(output.name + ".part").exists())
        self.assertTrue(response.closed)

    def test_oversized_artifact_removes_partial_file(self) -> None:
        from services.openai_backend_api import EditableFileArtifact

        response = self._Response([b"abc", b"def"])
        backend = self._backend(response)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                backend._download_editable_artifact(
                    "conversation",
                    EditableFileArtifact(file_id="file"),
                    Path(directory),
                    set(),
                    (),
                    ".zip",
                    max_bytes=5,
                )
            self.assertEqual([], list(Path(directory).iterdir()))
        self.assertTrue(response.closed)


if __name__ == "__main__":
    unittest.main()

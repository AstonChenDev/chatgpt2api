from __future__ import annotations

import base64
import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor as TestThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

import api.ai as ai_module
import services.editable_file_task_service as service_module
from services.editable_file_task_service import (
    MAX_EDITABLE_PROMPT_CHARS,
    EditableFileTaskOverloadedError,
    EditableFileTaskService,
    EditableFileTaskUnavailableError,
)
from services.image_task_runtime import ImageTaskDeadline


IDENTITY = {"id": "test-owner", "name": "测试", "role": "admin"}


def _tiny_png_data_uri() -> str:
    output = BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _wait_for_task(
    service: EditableFileTaskService,
    task_id: str,
    statuses: set[str],
    *,
    identity: dict[str, object] = IDENTITY,
    timeout: float = 2.0,
) -> dict[str, object]:
    expires_at = time.monotonic() + timeout
    while time.monotonic() < expires_at:
        payload = service.list_tasks(identity, [task_id])
        item = payload["items"][0]
        if item["status"] in statuses:
            return item
        time.sleep(0.01)
    raise AssertionError(f"任务 {task_id} 未在 {timeout} 秒内进入 {statuses}")


def _wait_for_idle(service: EditableFileTaskService, *, timeout: float = 2.0) -> None:
    expires_at = time.monotonic() + timeout
    while time.monotonic() < expires_at:
        if service.runtime_status()["admitted_jobs"] == 0:
            return
        time.sleep(0.01)
    raise AssertionError(f"任务服务未恢复空闲：{service.runtime_status()}")


class EditableFileTaskServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def _service(self, *, max_workers: int = 1, max_queue_size: int = 2) -> EditableFileTaskService:
        service = EditableFileTaskService(
            Path(self.temp_dir.name) / f"tasks-{time.monotonic_ns()}.json",
            max_workers=max_workers,
            max_queue_size=max_queue_size,
            enable_output_cleanup=False,
        )
        self.addCleanup(service.close)
        return service

    def test_rejects_unsafe_or_unbounded_input_before_admission(self) -> None:
        service = self._service()
        valid_image = _tiny_png_data_uri()

        with self.assertRaisesRegex(ValueError, "client_task_id"):
            service.submit_ppt(IDENTITY, client_task_id="../escape", prompt="x")
        with self.assertRaisesRegex(ValueError, "至少需要 1 张"):
            service.submit_psd(IDENTITY, client_task_id="psd-empty", prompt="x")
        with self.assertRaisesRegex(ValueError, "有效的 base64"):
            service.submit_psd(IDENTITY, client_task_id="bad-base64", base64_images=["!!!"])
        with self.assertRaisesRegex(ValueError, "最多支持 4 张"):
            service.submit_ppt(
                IDENTITY,
                client_task_id="too-many",
                base64_images=[valid_image] * 5,
            )
        with self.assertRaisesRegex(ValueError, "prompt"):
            service.submit_ppt(
                IDENTITY,
                client_task_id="long-prompt",
                prompt="x" * (MAX_EDITABLE_PROMPT_CHARS + 1),
            )

        self.assertEqual(service.runtime_status()["admitted_jobs"], 0)
        self.assertEqual(service.list_tasks(IDENTITY, [])["items"], [])

    def test_fixed_executor_and_bounded_queue_reject_excess_work(self) -> None:
        service = self._service(max_workers=1, max_queue_size=1)
        first_started = service_module.threading.Event()
        release_first = service_module.threading.Event()
        self.addCleanup(release_first.set)

        def fake_run(key, kind, prompt, images, identity, base_url, deadline) -> None:
            if prompt == "first":
                first_started.set()
                release_first.wait(2)
            service._update_task(
                key,
                status=service_module.TASK_STATUS_SUCCESS,
                result={"ok": True},
                ended_ts=time.time(),
            )

        settings = {
            "total_timeout_secs": 300,
            "queue_timeout_secs": 5,
            "max_concurrency": 8,
            "max_queue_size": 16,
        }
        with (
            mock.patch.object(service_module.config, "get_image_task_runtime_settings", return_value=settings),
            mock.patch.object(service, "_run_task", side_effect=fake_run) as run_task,
        ):
            first = service.submit_ppt(IDENTITY, client_task_id="first", prompt="first")
            self.assertTrue(first_started.wait(1))
            second = service.submit_ppt(IDENTITY, client_task_id="second", prompt="second")
            with self.assertRaises(EditableFileTaskOverloadedError):
                service.submit_ppt(IDENTITY, client_task_id="third", prompt="third")

            self.assertEqual(first["status"], "queued")
            self.assertEqual(second["status"], "queued")
            status = service.runtime_status()
            self.assertEqual(status["admitted_jobs"], 2)
            self.assertEqual(status["rejected_jobs"], 1)

            release_first.set()
            _wait_for_idle(service)
            self.assertEqual(run_task.call_count, 2)

    def test_image_validation_is_bounded_by_admission_capacity(self) -> None:
        service = self._service(max_workers=1, max_queue_size=1)
        both_validating = service_module.threading.Event()
        release_validation = service_module.threading.Event()
        counter_lock = service_module.threading.Lock()
        self.addCleanup(release_validation.set)
        validating = 0
        max_validating = 0

        def blocking_validation(kind, images, deadline):
            nonlocal validating, max_validating
            with counter_lock:
                validating += 1
                max_validating = max(max_validating, validating)
                if validating == 2:
                    both_validating.set()
            try:
                release_validation.wait(2)
                return []
            finally:
                with counter_lock:
                    validating -= 1

        def fake_run(key, kind, prompt, images, identity, base_url, deadline) -> None:
            service._update_task(
                key,
                status=service_module.TASK_STATUS_SUCCESS,
                result={"ok": True},
                ended_ts=time.time(),
            )

        settings = {
            "total_timeout_secs": 300,
            "queue_timeout_secs": 5,
            "max_concurrency": 8,
            "max_queue_size": 16,
        }
        with (
            mock.patch.object(service_module.config, "get_image_task_runtime_settings", return_value=settings),
            mock.patch.object(service, "_validate_base64_images", side_effect=blocking_validation) as validate,
            mock.patch.object(service, "_run_task", side_effect=fake_run),
            TestThreadPoolExecutor(max_workers=2) as callers,
        ):
            first = callers.submit(
                service.submit_ppt,
                IDENTITY,
                client_task_id="validate-1",
                prompt="one",
            )
            second = callers.submit(
                service.submit_ppt,
                IDENTITY,
                client_task_id="validate-2",
                prompt="two",
            )
            self.assertTrue(both_validating.wait(1))
            with self.assertRaises(EditableFileTaskOverloadedError):
                service.submit_ppt(
                    IDENTITY,
                    client_task_id="validate-3",
                    prompt="three",
                )

            self.assertEqual(service.runtime_status()["validating_jobs"], 2)
            release_validation.set()
            first.result(timeout=1)
            second.result(timeout=1)

        _wait_for_idle(service)
        self.assertEqual(validate.call_count, 2)
        self.assertEqual(max_validating, 2)

    def test_runtime_status_fails_fast_when_persistence_lock_is_busy(self) -> None:
        service = self._service()
        lock_held = service_module.threading.Event()
        release_lock = service_module.threading.Event()
        self.addCleanup(release_lock.set)

        def hold_status_lock() -> None:
            with service._lock:
                lock_held.set()
                release_lock.wait(2)

        holder = service_module.threading.Thread(target=hold_status_lock, daemon=True)
        holder.start()
        self.assertTrue(lock_held.wait(1))

        started_at = time.monotonic()
        status = service.runtime_status()
        elapsed = time.monotonic() - started_at

        self.assertFalse(status["healthy"])
        self.assertTrue(status["status_lock_busy"])
        self.assertLess(elapsed, 0.2)

        # 轮询和新提交也要快速失败，不能让大量客户端把 AnyIO 公共线程都堵在锁上。
        started_at = time.monotonic()
        with self.assertRaises(EditableFileTaskUnavailableError):
            service.list_tasks(IDENTITY, [])
        self.assertLess(time.monotonic() - started_at, 0.5)

        started_at = time.monotonic()
        with self.assertRaises(EditableFileTaskUnavailableError):
            service.submit_ppt(IDENTITY, client_task_id="lock-busy", prompt="test")
        self.assertLess(time.monotonic() - started_at, 0.5)

        release_lock.set()
        holder.join(timeout=1)
        self.assertFalse(holder.is_alive())

    def test_queued_task_times_out_without_entering_backend(self) -> None:
        service = self._service(max_workers=1, max_queue_size=1)
        first_started = service_module.threading.Event()
        release_first = service_module.threading.Event()
        self.addCleanup(release_first.set)

        def fake_run(key, kind, prompt, images, identity, base_url, deadline) -> None:
            first_started.set()
            release_first.wait(2)
            service._update_task(
                key,
                status=service_module.TASK_STATUS_SUCCESS,
                result={"ok": True},
                ended_ts=time.time(),
            )

        settings = {
            "total_timeout_secs": 300,
            "queue_timeout_secs": 0.05,
            "max_concurrency": 8,
            "max_queue_size": 16,
        }
        with (
            mock.patch.object(service_module.config, "get_image_task_runtime_settings", return_value=settings),
            mock.patch.object(service, "_run_task", side_effect=fake_run) as run_task,
        ):
            service.submit_ppt(IDENTITY, client_task_id="running", prompt="running")
            self.assertTrue(first_started.wait(1))
            service.submit_ppt(IDENTITY, client_task_id="queued", prompt="queued")

            queued = _wait_for_task(service, "queued", {"error"})
            self.assertIn("排队超时", str(queued["error"]))
            self.assertEqual(run_task.call_count, 1)

            release_first.set()
            _wait_for_idle(service)
            self.assertEqual(run_task.call_count, 1)

    def test_watchdog_finishes_status_even_if_worker_ignores_deadline(self) -> None:
        service = self._service(max_workers=1, max_queue_size=0)
        worker_started = service_module.threading.Event()
        release_worker = service_module.threading.Event()
        self.addCleanup(release_worker.set)

        def stuck_run(*args, **kwargs) -> None:
            worker_started.set()
            release_worker.wait(2)

        deadline = ImageTaskDeadline(0.1)
        with mock.patch.object(service, "_run_task", side_effect=stuck_run):
            service.submit_ppt(
                IDENTITY,
                client_task_id="deadline",
                prompt="deadline",
                deadline=deadline,
            )
            self.assertTrue(worker_started.wait(1))
            failed = _wait_for_task(service, "deadline", {"error"})
            self.assertIn("总时限", str(failed["error"]))
            self.assertEqual(service.runtime_status()["active_jobs"], 1)

            release_worker.set()
            _wait_for_idle(service)
            final = service.list_tasks(IDENTITY, ["deadline"])["items"][0]
            self.assertEqual(final["status"], "error")

    def test_default_config_deadline_is_shared_with_review_and_backend(self) -> None:
        service = self._service()
        backend = mock.MagicMock()
        backend.export_ppt_zip.return_value = SimpleNamespace(
            conversation_id="conversation-1",
            primary_path=Path("/tmp/result.pptx"),
            zip_path=Path("/tmp/result.zip"),
        )
        settings = {
            "total_timeout_secs": 300,
            "queue_timeout_secs": 5,
            "max_concurrency": 8,
            "max_queue_size": 16,
        }

        with (
            mock.patch.object(service_module.config, "get_image_task_runtime_settings", return_value=settings),
            mock.patch.object(service_module, "check_request") as check_request,
            mock.patch.object(service_module, "_editable_access_token", return_value="token") as access_token,
            mock.patch.object(service_module.account_service, "get_account", return_value={"email": "test@example.com"}),
            mock.patch.object(service_module.account_service, "mark_text_used") as mark_text_used,
            mock.patch.object(service_module, "OpenAIBackendAPI", return_value=backend) as backend_factory,
            mock.patch.object(service_module, "_file_url", side_effect=["/files/result.pptx", "/files/result.zip"]),
            mock.patch.object(service_module.log_service, "add"),
        ):
            service.submit_ppt(IDENTITY, client_task_id="success", prompt="生成演示文稿")
            result = _wait_for_task(service, "success", {"success", "error"})
            _wait_for_idle(service)

        self.assertEqual(result["status"], "success", result)
        deadline = backend_factory.call_args.kwargs["image_deadline"]
        self.assertIsInstance(deadline, ImageTaskDeadline)
        self.assertEqual(deadline.timeout_secs, 300)
        check_request.assert_called_once_with("生成演示文稿", deadline=deadline)
        access_token.assert_called_once_with(deadline)
        export_kwargs = backend.export_ppt_zip.call_args.kwargs
        self.assertGreater(export_kwargs["timeout_secs"], 0)
        self.assertLessEqual(export_kwargs["timeout_secs"], 300)
        self.assertLessEqual(export_kwargs["poll_interval_secs"], 1)
        mark_text_used.assert_called_once_with("token")
        backend.close.assert_called_once_with()

    def test_failed_task_removes_partial_output_directory(self) -> None:
        service = self._service()
        files_root = Path(self.temp_dir.name) / "files"
        backend = mock.MagicMock()
        output_dirs: list[Path] = []

        def fail_after_partial_download(images, prompt, output_dir, **kwargs):
            path = Path(output_dir)
            output_dirs.append(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "partial.pptx").write_bytes(b"partial")
            raise RuntimeError("download failed")

        backend.export_ppt_zip.side_effect = fail_after_partial_download
        settings = {
            "total_timeout_secs": 300,
            "queue_timeout_secs": 5,
            "max_concurrency": 8,
            "max_queue_size": 16,
        }
        with (
            mock.patch.object(service_module, "EDITABLE_FILE_ROOT", files_root),
            mock.patch.object(service_module.config, "get_image_task_runtime_settings", return_value=settings),
            mock.patch.object(service_module, "check_request"),
            mock.patch.object(service_module, "_editable_access_token", return_value="token"),
            mock.patch.object(service_module.account_service, "get_account", return_value={}),
            mock.patch.object(service_module, "OpenAIBackendAPI", return_value=backend),
            mock.patch.object(service_module.log_service, "add"),
        ):
            service.submit_ppt(IDENTITY, client_task_id="partial", prompt="test")
            failed = _wait_for_task(service, "partial", {"error"})
            _wait_for_idle(service)

        self.assertIn("download failed", str(failed["error"]))
        self.assertEqual(len(output_dirs), 1)
        self.assertFalse(output_dirs[0].exists())
        backend.close.assert_called_once_with()

    def test_same_task_id_from_different_owners_uses_isolated_output_directories(self) -> None:
        service = self._service()
        files_root = Path(self.temp_dir.name) / "files"
        other_identity = {"id": "other-owner", "name": "另一个用户", "role": "user"}
        backend = mock.MagicMock()
        output_dirs: list[Path] = []

        def export_success(images, prompt, output_dir, **kwargs):
            path = Path(output_dir)
            output_dirs.append(path)
            path.mkdir(parents=True, exist_ok=True)
            primary = path / "result.pptx"
            archive = path / "result.zip"
            primary.write_bytes(b"pptx")
            archive.write_bytes(b"zip")
            return SimpleNamespace(
                conversation_id=f"conversation-{len(output_dirs)}",
                primary_path=primary,
                zip_path=archive,
            )

        backend.export_ppt_zip.side_effect = export_success
        settings = {
            "total_timeout_secs": 300,
            "queue_timeout_secs": 5,
            "max_concurrency": 8,
            "max_queue_size": 16,
        }
        with (
            mock.patch.object(service_module, "EDITABLE_FILE_ROOT", files_root),
            mock.patch.object(service_module.config, "get_image_task_runtime_settings", return_value=settings),
            mock.patch.object(service_module, "check_request"),
            mock.patch.object(service_module, "_editable_access_token", return_value="token"),
            mock.patch.object(service_module.account_service, "get_account", return_value={}),
            mock.patch.object(service_module.account_service, "mark_text_used"),
            mock.patch.object(service_module, "OpenAIBackendAPI", return_value=backend),
            mock.patch.object(service_module.log_service, "add"),
        ):
            service.submit_ppt(IDENTITY, client_task_id="same-id", prompt="owner one")
            service.submit_ppt(other_identity, client_task_id="same-id", prompt="owner two")
            first = _wait_for_task(service, "same-id", {"success", "error"})
            second = _wait_for_task(
                service,
                "same-id",
                {"success", "error"},
                identity=other_identity,
            )
            _wait_for_idle(service)

            expected_first = service_module._task_output_dir("ppt", "test-owner:same-id")
            expected_second = service_module._task_output_dir("ppt", "other-owner:same-id")

        self.assertEqual(first["status"], "success", first)
        self.assertEqual(second["status"], "success", second)
        self.assertEqual(set(output_dirs), {expected_first, expected_second})
        self.assertNotEqual(expected_first, expected_second)
        self.assertTrue(expected_first.is_dir())
        self.assertTrue(expected_second.is_dir())
        self.assertNotEqual(first["result"]["primary_url"], second["result"]["primary_url"])

    def test_expired_output_cleanup_only_deletes_old_first_level_task_directories(self) -> None:
        service = self._service()
        files_root = Path(self.temp_dir.name) / "files"
        ppt_root = files_root / "ppt"
        psd_root = files_root / "psd"
        old_ppt = ppt_root / "old-ppt-0123456789abcdef"
        fresh_ppt = ppt_root / "fresh-ppt-0123456789abcdef"
        old_psd = psd_root / "old-psd-0123456789abcdef"
        for path in (old_ppt, fresh_ppt, old_psd):
            path.mkdir(parents=True, exist_ok=True)
        nested = fresh_ppt / "nested-old"
        nested.mkdir()
        plain_file = ppt_root / "not-a-task.txt"
        plain_file.write_text("keep", encoding="utf-8")
        outside = Path(self.temp_dir.name) / "outside"
        outside.mkdir()
        symlink = ppt_root / "outside-link"
        symlink.symlink_to(outside, target_is_directory=True)

        now = time.time()
        expired_mtime = now - service_module.EDITABLE_TASK_RETENTION_SECS - 60
        os.utime(old_ppt, (expired_mtime, expired_mtime))
        os.utime(old_psd, (expired_mtime, expired_mtime))
        os.utime(nested, (expired_mtime, expired_mtime))

        lock_held = service_module.threading.Event()
        release_lock = service_module.threading.Event()
        self.addCleanup(release_lock.set)

        def hold_task_lock() -> None:
            with service._lock:
                lock_held.set()
                release_lock.wait(2)

        holder = service_module.threading.Thread(target=hold_task_lock, daemon=True)
        holder.start()
        self.assertTrue(lock_held.wait(1))
        started_at = time.monotonic()
        with mock.patch.object(service_module, "EDITABLE_FILE_ROOT", files_root):
            result = service.cleanup_expired_outputs(now=now)
        self.assertLess(time.monotonic() - started_at, 0.5)
        release_lock.set()
        holder.join(timeout=1)
        self.assertFalse(holder.is_alive())

        self.assertEqual(result["removed"], 2, result)
        self.assertFalse(old_ppt.exists())
        self.assertFalse(old_psd.exists())
        self.assertTrue(fresh_ppt.is_dir())
        self.assertTrue(nested.is_dir())
        self.assertTrue(plain_file.is_file())
        self.assertTrue(symlink.is_symlink())
        self.assertTrue(outside.is_dir())

    def test_restart_marks_unfinished_task_error_and_removes_partial_files(self) -> None:
        files_root = Path(self.temp_dir.name) / "files"
        tasks_path = Path(self.temp_dir.name) / "restart-tasks.json"
        with mock.patch.object(service_module, "EDITABLE_FILE_ROOT", files_root):
            output_dir = service_module._task_output_dir(
                "ppt",
                f"{IDENTITY['id']}:interrupted",
            )
        output_dir.mkdir(parents=True)
        (output_dir / "partial.zip").write_bytes(b"partial")
        tasks_path.write_text(
            service_module.json.dumps({
                "tasks": [{
                    "id": "interrupted",
                    "owner_id": IDENTITY["id"],
                    "status": "running",
                    "kind": "ppt",
                    "created_at": "2026-01-01 00:00:00",
                    "updated_at": "2026-01-01 00:00:00",
                    "created_ts": time.time(),
                    "updated_ts": time.time(),
                }],
            }),
            encoding="utf-8",
        )

        with mock.patch.object(service_module, "EDITABLE_FILE_ROOT", files_root):
            service = EditableFileTaskService(tasks_path)
        self.addCleanup(service.close)

        recovered = service.list_tasks(IDENTITY, ["interrupted"])["items"][0]
        self.assertEqual(recovered["status"], "error")
        self.assertIn("服务已重启", str(recovered["error"]))
        self.assertFalse(output_dir.exists())


class EditableFileTaskAPITests(unittest.TestCase):
    def test_request_model_enforces_small_fields_before_service(self) -> None:
        with self.assertRaises(ValidationError):
            ai_module.EditableFileTaskRequest.model_validate({"base64_images": ["x"] * 5})
        with self.assertRaises(ValidationError):
            ai_module.EditableFileTaskRequest.model_validate(
                {"prompt": "x" * (MAX_EDITABLE_PROMPT_CHARS + 1)}
            )
        with self.assertRaises(ValidationError):
            ai_module.EditableFileTaskRequest.model_validate({"client_task_id": "x" * 129})

    def test_queue_full_returns_openai_429_with_retry_after(self) -> None:
        app = FastAPI()
        app.include_router(ai_module.create_router())
        fake_service = mock.MagicMock()
        fake_service.submit_ppt.side_effect = EditableFileTaskOverloadedError()

        with (
            mock.patch.object(ai_module, "require_identity", return_value=IDENTITY),
            mock.patch.object(ai_module, "editable_file_task_service", fake_service),
            TestClient(app) as client,
        ):
            response = client.post("/v1/ppt/generations", json={"prompt": "test"})

        self.assertEqual(response.status_code, 429, response.text)
        self.assertEqual(response.headers.get("retry-after"), "5")
        self.assertEqual(response.json()["error"]["code"], "editable_file_queue_full")

    def test_route_passes_request_deadline_to_background_service(self) -> None:
        app = FastAPI()
        app.include_router(ai_module.create_router())
        fake_service = mock.MagicMock()
        fake_service.submit_ppt.return_value = {"id": "task", "status": "queued"}
        expected_deadline = ImageTaskDeadline(123)

        with (
            mock.patch.object(ai_module, "require_identity", return_value=IDENTITY),
            mock.patch.object(ai_module, "editable_file_task_service", fake_service),
            mock.patch.object(ai_module.image_task_runtime, "new_deadline", return_value=expected_deadline),
            mock.patch.object(
                ai_module,
                "run_in_threadpool",
                side_effect=AssertionError("PPT 提交不能依赖 AnyIO 公共线程池"),
            ),
            TestClient(app) as client,
        ):
            response = client.post("/v1/ppt/generations", json={"prompt": "test"})

        self.assertEqual(response.status_code, 200, response.text)
        deadline = fake_service.submit_ppt.call_args.kwargs["deadline"]
        self.assertIs(deadline, expected_deadline)


if __name__ == "__main__":
    unittest.main()

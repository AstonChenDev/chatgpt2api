from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from services.config import DATA_DIR, config
from services.content_filter import check_request, request_text
from services.image_task_runtime import (
    ImageTaskDeadline,
    ImageTaskDeadlineExceeded,
    ImageTaskRuntimeError,
    image_task_runtime,
)
from services.log_service import LOG_TYPE_CALL, log_service
from services.protocol import openai_v1_image_edit, openai_v1_image_generations

TASK_STATUS_QUEUED = "queued"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_SUCCESS = "success"
TASK_STATUS_ERROR = "error"
TERMINAL_STATUSES = {TASK_STATUS_SUCCESS, TASK_STATUS_ERROR}
UNFINISHED_STATUSES = {TASK_STATUS_QUEUED, TASK_STATUS_RUNNING}
MAX_CLIENT_TASK_ID_CHARS = 128
MAX_IMAGE_PROMPT_CHARS = 32_000
MAX_IMAGE_MODEL_CHARS = 128
MAX_PERSISTED_TASKS = 5_000


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _timestamp(value: object) -> float:
    if not isinstance(value, str) or not value.strip():
        return 0.0
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value[:26], fmt).timestamp()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _clean(value: object, default: str = "") -> str:
    return str(value or default).strip()


def _owner_id(identity: dict[str, object]) -> str:
    return _clean(identity.get("id")) or "anonymous"


def _task_key(owner_id: str, task_id: str) -> str:
    return f"{owner_id}:{task_id}"


def _collect_image_urls(data: list[Any]) -> list[str]:
    urls: list[str] = []
    for item in data:
        if isinstance(item, dict):
            url = item.get("url")
            if isinstance(url, str) and url:
                urls.append(url)
    return urls


def _public_task(task: dict[str, Any]) -> dict[str, Any]:
    item = {
        "id": task.get("id"),
        "status": task.get("status"),
        "mode": task.get("mode"),
        "model": task.get("model"),
        "size": task.get("size"),
        "quality": task.get("quality"),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at"),
    }
    if task.get("conversation_id"):
        item["conversation_id"] = task.get("conversation_id")
    if task.get("data") is not None:
        item["data"] = task.get("data")
    if task.get("usage") is not None:
        item["usage"] = task.get("usage")
    if task.get("error"):
        item["error"] = task.get("error")
    if task.get("progress"):
        item["progress"] = task.get("progress")
    if task.get("duration_ms") is not None:
        item["duration_ms"] = task.get("duration_ms")
    if task.get("status") in (TASK_STATUS_RUNNING, TASK_STATUS_QUEUED):
        if task.get("status") == TASK_STATUS_RUNNING:
            # RUNNING 状态仅在 started_ts 被设置后（image_stream_resolve_start）才计时
            base_ts = task.get("started_ts")
        else:
            # QUEUED 状态从 created_ts 开始计时（排队等待中）
            base_ts = task.get("created_ts") or task.get("updated_ts")
        if base_ts:
            item["elapsed_secs"] = round(time.time() - base_ts, 1)
    return item


class ImageTaskService:
    def __init__(
        self,
        path: Path,
        *,
        generation_handler: Callable[[dict[str, Any]], dict[str, Any]] = openai_v1_image_generations.handle,
        edit_handler: Callable[[dict[str, Any]], dict[str, Any]] = openai_v1_image_edit.handle,
        retention_days_getter: Callable[[], int] | None = None,
    ):
        self.path = path
        self.generation_handler = generation_handler
        self.edit_handler = edit_handler
        self.retention_days_getter = retention_days_getter or (lambda: config.image_retention_days)
        self._lock = threading.RLock()
        self._tasks: dict[str, dict[str, Any]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._tasks = self._load_locked()
            changed = self._recover_unfinished_locked()
            changed = self._cleanup_locked() or changed
            if changed:
                self._save_locked()

    def submit_generation(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        prompt: str,
        model: str,
        size: str | None,
        quality: str = "auto",
        base_url: str = "",
        deadline: ImageTaskDeadline | None = None,
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "model": model,
            "n": 1,
            "size": size,
            "quality": quality,
            "response_format": "url",
            "base_url": base_url,
        }
        return self._submit(
            identity,
            client_task_id=client_task_id,
            mode="generate",
            payload=payload,
            deadline=deadline,
        )

    def submit_edit(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        prompt: str,
        model: str,
        size: str | None,
        quality: str = "auto",
        base_url: str = "",
        images: list[tuple[bytes, str, str]] | None = None,
        masks: list[tuple[bytes, str, str]] | None = None,
        deadline: ImageTaskDeadline | None = None,
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "images": images or [],
            "mask": masks or [],
            "model": model,
            "n": 1,
            "size": size,
            "quality": quality,
            "response_format": "url",
            "base_url": base_url,
        }
        return self._submit(
            identity,
            client_task_id=client_task_id,
            mode="edit",
            payload=payload,
            deadline=deadline,
            weight=max(1, len(images or []) + len(masks or [])),
        )

    def submit_edit_with_loader(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        prompt: str,
        model: str,
        size: str | None,
        quality: str = "auto",
        base_url: str = "",
        input_loader: Callable[[ImageTaskDeadline], tuple[list[tuple[bytes, str, str]], list[tuple[bytes, str, str]] | None]],
        deadline: ImageTaskDeadline | None = None,
        weight: int = 1,
    ) -> dict[str, Any]:
        """提交后台编辑任务，并在同一个运行时槽内完成输入读取和生成。

        调用方会等到 input_loader 已把 UploadFile/base64 安全消费完再结束 HTTP
        请求；图片 bytes 随后只存在于正在运行的工作线程，不会堆进等待队列。
        """

        payload = {
            "prompt": prompt,
            "model": model,
            "n": 1,
            "size": size,
            "quality": quality,
            "response_format": "url",
            "base_url": base_url,
        }

        def load_payload(active_deadline: ImageTaskDeadline) -> dict[str, Any]:
            images, masks = input_loader(active_deadline)
            return {**payload, "images": images, "mask": masks or []}

        return self._submit(
            identity,
            client_task_id=client_task_id,
            mode="edit",
            payload=payload,
            deadline=deadline,
            payload_loader=load_payload,
            weight=weight,
        )

    def list_tasks(self, identity: dict[str, object], task_ids: list[str]) -> dict[str, Any]:
        owner = _owner_id(identity)
        requested_ids = [_clean(task_id) for task_id in task_ids if _clean(task_id)]
        with self._lock:
            if self._cleanup_locked():
                self._save_locked()
            items = []
            missing_ids = []
            for task_id in requested_ids:
                task = self._tasks.get(_task_key(owner, task_id))
                if task is None:
                    missing_ids.append(task_id)
                else:
                    items.append(_public_task(task))
            if not requested_ids:
                items = [
                    _public_task(task)
                    for task in self._tasks.values()
                    if task.get("owner_id") == owner
                ]
                items.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
                missing_ids = []
            return {"items": items, "missing_ids": missing_ids}

    def _submit(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        mode: str,
        payload: dict[str, Any],
        deadline: ImageTaskDeadline | None = None,
        payload_loader: Callable[[ImageTaskDeadline], dict[str, Any]] | None = None,
        weight: int | None = None,
    ) -> dict[str, Any]:
        task_id = _clean(client_task_id)
        if not task_id:
            raise ValueError("client_task_id is required")
        if len(task_id) > MAX_CLIENT_TASK_ID_CHARS:
            raise ValueError("client_task_id 最多支持 128 个字符")
        prompt = _clean(payload.get("prompt"))
        if len(prompt) > MAX_IMAGE_PROMPT_CHARS:
            raise ValueError("prompt 最多支持 32000 个字符")
        model_name = _clean(payload.get("model"), "gpt-image-2")
        if len(model_name) > MAX_IMAGE_MODEL_CHARS:
            raise ValueError("model 最多支持 128 个字符")
        owner = _owner_id(identity)
        key = _task_key(owner, task_id)
        now = _now_iso()
        should_start = False
        with self._lock:
            cleaned = self._cleanup_locked(reserve=1)
            task = self._tasks.get(key)
            if task is not None:
                if cleaned:
                    self._save_locked()
                return _public_task(task)
            task = {
                "id": task_id,
                "owner_id": owner,
                "status": TASK_STATUS_QUEUED,
                "mode": mode,
                "model": model_name,
                "size": _clean(payload.get("size")),
                "quality": _clean(payload.get("quality"), "auto"),
                "created_at": now,
                "updated_at": now,
                "created_ts": time.time(),
            }
            self._tasks[key] = task
            self._save_locked()
            should_start = True

        if should_start:
            active_deadline = deadline or image_task_runtime.new_deadline()
            payload_with_deadline = {**payload, "_image_task_deadline": active_deadline}
            payload_ready = threading.Event() if payload_loader is not None else None
            payload_errors: list[BaseException] = []
            try:
                job = image_task_runtime.submit(
                    self._run_task,
                    key,
                    mode,
                    payload_with_deadline,
                    dict(identity),
                    _clean(payload.get("model"), "gpt-image-2"),
                    payload_loader,
                    payload_ready,
                    payload_errors,
                    deadline=active_deadline,
                    weight=max(1, int(weight if weight is not None else payload.get("n") or 1)),
                    name=f"background-image-{mode}",
                )
            except ImageTaskRuntimeError as exc:
                self._update_task(key, status=TASK_STATUS_ERROR, error=str(exc), data=[])
                raise

            def record_queue_failure(future) -> None:
                """任务若在开跑前排队超时，工作函数不会执行，需在此落库。"""

                try:
                    error = future.exception()
                except BaseException as exc:  # pragma: no cover - 防御 Future 自身异常
                    error = exc
                if error is None:
                    return
                with self._lock:
                    current = self._tasks.get(key)
                    still_queued = current is not None and current.get("status") == TASK_STATUS_QUEUED
                if still_queued:
                    self._update_task(key, status=TASK_STATUS_ERROR, error=str(error), data=[])
                if payload_ready is not None and not payload_ready.is_set():
                    payload_errors.append(error)
                    payload_ready.set()

            job.add_done_callback(record_queue_failure)
            if payload_ready is not None:
                # 保持请求体中间件的大请求许可，直到上传/base64 已在工作线程消费。
                # 最长仍只服从同一个总截止线，调度器故障也不会无限等。
                if not payload_ready.wait(timeout=active_deadline.remaining() + 0.2):
                    active_deadline.cancel("后台图片输入等待达到总时限")
                    raise ImageTaskDeadlineExceeded(active_deadline.timeout_secs, "后台图片输入等待")
                if payload_errors:
                    raise payload_errors[0]
        return _public_task(task)

    def _run_task(
        self,
        key: str,
        mode: str,
        payload: dict[str, Any],
        identity: dict[str, object],
        model: str,
        payload_loader: Callable[[ImageTaskDeadline], dict[str, Any]] | None = None,
        payload_ready: threading.Event | None = None,
        payload_errors: list[BaseException] | None = None,
    ) -> None:
        started = time.time()
        self._update_task(key, status=TASK_STATUS_RUNNING, error="")
        # 创建进度回调，每个步骤完成后更新任务状态
        def progress_callback(step: str) -> None:
            if step == "image_stream_resolve_start":
                self._update_task(key, started_ts=time.time())
            self._update_task(key, progress=step)
        try:
            if payload_loader is not None:
                active_deadline = payload.get("_image_task_deadline")
                if not isinstance(active_deadline, ImageTaskDeadline):
                    raise RuntimeError("image task deadline is missing")
                try:
                    loaded = payload_loader(active_deadline)
                    payload = {**loaded, "_image_task_deadline": active_deadline}
                except BaseException as exc:
                    if payload_errors is not None:
                        payload_errors.append(exc)
                    raise
                finally:
                    if payload_ready is not None:
                        payload_ready.set()
            # 将进度回调添加到 payload 中（handler 会提取并传递给 ConversationRequest）
            payload_with_progress = {**payload, "progress_callback": progress_callback}
            # 后台任务的审核也属于图片工作负载，必须留在专用执行器内。
            check_request(
                request_text(payload.get("prompt")),
                deadline=payload.get("_image_task_deadline"),
            )
            handler = self.edit_handler if mode == "edit" else self.generation_handler
            result = handler(payload_with_progress)
            if not isinstance(result, dict):
                raise RuntimeError("image task returned streaming result unexpectedly")
            data = result.get("data")
            account_email = _clean(result.get("_account_email") or result.get("account_email"))
            if not isinstance(data, list) or not data:
                upstream = _clean(result.get("message"))
                if upstream:
                    message = upstream
                else:
                    message = "号池中没有可用账号或所有账号均被限流，请检查号池状态（账号额度、是否被封禁、是否到达生图上限）"
                error = RuntimeError(message)
                if account_email:
                    setattr(error, "account_email", account_email)
                raise error
            usage = result.get("usage")
            duration_ms = int((time.time() - started) * 1000)
            self._update_task(
                key,
                status=TASK_STATUS_SUCCESS,
                data=data,
                usage=usage,
                error="",
                duration_ms=duration_ms,
                **({"account_email": account_email} if account_email else {}),
            )
            self._log_call(
                identity,
                mode,
                model,
                started,
                "调用完成",
                request_preview=request_text(payload.get("prompt")),
                urls=_collect_image_urls(data),
                account_email=account_email,
            )
        except Exception as exc:
            error_message = str(exc) or "image task failed"
            account_email = _clean(getattr(exc, "account_email", ""))
            conversation_id = _clean(getattr(exc, "conversation_id", ""))
            duration_ms = int((time.time() - started) * 1000)
            self._update_task(key, status=TASK_STATUS_ERROR, error=error_message, data=[],
                              duration_ms=duration_ms,
                              **({"conversation_id": conversation_id} if conversation_id else {}),
                              **({"account_email": account_email} if account_email else {}))
            self._log_call(
                identity,
                mode,
                model,
                started,
                "调用失败",
                request_preview=request_text(payload.get("prompt")),
                status="failed",
                error=error_message,
                account_email=account_email,
            )
        finally:
            # 防御 payload_loader 抛出 BaseException 等极端路径，避免 HTTP 调用方失联。
            if payload_ready is not None and not payload_ready.is_set():
                payload_ready.set()

    def _log_call(
        self,
        identity: dict[str, object],
        mode: str,
        model: str,
        started: float,
        suffix: str,
        *,
        request_preview: str = "",
        status: str = "success",
        error: str = "",
        urls: list[str] | None = None,
        account_email: str = "",
    ) -> None:
        endpoint = "/v1/images/edits" if mode == "edit" else "/v1/images/generations"
        summary_prefix = "图生图" if mode == "edit" else "文生图"
        detail = {
            "key_id": identity.get("id"),
            "key_name": identity.get("name"),
            "role": identity.get("role"),
            "endpoint": endpoint,
            "model": model,
            "started_at": datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": _now_iso(),
            "duration_ms": int((time.time() - started) * 1000),
            "status": status,
        }
        if request_preview:
            detail["request_text"] = request_preview
        if error:
            detail["error"] = error
        if account_email:
            detail["account_email"] = account_email
        if urls:
            detail["urls"] = list(dict.fromkeys(urls))
        try:
            log_service.add(LOG_TYPE_CALL, f"{summary_prefix}{suffix}", detail)
        except Exception:
            pass

    def _update_task(self, key: str, **updates: Any) -> None:
        with self._lock:
            task = self._tasks.get(key)
            if task is None:
                return
            task.update(updates)
            task["updated_at"] = _now_iso()
            task["updated_ts"] = time.time()
            self._save_locked()

    def _load_locked(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        raw_items = raw.get("tasks") if isinstance(raw, dict) else raw
        if not isinstance(raw_items, list):
            return {}
        tasks: dict[str, dict[str, Any]] = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            task_id = _clean(item.get("id"))
            owner = _clean(item.get("owner_id"))
            if not task_id or not owner:
                continue
            status = _clean(item.get("status"))
            if status not in {TASK_STATUS_QUEUED, TASK_STATUS_RUNNING, TASK_STATUS_SUCCESS, TASK_STATUS_ERROR}:
                status = TASK_STATUS_ERROR
            task = {
                "id": task_id,
                "owner_id": owner,
                "status": status,
                "mode": "edit" if item.get("mode") == "edit" else "generate",
                "model": _clean(item.get("model"), "gpt-image-2"),
                "size": _clean(item.get("size")),
                "quality": _clean(item.get("quality"), "auto"),
                "created_at": _clean(item.get("created_at"), _now_iso()),
                "updated_at": _clean(item.get("updated_at"), _clean(item.get("created_at"), _now_iso())),
                "created_ts": item.get("created_ts"),
                "updated_ts": item.get("updated_ts"),
                "started_ts": item.get("started_ts"),
                "duration_ms": item.get("duration_ms"),
            }
            data = item.get("data")
            if isinstance(data, list):
                task["data"] = data
            usage = item.get("usage")
            if isinstance(usage, dict):
                task["usage"] = usage
            error = _clean(item.get("error"))
            if error:
                task["error"] = error
            conversation_id = _clean(item.get("conversation_id"))
            if conversation_id:
                task["conversation_id"] = conversation_id
            # 仅内部持久化，用于超时后以原账号继续轮询；不会出现在公开任务响应中。
            account_email = _clean(item.get("account_email"))
            if account_email:
                task["account_email"] = account_email
            tasks[_task_key(owner, task_id)] = task
        return tasks

    def _save_locked(self) -> None:
        items = sorted(self._tasks.values(), key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps({"tasks": items}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(self.path)

    def _recover_unfinished_locked(self) -> bool:
        changed = False
        for task in self._tasks.values():
            if task.get("status") in UNFINISHED_STATUSES:
                task["status"] = TASK_STATUS_ERROR
                task["error"] = "服务已重启，未完成的图片任务已中断"
                task["updated_at"] = _now_iso()
                changed = True
        return changed

    def _cleanup_locked(self, *, reserve: int = 0) -> bool:
        try:
            retention_days = max(1, int(self.retention_days_getter()))
        except Exception:
            retention_days = 30
        cutoff = time.time() - retention_days * 86400
        removed_keys = [
            key
            for key, task in self._tasks.items()
            if task.get("status") in TERMINAL_STATUSES and _timestamp(task.get("updated_at")) < cutoff
        ]
        for key in removed_keys:
            self._tasks.pop(key, None)

        # 保留期内的终结任务也必须有硬上限，否则随机 client_task_id 会让 JSON
        # 文件永久增长。只淘汰最旧的终结记录，绝不删除排队或运行中的任务。
        target_size = max(0, MAX_PERSISTED_TASKS - max(0, int(reserve)))
        overflow = max(0, len(self._tasks) - target_size)
        if overflow:
            terminal_oldest = sorted(
                (
                    (key, _timestamp(task.get("updated_at")))
                    for key, task in self._tasks.items()
                    if task.get("status") in TERMINAL_STATUSES
                ),
                key=lambda item: item[1],
            )
            for key, _ in terminal_oldest[:overflow]:
                self._tasks.pop(key, None)
                removed_keys.append(key)
        return bool(removed_keys)

    def resume_poll(
        self,
        identity: dict[str, object],
        task_id: str,
        extra_timeout_secs: float = 30.0,
    ) -> dict[str, Any]:
        """恢复对已超时任务的轮询，额外等待 extra_timeout_secs 秒。"""
        owner = _owner_id(identity)
        key = _task_key(owner, _clean(task_id))
        with self._lock:
            task = self._tasks.get(key)
            if task is None:
                raise ValueError("task not found")
            if task.get("status") != TASK_STATUS_ERROR:
                raise ValueError("task is not in error state")
            error_msg = _clean(task.get("error"))
            if "超时" not in error_msg and "时限" not in error_msg:
                raise ValueError("task error is not a timeout error")
            conversation_id = _clean(task.get("conversation_id"))
            if not conversation_id:
                raise ValueError("task has no conversation_id")
            mode = task.get("mode", "generate")
            model = task.get("model", "gpt-image-2")
            account_email = _clean(task.get("account_email"))
            # 将任务状态重置为 running
            self._update_task(key, status=TASK_STATUS_RUNNING, error="")

        from services.account_service import account_service

        if not account_email:
            self._update_task(key, status=TASK_STATUS_ERROR, error="原图片任务未记录账号，无法安全续轮询")
            raise ValueError("original image task account is unavailable")

        access_token = next(
            (
                _clean(account.get("access_token"))
                for account in account_service.list_accounts()
                if _clean(account.get("email")).lower() == account_email.lower()
            ),
            "",
        )
        if not access_token:
            self._update_task(key, status=TASK_STATUS_ERROR, error="无法找到原图片任务使用的账号")
            raise ValueError("original image task account is unavailable")

        deadline = image_task_runtime.new_deadline(extra_timeout_secs)
        try:
            job = image_task_runtime.submit(
                self._run_resume_poll,
                key,
                conversation_id,
                extra_timeout_secs,
                dict(identity),
                mode,
                model,
                access_token,
                deadline,
                deadline=deadline,
                name="background-image-resume-poll",
            )
        except ImageTaskRuntimeError as exc:
            self._update_task(key, status=TASK_STATUS_ERROR, error=str(exc))
            raise

        def record_resume_queue_failure(future) -> None:
            try:
                error = future.exception()
            except BaseException as exc:  # pragma: no cover - 防御 Future 自身异常
                error = exc
            if error is not None:
                with self._lock:
                    current = self._tasks.get(key)
                    still_running = current is not None and current.get("status") == TASK_STATUS_RUNNING
                if still_running:
                    self._update_task(key, status=TASK_STATUS_ERROR, error=str(error), data=[])

        job.add_done_callback(record_resume_queue_failure)
        return _public_task(task)

    def _run_resume_poll(
        self,
        key: str,
        conversation_id: str,
        extra_timeout_secs: float,
        identity: dict[str, object],
        mode: str,
        model: str,
        access_token: str,
        deadline: ImageTaskDeadline,
    ) -> None:
        """后台线程：继续轮询已有 conversation_id 的图片结果。"""
        started = time.time()
        backend = None
        try:
            from services.openai_backend_api import OpenAIBackendAPI
            from services.protocol.conversation import format_image_result

            # 私有 conversation 必须使用最初生成任务的账号；匿名客户端无法续轮询。
            backend = OpenAIBackendAPI(access_token=access_token, image_deadline=deadline)
            file_ids, sediment_ids = backend._poll_image_results(
                conversation_id,
                extra_timeout_secs,
            )
            if not file_ids and not sediment_ids:
                raise RuntimeError(
                    f"继续等待 {extra_timeout_secs} 秒后仍未找到图片结果。"
                )

            image_urls = backend.resolve_conversation_image_urls(
                conversation_id, file_ids, sediment_ids, poll=False,
            )
            if not image_urls:
                raise RuntimeError("图片 URL 解析失败")

            downloaded_images = [image_data for image_data in backend.download_image_bytes(image_urls) if image_data]
            if not downloaded_images:
                raise RuntimeError("图片下载结果为空")
            image_items = [
                {"b64_json": __import__("base64").b64encode(image_data).decode("ascii")}
                for image_data in downloaded_images
            ]
            # 获取 task 的原始 prompt（从 _public_task 的 mode 判断）
            with self._lock:
                task = self._tasks.get(key)
                quality = _clean(task.get("quality"), "auto") if task else "auto"
                size = _clean(task.get("size")) if task else None
            data = format_image_result(
                image_items,
                "",  # prompt 已不重要，结果已经拿到了
                "b64_json",
                "",
                int(time.time()),
                deadline=deadline,
            )["data"]
            if not data:
                raise RuntimeError("图片结果格式化后为空")
            self._update_task(key, status=TASK_STATUS_SUCCESS, data=data, error="", duration_ms=int((time.time() - started) * 1000))
            self._log_call(
                identity,
                mode,
                model,
                started,
                "调用完成（续轮询）",
                status="success",
                urls=_collect_image_urls(data),
            )
        except Exception as exc:
            error_message = str(exc) or "resume poll failed"
            duration_ms = int((time.time() - started) * 1000)
            self._update_task(key, status=TASK_STATUS_ERROR, error=error_message, data=[], duration_ms=duration_ms)
            self._log_call(
                identity,
                mode,
                model,
                started,
                "调用失败（续轮询）",
                status="failed",
                error=error_message,
            )
        finally:
            if backend is not None:
                backend.close()


image_task_service = ImageTaskService(DATA_DIR / "image_tasks.json")

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import quote

from PIL import Image, UnidentifiedImageError

from services.account_service import account_service
from services.config import DATA_DIR, config
from services.content_filter import check_request, request_text
from services.disk_space_guard import ensure_disk_space
from services.image_task_runtime import ImageTaskDeadline, ImageTaskDeadlineExceeded, ImageTaskRuntimeError
from services.log_service import LOG_TYPE_CALL, log_service
from services.openai_backend_api import EDITABLE_ARTIFACT_TOTAL_MAX_BYTES, EDITABLE_FILE_MODEL, OpenAIBackendAPI
from utils.helper import new_uuid
from utils.log import logger


TASK_STATUS_QUEUED = "queued"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_SUCCESS = "success"
TASK_STATUS_ERROR = "error"
UNFINISHED_STATUSES = {TASK_STATUS_QUEUED, TASK_STATUS_RUNNING}
TERMINAL_STATUSES = {TASK_STATUS_SUCCESS, TASK_STATUS_ERROR}
EDITABLE_FILE_PLAN_TYPES = ("Plus", "Team", "Pro", "Enterprise")
EDITABLE_FILE_ROOT = DATA_DIR / "files"
EDITABLE_FILE_TASKS_PATH = DATA_DIR / "editable_file_tasks.json"

# 可编辑文件任务通常会占用较多内存和一个长连接。这里故意采用独立、保守的固定容量，
# 不与普通 API 或图片生成共用线程池；达到上限就快速返回 429，而不是无限创建线程。
DEFAULT_EDITABLE_MAX_WORKERS = 1
DEFAULT_EDITABLE_MAX_QUEUE_SIZE = 2

# 输入边界既限制单次解码峰值，也限制排队闭包长期持有的 base64 总量。
MAX_EDITABLE_PROMPT_CHARS = 32_000
MAX_EDITABLE_TASK_ID_CHARS = 128
MAX_EDITABLE_IMAGE_COUNT = 4
MAX_EDITABLE_IMAGE_BYTES = 10 * 1024 * 1024
MAX_EDITABLE_TOTAL_IMAGE_BYTES = 20 * 1024 * 1024
MAX_EDITABLE_IMAGE_PIXELS = 25_000_000
MAX_EDITABLE_BASE64_CHARS = ((MAX_EDITABLE_IMAGE_BYTES + 2) // 3) * 4 + 256
MAX_EDITABLE_TASK_QUERY_IDS = 100
MAX_EDITABLE_TASKS_PER_OWNER = 200
MAX_EDITABLE_TASK_RECORDS = 1_000
EDITABLE_TASK_RETENTION_SECS = 30 * 24 * 60 * 60
EDITABLE_OUTPUT_CLEANUP_INTERVAL_SECS = 6 * 60 * 60
MAX_EDITABLE_ERROR_CHARS = 2_000
MAX_EDITABLE_LOG_TEXT_CHARS = 1_000
EDITABLE_STATUS_LOCK_TIMEOUT_SECS = 0.05
EDITABLE_PUBLIC_LOCK_TIMEOUT_SECS = 0.25

_EDITABLE_TASK_ID_RE = re.compile(
    rf"^[A-Za-z0-9][A-Za-z0-9_-]{{0,{MAX_EDITABLE_TASK_ID_CHARS - 1}}}$"
)
_EDITABLE_DATA_URI_RE = re.compile(r"^data:([^;]+);base64,(.*)$", re.IGNORECASE | re.DOTALL)
_ALLOWED_EDITABLE_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP"}
_OUTPUT_DIR_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9_-]+")


class EditableFileTaskOverloadedError(ImageTaskRuntimeError):
    status_code = 429
    error_type = "rate_limit_error"
    code = "editable_file_queue_full"

    def __init__(self, message: str = "可编辑文件任务繁忙，请稍后重试。") -> None:
        super().__init__(message)


class EditableFileTaskUnavailableError(ImageTaskRuntimeError):
    status_code = 503
    error_type = "server_error"
    code = "editable_file_runtime_unavailable"

    def __init__(self, message: str = "可编辑文件任务暂时不可用，请稍后重试。") -> None:
        super().__init__(message)


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _clean(value: object, default: str = "") -> str:
    return str(value or default).strip()


def _safe_float(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _safe_error(error: object) -> str:
    text = _clean(error, "可编辑文件任务失败")
    return text[:MAX_EDITABLE_ERROR_CHARS]


def _owner_id(identity: dict[str, object]) -> str:
    return _clean(identity.get("id")) or "anonymous"


def _task_key(owner_id: str, task_id: str) -> str:
    return f"{owner_id}:{task_id}"


def _elapsed_seconds(task: dict[str, Any]) -> int:
    start = _safe_float(task.get("started_ts") or task.get("created_ts"))
    end = _safe_float(task.get("ended_ts")) or time.time()
    return max(0, int(end - start)) if start else 0


def _file_url(path: Path, base_url: str) -> str:
    rel = path.resolve().relative_to(EDITABLE_FILE_ROOT.resolve()).as_posix()
    prefix = str(base_url or "").strip().rstrip("/")
    return f"{prefix}/files/{quote(rel, safe='/')}" if prefix else f"/files/{quote(rel, safe='/')}"


def _task_output_dir(kind: str, key: str) -> Path:
    """用 owner+task key 的稳定短哈希隔离同 ID 任务，同时保留可读的安全前缀。"""

    normalized_kind = "psd" if kind == "psd" else "ppt"
    task_id = str(key or "").rsplit(":", 1)[-1]
    safe_prefix = _OUTPUT_DIR_SAFE_CHARS_RE.sub("_", task_id).strip("_-")[:48] or "task"
    digest = hashlib.sha256(str(key or "").encode("utf-8")).hexdigest()[:16]
    return EDITABLE_FILE_ROOT / normalized_kind / f"{safe_prefix}-{digest}"


def _cleanup_failed_output_directory(path: Path) -> None:
    """只清理 files 根目录下的单任务目录，绝不允许误删持久化根目录。"""

    root = EDITABLE_FILE_ROOT.resolve()
    resolved = path.resolve()
    resolved.relative_to(root)
    if resolved.parent.parent != root or resolved.parent.name not in {"ppt", "psd"}:
        raise ValueError("只允许清理 files/ppt 或 files/psd 下的一级任务目录")
    shutil.rmtree(resolved, ignore_errors=False)


def _editable_access_token(deadline: ImageTaskDeadline) -> str:
    deadline.check("选择可编辑文件账号")
    accounts = [
        item for item in account_service.list_accounts()
        if _clean(item.get("access_token"))
           and item.get("status") not in {"禁用", "异常"}
           and account_service._account_matches_any_plan_type(item, EDITABLE_FILE_PLAN_TYPES)
    ]
    if not accounts:
        raise RuntimeError("没有可用于生成可编辑文件的 Plus/Team/Pro 账号")
    accounts.sort(key=lambda item: _clean(item.get("last_used_at")))
    token = _clean(accounts[0].get("access_token"))
    return account_service.refresh_access_token(
        token,
        event="editable_file_task",
        deadline=deadline,
    ) or token


def _public_task(task: dict[str, Any]) -> dict[str, Any]:
    item = {
        "id": task.get("id"),
        "taskId": task.get("id"),
        "status": task.get("status"),
        "kind": task.get("kind"),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at"),
        "elapsed_seconds": _elapsed_seconds(task),
    }
    for key in ("result", "error"):
        if task.get(key):
            item[key] = task[key]
    return item


class EditableFileTaskService:
    """PPT/PSD 后台任务服务。

    运行中的工作不可被 Python 强行杀死，因此容量许可要一直保留到工作线程真正返回；
    即便上游库发生不可中断的异常阻塞，新请求也只会得到 429，不会继续堆积线程和内存。
    单个固定看门狗只负责把超时任务可靠落成终态，不会为每个请求另建计时线程。
    """

    def __init__(
        self,
        path: Path = EDITABLE_FILE_TASKS_PATH,
        *,
        max_workers: int = DEFAULT_EDITABLE_MAX_WORKERS,
        max_queue_size: int = DEFAULT_EDITABLE_MAX_QUEUE_SIZE,
        enable_output_cleanup: bool = True,
    ) -> None:
        self.path = path
        self.max_workers = max(1, int(max_workers))
        self.max_queue_size = max(0, int(max_queue_size))
        self._capacity = self.max_workers + self.max_queue_size
        self._lock = threading.RLock()
        self._tasks: dict[str, dict[str, Any]] = {}
        self._jobs: dict[str, tuple[ImageTaskDeadline, float]] = {}
        self._reservations: dict[str, ImageTaskDeadline] = {}
        self._pending_keys: set[str] = set()
        self._admitted_jobs = 0
        self._active_jobs = 0
        self._submitted_jobs = 0
        self._rejected_jobs = 0
        self._queue_timeouts = 0
        self._task_timeouts = 0
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="editable-file-worker",
        )
        self._watchdog_stop = threading.Event()
        self._watchdog_wake = threading.Event()
        self._output_cleanup_enabled = bool(enable_output_cleanup)
        self._next_output_cleanup_at = time.monotonic() + EDITABLE_OUTPUT_CLEANUP_INTERVAL_SECS
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._tasks = self._load_locked()
            changed, interrupted_outputs = self._recover_unfinished_locked()
            changed = self._cleanup_locked() or changed
            if changed:
                self._save_locked()
        # 文件树扫描和递归删除必须在任务锁外进行，避免健康检查、提交和轮询被磁盘拖住。
        for output_dir in interrupted_outputs:
            self._cleanup_output_directory_with_log(output_dir, "editable_file_recovery_cleanup_error")
        if self._output_cleanup_enabled:
            self._run_expired_output_cleanup_safely()
        self._watchdog = threading.Thread(
            target=self._watchdog_loop,
            name="editable-file-watchdog",
            daemon=True,
        )
        self._watchdog.start()

    def submit_ppt(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str = "",
        prompt: str = "",
        base64_images: list[str] | None = None,
        base_url: str = "",
        deadline: ImageTaskDeadline | None = None,
    ) -> dict[str, Any]:
        return self._submit(
            identity,
            client_task_id=client_task_id,
            kind="ppt",
            prompt=prompt,
            base64_images=base64_images or [],
            base_url=base_url,
            deadline=deadline,
        )

    def submit_psd(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str = "",
        prompt: str = "",
        base64_images: list[str] | None = None,
        base_url: str = "",
        deadline: ImageTaskDeadline | None = None,
    ) -> dict[str, Any]:
        return self._submit(
            identity,
            client_task_id=client_task_id,
            kind="psd",
            prompt=prompt,
            base64_images=base64_images or [],
            base_url=base_url,
            deadline=deadline,
        )

    def list_tasks(self, identity: dict[str, object], task_ids: list[str]) -> dict[str, Any]:
        owner = _owner_id(identity)
        requested = [_clean(item) for item in task_ids if _clean(item)]
        if len(requested) > MAX_EDITABLE_TASK_QUERY_IDS:
            raise ValueError(f"一次最多查询 {MAX_EDITABLE_TASK_QUERY_IDS} 个任务")
        if not self._lock.acquire(timeout=EDITABLE_PUBLIC_LOCK_TIMEOUT_SECS):
            raise EditableFileTaskUnavailableError("任务状态正在持久化，请稍后重试。")
        try:
            if requested:
                items = [task for task_id in requested if (task := self._tasks.get(_task_key(owner, task_id)))]
                return {
                    "items": [_public_task(item) for item in items],
                    "missing_ids": [task_id for task_id in requested if _task_key(owner, task_id) not in self._tasks],
                }
            items = [task for task in self._tasks.values() if task.get("owner_id") == owner]
        finally:
            self._lock.release()
        items.sort(key=lambda item: _safe_float(item.get("updated_ts")), reverse=True)
        return {
            "items": [_public_task(item) for item in items[:MAX_EDITABLE_TASKS_PER_OWNER]],
            "missing_ids": [],
        }

    def runtime_status(self) -> dict[str, int | bool]:
        """返回轻量运行状态，便于测试和后续接入健康检查。"""

        now = time.monotonic()
        # 健康检查不能反过来阻塞事件循环：任务状态写盘时持有同一把锁，磁盘若异常
        # 缓慢，healthz 最多等待 50ms 就返回明确的不健康状态。
        if not self._lock.acquire(timeout=EDITABLE_STATUS_LOCK_TIMEOUT_SECS):
            return {
                "healthy": False,
                "status_lock_busy": True,
                "max_workers": self.max_workers,
                "max_queue_size": self.max_queue_size,
                "active_jobs": self._active_jobs,
                "queued_jobs": -1,
                "validating_jobs": -1,
                "admitted_jobs": self._admitted_jobs,
                "overdue_jobs": -1,
                "submitted_jobs": self._submitted_jobs,
                "rejected_jobs": self._rejected_jobs,
                "queue_timeouts": self._queue_timeouts,
                "task_timeouts": self._task_timeouts,
            }
        try:
            overdue = sum(1 for deadline, _ in self._jobs.values() if now >= deadline.expires_at)
            overdue += sum(1 for deadline in self._reservations.values() if now >= deadline.expires_at)
            return {
                "healthy": overdue == 0 and self._watchdog.is_alive(),
                "status_lock_busy": False,
                "max_workers": self.max_workers,
                "max_queue_size": self.max_queue_size,
                "active_jobs": self._active_jobs,
                "queued_jobs": len(self._pending_keys),
                "validating_jobs": len(self._reservations),
                "admitted_jobs": self._admitted_jobs,
                "overdue_jobs": overdue,
                "submitted_jobs": self._submitted_jobs,
                "rejected_jobs": self._rejected_jobs,
                "queue_timeouts": self._queue_timeouts,
                "task_timeouts": self._task_timeouts,
            }
        finally:
            self._lock.release()

    def close(self, *, wait: bool = True) -> None:
        """仅供测试或进程优雅退出使用；不会取消已经开始的上游请求。"""

        self._watchdog_stop.set()
        self._watchdog_wake.set()
        self._executor.shutdown(wait=wait, cancel_futures=False)
        if wait and self._watchdog.is_alive():
            self._watchdog.join(timeout=2)

    def cleanup_expired_outputs(
        self,
        *,
        now: float | None = None,
        retention_secs: float = EDITABLE_TASK_RETENTION_SECS,
    ) -> dict[str, int]:
        """删除超过保留期的成功产物目录；仅扫描 ppt/psd 下一级目录。

        该函数不获取任务锁，可由启动流程、固定看门狗或低磁盘保护安全调用。
        符号链接、越界路径、普通文件和类型目录本身一律不会删除。
        """

        cutoff = (time.time() if now is None else float(now)) - max(0.0, float(retention_secs))
        result = {"removed": 0, "kept": 0, "skipped": 0, "failed": 0}
        root = EDITABLE_FILE_ROOT.resolve()
        for kind in ("ppt", "psd"):
            kind_root = (root / kind).resolve()
            try:
                kind_root.relative_to(root)
            except ValueError:
                result["failed"] += 1
                continue
            if not kind_root.is_dir():
                continue
            try:
                # 流式遍历目录项，历史遗留目录很多时也不会一次性构造巨大列表。
                for candidate in kind_root.iterdir():
                    try:
                        # is_dir 会跟随符号链接；必须先明确跳过，防止删除链接目标。
                        if candidate.is_symlink() or not candidate.is_dir():
                            result["skipped"] += 1
                            continue
                        resolved = candidate.resolve()
                        resolved.relative_to(kind_root)
                        if resolved.parent != kind_root:
                            raise ValueError("只允许清理一级任务目录")
                        if candidate.stat().st_mtime >= cutoff:
                            result["kept"] += 1
                            continue
                        _cleanup_failed_output_directory(candidate)
                        result["removed"] += 1
                    except BaseException as exc:
                        result["failed"] += 1
                        logger.error({
                            "event": "editable_file_expired_output_cleanup_error",
                            "path": str(candidate),
                            "error_type": exc.__class__.__name__,
                            "error": _safe_error(exc),
                        })
            except OSError as exc:
                result["failed"] += 1
                logger.error({
                    "event": "editable_file_output_scan_error",
                    "path": str(kind_root),
                    "error_type": exc.__class__.__name__,
                    "error": _safe_error(exc),
                })
        return result

    @staticmethod
    def _cleanup_output_directory_with_log(path: Path, event: str) -> None:
        if not path.exists():
            return
        try:
            _cleanup_failed_output_directory(path)
        except BaseException as exc:
            logger.error({
                "event": event,
                "path": str(path),
                "error_type": exc.__class__.__name__,
                "error": _safe_error(exc),
            })

    def _run_expired_output_cleanup_safely(self) -> None:
        try:
            self.cleanup_expired_outputs()
        except BaseException as exc:
            logger.error({
                "event": "editable_file_output_cleanup_unexpected_error",
                "error_type": exc.__class__.__name__,
                "error": _safe_error(exc),
            })

    def _submit(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        kind: str,
        prompt: str,
        base64_images: list[str],
        base_url: str,
        deadline: ImageTaskDeadline | None,
    ) -> dict[str, Any]:
        supplied_task_id = _clean(client_task_id)
        if supplied_task_id and not _EDITABLE_TASK_ID_RE.fullmatch(supplied_task_id):
            raise ValueError("client_task_id 仅支持 1-128 位字母、数字、下划线和连字符")
        task_id = supplied_task_id or new_uuid()
        owner = _owner_id(identity)
        key = _task_key(owner, task_id)
        active_deadline = deadline or ImageTaskDeadline(
            float(config.get_image_task_runtime_settings()["total_timeout_secs"])
        )
        queue_timeout = float(config.get_image_task_runtime_settings()["queue_timeout_secs"])
        active_deadline.check("校验可编辑文件请求")

        # 必须先原子预留容量，再执行 base64/Pillow 校验。否则大量小体积压缩图可在
        # “队列是否已满”之前同时占住 FastAPI 公共线程池，后台仍会失去响应。
        if not self._lock.acquire(timeout=EDITABLE_PUBLIC_LOCK_TIMEOUT_SECS):
            raise EditableFileTaskUnavailableError("任务状态正在持久化，请稍后重试。")
        try:
            if key in self._tasks:
                return _public_task(self._tasks[key])
            if key in self._reservations:
                self._rejected_jobs += 1
                raise EditableFileTaskOverloadedError("相同 client_task_id 正在提交，请稍后查询或重试。")
            if self._admitted_jobs >= self._capacity:
                self._rejected_jobs += 1
                raise EditableFileTaskOverloadedError()
            if self._admitted_jobs >= self.max_workers and queue_timeout <= 0:
                self._rejected_jobs += 1
                raise EditableFileTaskOverloadedError("可编辑文件工作线程已满，当前配置不允许排队。")
            self._reservations[key] = active_deadline
            self._admitted_jobs += 1
        finally:
            self._lock.release()

        try:
            normalized_prompt = self._validate_prompt(prompt)
            normalized_images = self._validate_base64_images(kind, base64_images, active_deadline)
            active_deadline.check("提交可编辑文件任务")
        except BaseException:
            with self._lock:
                self._release_reservation_locked(key)
            raise

        now = _now_iso()
        monotonic_now = time.monotonic()
        with self._lock:
            # 正常情况下 reservation 已排除同 ID 并发；这里仍保留防御性检查，避免
            # 测试替身或未来持久层迁移造成重复任务。
            if key in self._tasks:
                self._release_reservation_locked(key)
                return _public_task(self._tasks[key])

            immediate_capacity = len(self._jobs) < self.max_workers
            if not immediate_capacity and queue_timeout <= 0:
                self._rejected_jobs += 1
                self._release_reservation_locked(key)
                raise EditableFileTaskOverloadedError("可编辑文件工作线程已满，当前配置不允许排队。")

            queue_deadline = active_deadline.expires_at
            if not immediate_capacity:
                queue_deadline = min(queue_deadline, monotonic_now + queue_timeout)
            ts = time.time()
            task = {
                "id": task_id,
                "owner_id": owner,
                "status": TASK_STATUS_QUEUED,
                "kind": kind,
                "model": EDITABLE_FILE_MODEL,
                "created_at": now,
                "updated_at": now,
                "created_ts": ts,
                "updated_ts": ts,
            }
            self._reservations.pop(key, None)
            self._tasks[key] = task
            self._cleanup_locked()
            self._jobs[key] = (active_deadline, queue_deadline)
            self._pending_keys.add(key)
            self._submitted_jobs += 1
            try:
                self._save_locked()
            except BaseException as exc:
                self._rollback_submission_locked(key)
                raise EditableFileTaskUnavailableError("无法保存可编辑文件任务，请稍后重试。") from exc

        response_task = _public_task(dict(task))
        try:
            self._executor.submit(
                self._run_task_entry,
                key,
                kind,
                normalized_prompt,
                normalized_images,
                dict(identity),
                str(base_url or "")[:2048],
                active_deadline,
            )
        except BaseException as exc:
            with self._lock:
                self._rollback_submission_locked(key)
                try:
                    self._save_locked()
                except BaseException:
                    pass
            raise EditableFileTaskUnavailableError() from exc
        self._watchdog_wake.set()
        return response_task

    @staticmethod
    def _validate_prompt(prompt: object) -> str:
        value = str(prompt or "")
        if len(value) > MAX_EDITABLE_PROMPT_CHARS:
            raise ValueError(f"prompt 不能超过 {MAX_EDITABLE_PROMPT_CHARS} 个字符")
        return value.strip()

    @staticmethod
    def _validate_base64_images(
        kind: str,
        base64_images: list[str],
        deadline: ImageTaskDeadline,
    ) -> list[str]:
        if not isinstance(base64_images, list):
            raise ValueError("base64_images 必须是数组")
        if kind == "psd" and not base64_images:
            raise ValueError("PSD 生成至少需要 1 张图片")
        if len(base64_images) > MAX_EDITABLE_IMAGE_COUNT:
            raise ValueError(f"base64_images 最多支持 {MAX_EDITABLE_IMAGE_COUNT} 张图片")

        validated: list[str] = []
        total_bytes = 0
        for index, item in enumerate(base64_images, start=1):
            deadline.check(f"校验第 {index} 张图片")
            raw = str(item or "").strip()
            if not raw:
                raise ValueError(f"第 {index} 张图片为空")
            if len(raw) > MAX_EDITABLE_BASE64_CHARS:
                raise ValueError(f"第 {index} 张图片不能超过 10MB")

            payload = raw
            match = _EDITABLE_DATA_URI_RE.fullmatch(raw)
            if match:
                declared_mime = _clean(match.group(1)).lower()
                if not declared_mime.startswith("image/"):
                    raise ValueError(f"第 {index} 个 data URI 不是图片")
                payload = _clean(match.group(2))

            # 先用编码长度估算总量，避免解码后才发现总量超限并产生额外内存峰值。
            estimated_bytes = (len(payload) * 3) // 4
            if total_bytes + estimated_bytes > MAX_EDITABLE_TOTAL_IMAGE_BYTES + 3:
                raise ValueError("全部图片解码后总大小不能超过 20MB")
            try:
                decoded = base64.b64decode(payload, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"第 {index} 张图片不是有效的 base64") from exc
            if len(decoded) > MAX_EDITABLE_IMAGE_BYTES:
                raise ValueError(f"第 {index} 张图片不能超过 10MB")
            total_bytes += len(decoded)
            if total_bytes > MAX_EDITABLE_TOTAL_IMAGE_BYTES:
                raise ValueError("全部图片解码后总大小不能超过 20MB")

            try:
                with Image.open(BytesIO(decoded)) as image:
                    image_format = str(image.format or "").upper()
                    width, height = image.size
                    if image_format not in _ALLOWED_EDITABLE_IMAGE_FORMATS:
                        raise ValueError(f"第 {index} 张图片仅支持 PNG、JPEG 或 WebP")
                    if width <= 0 or height <= 0 or width * height > MAX_EDITABLE_IMAGE_PIXELS:
                        raise ValueError(f"第 {index} 张图片像素不能超过 2500 万")
                    image.verify()
            except ValueError:
                raise
            except (UnidentifiedImageError, OSError, SyntaxError) as exc:
                raise ValueError(f"第 {index} 张图片文件损坏或格式不受支持") from exc
            except Exception as exc:
                raise ValueError(f"第 {index} 张图片无法安全解析") from exc
            deadline.check(f"校验第 {index} 张图片")
            validated.append(raw)
        return validated

    def _run_task_entry(
        self,
        key: str,
        kind: str,
        prompt: str,
        base64_images: list[str],
        identity: dict[str, object],
        base_url: str,
        deadline: ImageTaskDeadline,
    ) -> None:
        should_run = False
        try:
            with self._lock:
                watch = self._jobs.get(key)
                self._pending_keys.discard(key)
                task = self._tasks.get(key)
                if task is not None and task.get("status") in UNFINISHED_STATUSES:
                    if watch is not None and time.monotonic() >= watch[1]:
                        self._queue_timeouts += 1
                        self._finish_error_locked(key, "可编辑文件任务排队超时，请稍后重试。")
                        self._save_locked()
                    else:
                        self._active_jobs += 1
                        should_run = True
            if should_run:
                self._run_task(key, kind, prompt, base64_images, identity, base_url, deadline)
        except BaseException as exc:
            # _run_task 自身已有完整错误处理；这里是最后一道兜底，防止编程错误留下 running。
            self._update_task(
                key,
                status=TASK_STATUS_ERROR,
                error=_safe_error(exc),
                ended_ts=time.time(),
            )
        finally:
            with self._lock:
                if should_run:
                    self._active_jobs = max(0, self._active_jobs - 1)
                self._admitted_jobs = max(0, self._admitted_jobs - 1)
                self._jobs.pop(key, None)
                self._pending_keys.discard(key)
                task = self._tasks.get(key)
                if task is not None and task.get("status") in UNFINISHED_STATUSES:
                    self._finish_error_locked(key, "可编辑文件工作线程异常结束")
                    try:
                        self._save_locked()
                    except BaseException:
                        pass
            self._watchdog_wake.set()

    def _run_task(
        self,
        key: str,
        kind: str,
        prompt: str,
        base64_images: list[str],
        identity: dict[str, object],
        base_url: str,
        deadline: ImageTaskDeadline,
    ) -> None:
        started = time.time()
        token = ""
        account_email = ""
        backend: OpenAIBackendAPI | None = None
        output_dir: Path | None = None
        keep_output = False
        if not self._update_task(key, status=TASK_STATUS_RUNNING, error="", started_ts=started):
            return
        try:
            # 内容审核也放在本服务的专用线程内，避免慢审核占满 FastAPI 公共线程池。
            deadline.check("审核可编辑文件请求")
            check_request(prompt, deadline=deadline)
            token = _editable_access_token(deadline)
            account = account_service.get_account(token) or {}
            account_email = _clean(account.get("email"))
            backend = OpenAIBackendAPI(token, image_deadline=deadline)
            output_dir = _task_output_dir(kind, key)
            # 上游生成通常接近 5 分钟；若磁盘连完整结果上限都容纳不下，应在调用
            # 上游前快速失败，不能让用户等待数分钟后才在下载阶段发现空间不足。
            ensure_disk_space(output_dir, EDITABLE_ARTIFACT_TOTAL_MAX_BYTES)
            remaining = deadline.check("生成可编辑文件")
            export_kwargs = {
                "timeout_secs": remaining,
                # 后端旧轮询使用不可取消的 time.sleep；缩短间隔把超时误差控制在 1 秒内。
                "poll_interval_secs": min(1.0, max(0.1, remaining)),
            }
            if kind == "psd":
                result = backend.export_psd_zip(base64_images, prompt, output_dir, **export_kwargs)
            else:
                result = backend.export_ppt_zip(base64_images, prompt, output_dir, **export_kwargs)
            deadline.check("保存可编辑文件结果")
            account_service.mark_text_used(token)
            deadline.check("完成可编辑文件任务")
            data = {
                "conversation_id": result.conversation_id,
                "primary_url": _file_url(result.primary_path, base_url),
                "zip_url": _file_url(result.zip_path, base_url),
            }
            deadline.check("持久化可编辑文件任务结果")
            updated = self._finish_success(key, data, account_email, deadline)
            if updated:
                keep_output = True
                self._log_call(identity, kind, started, request_text(prompt), account_email=account_email, result=data)
        except BaseException as exc:
            error = _safe_error(exc)
            updated = self._update_task(
                key,
                status=TASK_STATUS_ERROR,
                error=error,
                account_email=account_email,
                ended_ts=time.time(),
            )
            if updated:
                self._log_call(
                    identity,
                    kind,
                    started,
                    request_text(prompt),
                    status="failed",
                    error=error,
                    account_email=account_email,
                )
        finally:
            if backend is not None:
                try:
                    backend.close()
                except BaseException as exc:
                    logger.warning({
                        "event": "editable_file_backend_close_failed",
                        "error_type": exc.__class__.__name__,
                        "error": _safe_error(exc),
                    })
            if output_dir is not None and not keep_output and output_dir.exists():
                try:
                    # 上游部分下载成功、后续下载失败或任务超时后，必须删除整个任务目录，
                    # 否则不可见的孤儿 PPT/PSD/ZIP 会持续吃满生产磁盘。
                    _cleanup_failed_output_directory(output_dir)
                except BaseException as exc:
                    logger.error({
                        "event": "editable_file_failed_output_cleanup_error",
                        "path": str(output_dir),
                        "error_type": exc.__class__.__name__,
                        "error": _safe_error(exc),
                    })

    def public_file_path(self, relative_path: str) -> Path:
        raw = str(relative_path or "").replace("\\", "/").lstrip("/")
        path = (EDITABLE_FILE_ROOT / raw).resolve()
        path.relative_to(EDITABLE_FILE_ROOT.resolve())
        if not path.is_file():
            raise FileNotFoundError(raw)
        return path

    def _update_task(self, key: str, **updates: Any) -> bool:
        with self._lock:
            task = self._tasks.get(key)
            if task is None:
                return False
            current_status = task.get("status")
            next_status = updates.get("status", current_status)
            # 看门狗写入 ERROR 后，迟到的上游成功结果不得把任务复活成 SUCCESS。
            if current_status in TERMINAL_STATUSES and next_status != current_status:
                return False
            task.update(updates)
            task["updated_at"] = _now_iso()
            task["updated_ts"] = time.time()
            self._save_locked()
            return True

    def _finish_success(
        self,
        key: str,
        result: dict[str, str],
        account_email: str,
        deadline: ImageTaskDeadline,
    ) -> bool:
        """在同一把锁内完成最后期限检查和成功落库，杜绝超时任务被迟到结果复活。"""

        with self._lock:
            task = self._tasks.get(key)
            if task is None or task.get("status") in TERMINAL_STATUSES:
                return False
            deadline.check("持久化可编辑文件任务结果")
            task.update({
                "status": TASK_STATUS_SUCCESS,
                "result": result,
                "account_email": account_email,
                "error": "",
                "ended_ts": time.time(),
                "updated_at": _now_iso(),
                "updated_ts": time.time(),
            })
            try:
                self._save_locked()
                # JSON 全量写盘也属于五分钟总预算；若写盘跨过截止线，必须改成 ERROR。
                deadline.check("持久化可编辑文件任务结果")
            except BaseException as exc:
                task.update({
                    "status": TASK_STATUS_ERROR,
                    "error": _safe_error(exc),
                    "ended_ts": time.time(),
                    "updated_at": _now_iso(),
                    "updated_ts": time.time(),
                })
                try:
                    self._save_locked()
                except BaseException:
                    pass
                raise
            return True

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.is_set():
            wait_for = 0.5
            try:
                now = time.monotonic()
                changed = False
                with self._lock:
                    for key, (deadline, queue_deadline) in list(self._jobs.items()):
                        task = self._tasks.get(key)
                        if task is None or task.get("status") not in UNFINISHED_STATUSES:
                            continue
                        if now >= deadline.expires_at:
                            self._task_timeouts += 1
                            error = ImageTaskDeadlineExceeded(deadline.timeout_secs, "可编辑文件任务")
                            changed = self._finish_error_locked(key, str(error)) or changed
                            continue
                        if key in self._pending_keys and now >= queue_deadline:
                            self._queue_timeouts += 1
                            changed = self._finish_error_locked(
                                key,
                                "可编辑文件任务排队超时，请稍后重试。",
                            ) or changed
                            continue
                        next_expiry = min(deadline.expires_at, queue_deadline if key in self._pending_keys else deadline.expires_at)
                        wait_for = min(wait_for, max(0.01, next_expiry - now))
                    if changed:
                        self._save_locked()
            except BaseException as exc:
                # 看门狗是任务状态的最后防线，单次磁盘或数据异常不能让它永久退出。
                logger.error({
                    "event": "editable_file_watchdog_error",
                    "error_type": exc.__class__.__name__,
                    "error": _safe_error(exc),
                })
                wait_for = 0.1
            self._watchdog_wake.wait(wait_for)
            self._watchdog_wake.clear()
            if self._output_cleanup_enabled and time.monotonic() >= self._next_output_cleanup_at:
                self._next_output_cleanup_at = time.monotonic() + EDITABLE_OUTPUT_CLEANUP_INTERVAL_SECS
                self._run_expired_output_cleanup_safely()

    def _finish_error_locked(self, key: str, error: str) -> bool:
        task = self._tasks.get(key)
        if task is None or task.get("status") not in UNFINISHED_STATUSES:
            return False
        task.update({
            "status": TASK_STATUS_ERROR,
            "error": _safe_error(error),
            "ended_ts": time.time(),
            "updated_at": _now_iso(),
            "updated_ts": time.time(),
        })
        return True

    def _rollback_submission_locked(self, key: str) -> None:
        self._tasks.pop(key, None)
        self._jobs.pop(key, None)
        self._pending_keys.discard(key)
        self._admitted_jobs = max(0, self._admitted_jobs - 1)
        self._submitted_jobs = max(0, self._submitted_jobs - 1)

    def _release_reservation_locked(self, key: str) -> None:
        if self._reservations.pop(key, None) is not None:
            self._admitted_jobs = max(0, self._admitted_jobs - 1)

    def _load_locked(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        tasks: dict[str, dict[str, Any]] = {}
        for item in (raw.get("tasks") if isinstance(raw, dict) else raw) or []:
            if not isinstance(item, dict):
                continue
            task_id = _clean(item.get("id"))
            owner = _clean(item.get("owner_id"))
            if not task_id or not owner:
                continue
            task = {
                "id": task_id,
                "owner_id": owner,
                "status": _clean(item.get("status"), TASK_STATUS_ERROR),
                "kind": "psd" if item.get("kind") == "psd" else "ppt",
                "created_at": _clean(item.get("created_at"), _now_iso()),
                "updated_at": _clean(item.get("updated_at"), _clean(item.get("created_at"), _now_iso())),
                "created_ts": _safe_float(item.get("created_ts")),
                "updated_ts": _safe_float(item.get("updated_ts")),
            }
            if isinstance(item.get("result"), dict):
                task["result"] = item["result"]
            if item.get("error"):
                task["error"] = _safe_error(item.get("error"))
            for field in ("started_ts", "ended_ts"):
                if item.get(field):
                    task[field] = _safe_float(item.get(field))
            tasks[_task_key(owner, task_id)] = task
        return tasks

    def _save_locked(self) -> None:
        items = sorted(self._tasks.values(), key=lambda item: _safe_float(item.get("updated_ts")), reverse=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps({"tasks": items}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(self.path)

    def _recover_unfinished_locked(self) -> tuple[bool, list[Path]]:
        changed = False
        interrupted_outputs: list[Path] = []
        for task in self._tasks.values():
            if task.get("status") in UNFINISHED_STATUSES:
                task["status"] = TASK_STATUS_ERROR
                task["error"] = "服务已重启，未完成的任务已中断"
                task["ended_ts"] = time.time()
                task["updated_at"] = _now_iso()
                task["updated_ts"] = time.time()
                interrupted_outputs.append(
                    _task_output_dir(
                        _clean(task.get("kind"), "ppt"),
                        _task_key(_clean(task.get("owner_id")), _clean(task.get("id"))),
                    )
                )
                changed = True
        return changed, interrupted_outputs

    def _cleanup_locked(self) -> bool:
        """清理过期终态记录并设置双重硬上限，避免 JSON 全量重写随时间无限变慢。"""

        changed = False
        cutoff = time.time() - EDITABLE_TASK_RETENTION_SECS
        for key, task in list(self._tasks.items()):
            updated_ts = _safe_float(task.get("updated_ts"))
            if task.get("status") in TERMINAL_STATUSES and updated_ts and updated_ts < cutoff:
                self._tasks.pop(key, None)
                changed = True

        by_owner: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for key, task in self._tasks.items():
            by_owner.setdefault(_clean(task.get("owner_id")), []).append((key, task))
        for items in by_owner.values():
            items.sort(key=lambda pair: _safe_float(pair[1].get("updated_ts")), reverse=True)
            terminal_seen = 0
            for key, task in items:
                if task.get("status") not in TERMINAL_STATUSES:
                    continue
                terminal_seen += 1
                if terminal_seen > MAX_EDITABLE_TASKS_PER_OWNER:
                    self._tasks.pop(key, None)
                    changed = True

        if len(self._tasks) > MAX_EDITABLE_TASK_RECORDS:
            unfinished = {
                key for key, task in self._tasks.items()
                if task.get("status") in UNFINISHED_STATUSES
            }
            terminal = sorted(
                (
                    (key, task) for key, task in self._tasks.items()
                    if key not in unfinished
                ),
                key=lambda pair: _safe_float(pair[1].get("updated_ts")),
                reverse=True,
            )
            keep_terminal = max(0, MAX_EDITABLE_TASK_RECORDS - len(unfinished))
            for key, _ in terminal[keep_terminal:]:
                self._tasks.pop(key, None)
                changed = True
        return changed

    def _log_call(
        self,
        identity: dict[str, object],
        kind: str,
        started: float,
        request_preview: str,
        *,
        status: str = "success",
        error: str = "",
        account_email: str = "",
        result: dict[str, str] | None = None,
    ) -> None:
        detail = {
            "key_id": identity.get("id"),
            "key_name": identity.get("name"),
            "role": identity.get("role"),
            "endpoint": f"/v1/{kind}/generations",
            "model": EDITABLE_FILE_MODEL,
            "started_at": datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": _now_iso(),
            "duration_ms": int((time.time() - started) * 1000),
            "status": status,
        }
        if request_preview:
            detail["request_text"] = str(request_preview)[:MAX_EDITABLE_LOG_TEXT_CHARS]
        if account_email:
            detail["account_email"] = account_email
        if error:
            detail["error"] = _safe_error(error)
        if result:
            detail["result"] = result
        try:
            log_service.add(LOG_TYPE_CALL, f"{kind.upper()}生成任务{'失败' if status == 'failed' else '完成'}", detail)
        except Exception:
            pass


editable_file_task_service = EditableFileTaskService()

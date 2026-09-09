from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from services.config import DATA_DIR

_PROTECTED_POST_PATHS = {
    "/v1/images/generations",
    "/v1/images/edits",
    "/v1/chat/completions",
    "/v1/responses",
    "/v1/messages",
    "/v1/search",
    "/v1/ppt/generations",
    "/v1/psd/generations",
    "/api/image-tasks/generations",
    "/api/image-tasks/edits",
}


class DeploymentRuntime:
    """蓝绿槽位的进程内状态。

    槽位和排空标记都位于持久化 data 目录。候选容器启动时保持待机，不会
    恢复共享任务或运行定时写任务；只有成为 active 后才允许接收生成请求。
    """

    def __init__(self) -> None:
        self.slot = str(os.getenv("CHATGPT2API_DEPLOY_SLOT", "") or "").strip().lower()
        if self.slot not in {"blue", "green"}:
            self.slot = ""
        state_dir = DATA_DIR / "deploy"
        self.active_path = state_dir / "active-slot"
        self.draining_path = state_dir / "draining-slot"
        self._lock = threading.Lock()
        self._ready = not bool(self.slot)
        self._background_running = not bool(self.slot)
        self._background_quiesced = bool(self.slot)
        self._activation_error = ""

    @property
    def managed(self) -> bool:
        return bool(self.slot)

    @staticmethod
    def _read_slot(path: Path) -> str:
        try:
            value = path.read_text(encoding="utf-8").strip().lower()
        except OSError:
            return ""
        return value if value in {"blue", "green", "legacy"} else ""

    def is_active(self) -> bool:
        return not self.managed or self._read_slot(self.active_path) == self.slot

    def is_draining(self) -> bool:
        if not self.managed:
            return self._read_slot(self.draining_path) == "legacy"
        return self._read_slot(self.draining_path) == self.slot

    def accepts_generation_requests(self) -> bool:
        with self._lock:
            ready = self._ready
        return self.is_active() and not self.is_draining() and ready

    def mark_transition(
        self,
        *,
        ready: bool,
        background_running: bool,
        background_quiesced: bool,
        error: str = "",
    ) -> None:
        with self._lock:
            self._ready = bool(ready)
            self._background_running = bool(background_running)
            self._background_quiesced = bool(background_quiesced)
            self._activation_error = str(error or "")[:300]

    def status(self) -> dict[str, Any]:
        active = self.is_active()
        draining = self.is_draining()
        with self._lock:
            ready = self._ready
            background_running = self._background_running
            background_quiesced = self._background_quiesced
            error = self._activation_error
        return {
            # 待机槽本身可以通过 Docker 健康检查；发布脚本另外检查 active/ready。
            "healthy": not bool(error),
            "managed": self.managed,
            "slot": self.slot or "legacy",
            "active": active,
            "draining": draining,
            "ready": ready,
            "accepting_generation_requests": active and not draining and ready,
            "background_running": background_running,
            "background_quiesced": background_quiesced,
            "error": error,
        }


deployment_runtime = DeploymentRuntime()


class DeploymentGateMiddleware:
    """切流期间快速拒绝新生成任务，但不打断已经进入应用的请求。"""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        path = str(scope.get("path") or "").rstrip("/") or "/"
        method = str(scope.get("method") or "").upper()
        task_state_request = path in {
            "/api/image-tasks",
            "/v1/editable-file-tasks",
        }
        protected = task_state_request or (
            method == "POST"
            and (
                path in _PROTECTED_POST_PATHS
                or (
                    path.startswith("/api/image-tasks/")
                    and path.endswith("/resume-poll")
                )
            )
        )
        if (
            scope.get("type") == "http"
            and protected
            and not deployment_runtime.accepts_generation_requests()
        ):
            body = json.dumps(
                {
                    "error": {
                        "message": "服务正在平滑切换，请在 3 秒后重试",
                        "type": "deployment_draining",
                        "code": "deployment_draining",
                    }
                },
                ensure_ascii=False,
            ).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 503,
                    "headers": [
                        (b"content-type", b"application/json; charset=utf-8"),
                        (b"content-length", str(len(body)).encode("ascii")),
                        (b"retry-after", b"3"),
                        (b"cache-control", b"no-store"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)

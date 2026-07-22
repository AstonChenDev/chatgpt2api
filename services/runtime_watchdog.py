from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from typing import Any

from utils.log import logger


def _enabled_from_env() -> bool:
    value = str(os.getenv("CHATGPT2API_WATCHDOG_ENABLED", "true") or "true").strip().lower()
    return value not in {"0", "false", "no", "off", "disabled"}


def _grace_from_env() -> float:
    try:
        value = float(os.getenv("CHATGPT2API_WATCHDOG_GRACE_SECS", "60") or 60)
    except (TypeError, ValueError, OverflowError):
        value = 60.0
    return max(15.0, min(300.0, value))


def runtime_component_statuses() -> dict[str, dict[str, Any]]:
    """读取纯内存状态；健康检查和看门狗都不得发起外部网络请求。"""

    from services.editable_file_task_service import editable_file_task_service
    from services.image_io_runtime import image_io_runtime
    from services.image_task_runtime import image_task_runtime

    return {
        "image_runtime": image_task_runtime.status(),
        "image_io": image_io_runtime.status(),
        "editable_file_runtime": editable_file_task_service.runtime_status(),
    }


def _watchdog_loop(
    stop_event: threading.Event,
    *,
    grace_secs: float,
    status_getter: Callable[[], dict[str, dict[str, Any]]],
    exit_process: Callable[[int], object],
) -> None:
    unhealthy_since: dict[str, float] = {}
    while not stop_event.wait(5.0):
        now = time.monotonic()
        try:
            statuses = status_getter()
        except Exception as exc:
            logger.error({
                "event": "runtime_watchdog_status_failed",
                "error_type": exc.__class__.__name__,
                "error": str(exc)[:300],
            })
            continue

        for component, status in statuses.items():
            if bool(status.get("healthy", True)):
                unhealthy_since.pop(component, None)
                continue
            first_seen = unhealthy_since.setdefault(component, now)
            unhealthy_for = now - first_seen
            if unhealthy_for < grace_secs:
                continue
            logger.error({
                "event": "runtime_watchdog_forced_restart",
                "component": component,
                "unhealthy_secs": round(unhealthy_for, 1),
                "grace_secs": grace_secs,
                "status": status,
            })
            # Python 无法安全终止永久阻塞的第三方线程。主动结束容器进程后，
            # Docker 的 unless-stopped 策略会重建所有线程和许可，这是最终兜底。
            exit_process(70)
            return


def start_runtime_watchdog(stop_event: threading.Event) -> threading.Thread | None:
    """启动进程级自愈看门狗；返回线程供 lifespan 在优雅退出时回收。"""

    if not _enabled_from_env():
        logger.info({"event": "runtime_watchdog_disabled"})
        return None
    grace_secs = _grace_from_env()
    thread = threading.Thread(
        target=_watchdog_loop,
        kwargs={
            "stop_event": stop_event,
            "grace_secs": grace_secs,
            "status_getter": runtime_component_statuses,
            "exit_process": os._exit,
        },
        name="runtime-self-heal-watchdog",
        daemon=True,
    )
    thread.start()
    logger.info({"event": "runtime_watchdog_started", "grace_secs": grace_secs})
    return thread

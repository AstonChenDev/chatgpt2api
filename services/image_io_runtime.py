from __future__ import annotations

import asyncio
import itertools
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, TypeVar

from fastapi import HTTPException


T = TypeVar("T")


class ImageIORuntime:
    """后台图库/COS 读取专用的有界执行器。

    远端缩略图、ZIP 和删除等同步 I/O 绝不能直接跑在 Uvicorn 唯一事件循环；
    也不能占用 AnyIO 公共线程池，否则慢 COS 会再次拖住登录和普通管理接口。
    """

    def __init__(self, max_workers: int = 4, max_queue_size: int = 8) -> None:
        self.max_workers = max(1, int(max_workers))
        self.max_queue_size = max(0, int(max_queue_size))
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="image-io")
        self._lock = threading.Lock()
        self._pending = 0
        self._active = 0
        self._rejected = 0
        self._sequence = itertools.count(1)
        self._running: dict[int, tuple[float, float]] = {}

    def _invoke(
        self,
        job_id: int,
        timeout_secs: float,
        func: Callable[..., T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> T:
        with self._lock:
            self._active += 1
            self._running[job_id] = (time.monotonic(), timeout_secs)
        try:
            return func(*args, **kwargs)
        finally:
            with self._lock:
                self._active = max(0, self._active - 1)
                self._running.pop(job_id, None)

    def _release_pending(self, _future: Future[Any]) -> None:
        with self._lock:
            self._pending = max(0, self._pending - 1)

    async def run(
        self,
        func: Callable[..., T],
        *args: Any,
        timeout_secs: float = 30.0,
        operation_name: str = "图片存储读取",
        **kwargs: Any,
    ) -> T:
        normalized_timeout = max(0.1, float(timeout_secs))
        operation = str(operation_name or "后台阻塞操作")[:50]
        with self._lock:
            if self._pending >= self.max_workers + self.max_queue_size:
                self._rejected += 1
                raise HTTPException(
                    status_code=503,
                    detail={"error": f"{operation}繁忙，请稍后重试"},
                    headers={"Retry-After": "3"},
                )
            self._pending += 1
            job_id = next(self._sequence)

        try:
            future = self._executor.submit(
                self._invoke,
                job_id,
                normalized_timeout,
                func,
                args,
                kwargs,
            )
        except BaseException:
            with self._lock:
                self._pending = max(0, self._pending - 1)
            raise
        future.add_done_callback(self._release_pending)
        wrapped = asyncio.wrap_future(future)
        try:
            return await asyncio.wait_for(asyncio.shield(wrapped), timeout=normalized_timeout)
        except asyncio.TimeoutError as exc:
            # 尚在内部队列中的任务可以真正取消，避免客户端早已超时后仍执行陈旧的
            # ZIP/删除操作；已经运行的同步 I/O 由容器看门狗负责最终自愈。
            future.cancel()
            wrapped.add_done_callback(self._consume_async_future)
            raise HTTPException(status_code=504, detail={"error": f"{operation}超时"}) from exc
        except asyncio.CancelledError:
            future.cancel()
            wrapped.add_done_callback(self._consume_async_future)
            raise

    @staticmethod
    def _consume_async_future(future: asyncio.Future[Any]) -> None:
        try:
            future.exception()
        except BaseException:
            pass

    def status(self) -> dict[str, int | float | bool]:
        now = time.monotonic()
        with self._lock:
            overdue_durations = [
                now - started - timeout_secs
                for started, timeout_secs in self._running.values()
                if now > started + timeout_secs
            ]
            return {
                "healthy": not overdue_durations,
                "active_jobs": self._active,
                "pending_jobs": self._pending,
                "overdue_jobs": len(overdue_durations),
                "oldest_overdue_secs": round(max(overdue_durations, default=0.0), 1),
                "max_workers": self.max_workers,
                "max_queue_size": self.max_queue_size,
                "rejected": self._rejected,
            }


image_io_runtime = ImageIORuntime()

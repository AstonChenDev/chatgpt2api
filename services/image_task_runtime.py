from __future__ import annotations

import asyncio
import itertools
import threading
import time
from collections import deque
from concurrent.futures import Future, InvalidStateError, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Generic, TypeVar

from services.image_task_runtime_config import DEFAULT_IMAGE_TASK_RUNTIME
from utils.log import logger


T = TypeVar("T")


def _try_set_future_result(future: Future[Any], value: Any) -> None:
    try:
        future.set_result(value)
    except InvalidStateError:
        pass


def _try_set_future_exception(future: Future[Any], error: BaseException) -> None:
    try:
        future.set_exception(error)
    except InvalidStateError:
        pass


class ImageTaskRuntimeError(RuntimeError):
    """图片任务运行时可安全返回给客户端的基础异常。"""

    status_code = 500
    error_type = "server_error"
    code = "image_runtime_error"

    def to_openai_error(self) -> dict[str, object]:
        return {
            "error": {
                "message": str(self),
                "type": self.error_type,
                "param": None,
                "code": self.code,
            }
        }


class ImageTaskOverloadedError(ImageTaskRuntimeError):
    status_code = 429
    error_type = "rate_limit_error"
    code = "image_queue_full"

    def __init__(self, message: str = "图片任务繁忙，请稍后重试。") -> None:
        super().__init__(message)


class ImageTaskQueueTimeoutError(ImageTaskRuntimeError):
    status_code = 429
    error_type = "rate_limit_error"
    code = "image_queue_timeout"

    def __init__(self, message: str = "图片任务排队超时，请稍后重试。") -> None:
        super().__init__(message)


class ImageTaskDeadlineExceeded(ImageTaskRuntimeError, TimeoutError):
    status_code = 504
    error_type = "timeout_error"
    code = "image_generation_timeout"

    def __init__(self, timeout_secs: float = 300, phase: str = "图片生成") -> None:
        timeout_text = int(timeout_secs) if float(timeout_secs).is_integer() else round(timeout_secs, 1)
        super().__init__(f"{phase}超过总时限（{timeout_text} 秒），任务已终止，请稍后重试。")
        self.timeout_secs = float(timeout_secs)
        self.phase = phase


class ImageTaskCancelled(ImageTaskRuntimeError):
    status_code = 499
    error_type = "request_cancelled"
    code = "image_request_cancelled"

    def __init__(self, message: str = "图片任务已取消。") -> None:
        super().__init__(message)


class ImageTaskDeadline:
    """一次图片任务唯一的单调时钟截止线。

    同一个实例必须贯穿排队、账号等待、上传、生成、轮询、下载和存储。
    子流程只能使用剩余预算，绝不能重新创建一份完整的超时预算。
    """

    def __init__(
        self,
        timeout_secs: float,
        *,
        started_at: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.timeout_secs = max(0.001, float(timeout_secs))
        self.started_at = time.monotonic() if started_at is None else float(started_at)
        self.expires_at = self.started_at + self.timeout_secs
        self._cancel_event = cancel_event or threading.Event()
        self._cancel_reason = ""
        self._cancel_lock = threading.Lock()

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @property
    def cancel_reason(self) -> str:
        with self._cancel_lock:
            return self._cancel_reason

    def cancel(self, reason: str = "客户端已断开连接") -> None:
        with self._cancel_lock:
            if not self._cancel_reason:
                self._cancel_reason = str(reason or "图片任务已取消")
        self._cancel_event.set()

    def remaining(self) -> float:
        return max(0.0, self.expires_at - time.monotonic())

    def check(self, phase: str = "图片生成") -> float:
        if self.cancelled:
            raise ImageTaskCancelled(self.cancel_reason or "图片任务已取消。")
        remaining = self.remaining()
        if remaining <= 0:
            raise ImageTaskDeadlineExceeded(self.timeout_secs, phase)
        return remaining

    def network_timeout(self, maximum_secs: float, phase: str = "上游请求") -> float:
        """返回不超过剩余总预算的网络超时，且避免传入零导致客户端无限等待。"""

        remaining = self.check(phase)
        return max(0.05, min(float(maximum_secs), remaining))

    def sleep(self, seconds: float, phase: str = "图片任务等待") -> None:
        """执行可取消、受总截止时间约束的等待。"""

        requested = max(0.0, float(seconds))
        if requested <= 0:
            self.check(phase)
            return
        remaining = self.check(phase)
        wait_for = min(requested, remaining)
        cancelled = self._cancel_event.wait(wait_for)
        if cancelled:
            raise ImageTaskCancelled(self.cancel_reason or "图片任务已取消。")
        self.check(phase)


@dataclass
class _WorkItem(Generic[T]):
    sequence: int
    name: str
    func: Callable[..., T]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    deadline: ImageTaskDeadline
    queue_deadline: float
    weight: int
    future: Future[T] = field(default_factory=Future)
    started: Future[bool] = field(default_factory=Future)
    enqueued_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None


class ImageTaskJob(Generic[T]):
    """供 API/后台任务持有的有界执行句柄。"""

    def __init__(self, runtime: "ImageTaskRuntime", item: _WorkItem[T]) -> None:
        self.runtime = runtime
        self.item = item

    @property
    def deadline(self) -> ImageTaskDeadline:
        return self.item.deadline

    def cancel(self, reason: str = "客户端已断开连接") -> None:
        self.item.deadline.cancel(reason)
        self.runtime.wake()

    def add_done_callback(self, callback: Callable[[Future[T]], object]) -> None:
        """注册底层完成回调；后台任务用它处理尚未开跑就排队超时的情况。"""

        # concurrent Future 的回调会在 set_result 的线程同步执行。不能让持久化回调
        # 阻塞唯一调度线程，因此统一转交小型回调执行器。
        def dispatch_callback(future: Future[T]) -> None:
            try:
                self.runtime._callback_executor.submit(callback, future)
            except RuntimeError:
                # 进程退出、回调执行器已关闭时仍需完成收尾，不能遗留任务状态。
                callback(future)

        self.item.future.add_done_callback(dispatch_callback)

    def add_cleanup_callback(self, callback: Callable[[], object]) -> None:
        """注册资源清理；任务未开跑或排队超时也保证执行。"""

        def cleanup(_future: Future[T]) -> None:
            try:
                callback()
            except Exception as exc:  # pragma: no cover - 清理失败不能破坏 Future 完成链
                logger.warning({
                    "event": "image_runtime_cleanup_failed",
                    "task_name": self.item.name,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc)[:300],
                })

        # Future 回调会在 set_result/set_exception 所在线程同步执行。清理动作即使通常
        # 很快，也可能受慢磁盘或异常文件对象影响，因此必须转交独立回调执行器，
        # 绝不能占住图片调度锁或唯一调度线程。
        self.add_done_callback(cleanup)

    async def wait_started(self) -> None:
        wrapped = asyncio.wrap_future(self.item.started)
        try:
            # 调度线程本身若意外退出，也不能让流式请求永远卡在 HTTP 响应头之前。
            await asyncio.wait_for(asyncio.shield(wrapped), self.deadline.remaining() + 0.2)
        except ImageTaskDeadlineExceeded:
            raise
        except asyncio.TimeoutError as exc:
            wrapped.add_done_callback(self._consume_async_future)
            self.cancel("图片任务等待调度达到总时限")
            raise ImageTaskDeadlineExceeded(self.deadline.timeout_secs, "图片任务排队调度") from exc
        except asyncio.CancelledError:
            wrapped.add_done_callback(self._consume_async_future)
            self.cancel()
            raise

    async def result(self) -> T:
        """等待任务结果；客户端等待超时不提前释放仍在运行的工作许可。"""

        wrapped = asyncio.wrap_future(self.item.future)
        try:
            # 额外 0.2 秒只用于异常在线程与事件循环之间传递，不会扩展底层任务预算。
            return await asyncio.wait_for(asyncio.shield(wrapped), self.deadline.remaining() + 0.2)
        except ImageTaskDeadlineExceeded:
            # 本异常也继承 TimeoutError，必须保留工作函数给出的精确阶段。
            raise
        except asyncio.TimeoutError as exc:
            # 底层线程不可被 Python 强杀；它稍后结束时仍可能把异常写入包装 Future。
            # 主动消费该异常，既保留工作许可到真实结束，又避免事件循环报告
            # “Future exception was never retrieved”。
            wrapped.add_done_callback(self._consume_async_future)
            self.cancel("图片任务达到总时限")
            raise ImageTaskDeadlineExceeded(self.deadline.timeout_secs) from exc
        except asyncio.CancelledError:
            wrapped.add_done_callback(self._consume_async_future)
            self.cancel()
            raise

    @staticmethod
    def _consume_async_future(future: asyncio.Future[Any]) -> None:
        try:
            future.exception()
        except BaseException:
            pass


class _StreamBridge:
    """把专用工作线程中的同步迭代器安全桥接成异步迭代器。"""

    _ITEM = "item"
    _ERROR = "error"
    _END = "end"

    def __init__(self, deadline: ImageTaskDeadline, max_buffer: int = 64) -> None:
        self.deadline = deadline
        self.loop = asyncio.get_running_loop()
        self.queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue(maxsize=max_buffer)
        self.ready: Future[tuple[str, object]] = Future()
        self.closed = threading.Event()

    def _put_from_worker(self, kind: str, value: object) -> None:
        while not self.closed.is_set():
            self.deadline.check("图片流式响应")
            coroutine: Coroutine[Any, Any, None] = self.queue.put((kind, value))
            try:
                if self.loop.is_closed():
                    raise RuntimeError("图片流式事件循环已关闭")
                future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
            except Exception as exc:
                coroutine.close()
                self.deadline.cancel("图片流式客户端已断开")
                raise ImageTaskCancelled("图片流式客户端已断开。") from exc
            try:
                future.result(timeout=min(1.0, self.deadline.check("图片流式响应")))
                _try_set_future_result(self.ready, (kind, value))
                return
            except FutureTimeoutError:
                future.cancel()
                continue
            except Exception:
                future.cancel()
                self.deadline.cancel("图片流式客户端已断开")
                raise ImageTaskCancelled("图片流式客户端已断开。")

    def push(self, value: object) -> None:
        self._put_from_worker(self._ITEM, value)

    def fail(self, exc: BaseException) -> None:
        try:
            self._put_from_worker(self._ERROR, exc)
        except ImageTaskRuntimeError:
            pass

    def finish(self) -> None:
        try:
            self._put_from_worker(self._END, None)
        except ImageTaskRuntimeError:
            pass

    async def next(self) -> object:
        try:
            kind, value = await asyncio.wait_for(self.queue.get(), self.deadline.remaining() + 0.2)
        except asyncio.TimeoutError as exc:
            self.deadline.cancel("图片流式请求达到总时限")
            raise ImageTaskDeadlineExceeded(self.deadline.timeout_secs) from exc
        if kind == self._ITEM:
            return value
        if kind == self._ERROR:
            if isinstance(value, BaseException):
                raise value
            raise RuntimeError(str(value))
        raise StopAsyncIteration

    async def wait_ready(self, preflight_secs: float = 1.0) -> tuple[str, object] | None:
        """短暂等待首项/立即错误；不能因上游出图而数分钟不发送响应头。"""

        wrapped = asyncio.wrap_future(self.ready)
        try:
            wait_for = min(max(0.01, preflight_secs), self.deadline.remaining() + 0.2)
            return await asyncio.wait_for(asyncio.shield(wrapped), wait_for)
        except asyncio.TimeoutError:
            # 这只是 HTTP 状态预检窗口结束，不是任务总超时。流响应随后会立即发出
            # `: stream-open`，真正的总截止线仍由 bridge.next 强制执行。
            wrapped.add_done_callback(ImageTaskJob._consume_async_future)
            if self.deadline.remaining() <= 0:
                self.deadline.cancel("图片流式请求达到总时限")
                raise ImageTaskDeadlineExceeded(self.deadline.timeout_secs)
            return None
        except asyncio.CancelledError:
            wrapped.add_done_callback(ImageTaskJob._consume_async_future)
            self.close()
            raise

    def close(self) -> None:
        self.closed.set()
        self.deadline.cancel("图片流式客户端已断开")


class ImageStreamJob(ImageTaskJob[None]):
    def __init__(self, runtime: "ImageTaskRuntime", item: _WorkItem[None], bridge: _StreamBridge) -> None:
        super().__init__(runtime, item)
        self.bridge = bridge

    def __aiter__(self) -> "ImageStreamJob":
        return self

    async def __anext__(self) -> object:
        return await self.bridge.next()

    async def wait_ready(self, preflight_secs: float = 1.0) -> tuple[str, object] | None:
        return await self.bridge.wait_ready(preflight_secs)

    async def aclose(self) -> None:
        self.bridge.close()
        self.cancel("图片流式客户端已断开")


class ImageTaskRuntime:
    """全进程共享的图片任务隔离执行器与有界等待队列。

    调度线程只会把获得全局许可的任务提交给固定大小执行器，因此不存在
    ThreadPoolExecutor 的无界内部任务堆积。即使某个第三方调用异常卡住，也只会
    占住图片专用工作线程，不会再耗尽 FastAPI/AnyIO 的公共线程池。
    """

    MAX_EXECUTOR_WORKERS = 32

    def __init__(self, settings_getter: Callable[[], dict[str, int]] | None = None) -> None:
        self._settings_getter = settings_getter or self._load_settings
        self._condition = threading.Condition(threading.RLock())
        self._queue: deque[_WorkItem[Any]] = deque()
        self._running: dict[int, _WorkItem[Any]] = {}
        self._active_units = 0
        self._sequence = itertools.count(1)
        self._stopping = False
        self._executor = ThreadPoolExecutor(
            max_workers=self.MAX_EXECUTOR_WORKERS,
            thread_name_prefix="image-runtime",
        )
        self._callback_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="image-runtime-callback",
        )
        self._stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "rejected": 0,
            "queue_timeouts": 0,
            "task_timeouts": 0,
            "cancelled": 0,
            "max_active_units": 0,
            "max_queue_depth": 0,
        }
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            name="image-runtime-dispatcher",
            daemon=True,
        )
        self._dispatcher.start()

    @staticmethod
    def _load_settings() -> dict[str, int]:
        # 延迟导入避免 services.config 初始化时形成循环依赖。
        from services.config import config

        return config.get_image_task_runtime_settings()

    def settings(self) -> dict[str, int]:
        try:
            return dict(self._settings_getter())
        except Exception:
            return dict(DEFAULT_IMAGE_TASK_RUNTIME)

    def new_deadline(
        self,
        timeout_secs: float | None = None,
        *,
        started_at: float | None = None,
    ) -> ImageTaskDeadline:
        """创建总截止线；started_at 可由 HTTP 中间件传入以覆盖请求体解析时间。"""

        timeout = self.settings()["total_timeout_secs"] if timeout_secs is None else float(timeout_secs)
        return ImageTaskDeadline(timeout, started_at=started_at)

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def _reject_locked(self, message: str = "图片任务队列已满，请稍后重试。") -> None:
        self._stats["rejected"] += 1
        raise ImageTaskOverloadedError(message)

    def _enqueue(
        self,
        func: Callable[..., T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        deadline: ImageTaskDeadline,
        weight: int,
        name: str,
    ) -> _WorkItem[T]:
        settings = self.settings()
        normalized_weight = max(1, int(weight))
        max_concurrency = settings["max_concurrency"]
        now = time.monotonic()
        with self._condition:
            if self._stopping:
                self._reject_locked("图片任务运行时正在停止，请稍后重试。")
            if normalized_weight > max_concurrency:
                self._reject_locked(
                    f"本次请求需要 {normalized_weight} 个图片并发槽，但全局上限为 {max_concurrency}。"
                )

            reserved_units = self._active_units + sum(item.weight for item in self._queue)
            immediate_capacity = reserved_units + normalized_weight <= max_concurrency
            queue_timeout = settings["queue_timeout_secs"]
            if not immediate_capacity:
                # 队列上限按“等待中的请求数”硬限制。调度线程尚未转为 running 的
                # 即时任务也保守计入队列，宁可短暂早拒绝，也绝不让高 weight 请求
                # 绕过上限造成无界积压。
                if len(self._queue) >= settings["max_queue_size"] or queue_timeout <= 0:
                    self._reject_locked()

            queue_deadline = deadline.expires_at
            if not immediate_capacity:
                queue_deadline = min(queue_deadline, now + queue_timeout)
            item: _WorkItem[T] = _WorkItem(
                sequence=next(self._sequence),
                name=name,
                func=func,
                args=args,
                kwargs=kwargs,
                deadline=deadline,
                queue_deadline=queue_deadline,
                weight=normalized_weight,
            )
            self._queue.append(item)
            self._stats["submitted"] += 1
            self._stats["max_queue_depth"] = max(self._stats["max_queue_depth"], len(self._queue))
            self._condition.notify_all()
            return item

    def submit(
        self,
        func: Callable[..., T],
        *args: Any,
        deadline: ImageTaskDeadline | None = None,
        weight: int = 1,
        name: str = "image-task",
        **kwargs: Any,
    ) -> ImageTaskJob[T]:
        active_deadline = deadline or self.new_deadline()
        item = self._enqueue(func, args, kwargs, deadline=active_deadline, weight=weight, name=name)
        return ImageTaskJob(self, item)

    def submit_stream(
        self,
        func: Callable[..., object],
        *args: Any,
        deadline: ImageTaskDeadline | None = None,
        weight: int = 1,
        name: str = "image-stream",
        **kwargs: Any,
    ) -> ImageStreamJob:
        active_deadline = deadline or self.new_deadline()
        bridge = _StreamBridge(active_deadline)

        def produce() -> None:
            try:
                result = func(*args, **kwargs)
                if isinstance(result, dict):
                    bridge.push(result)
                else:
                    for item in result:  # type: ignore[union-attr]
                        active_deadline.check("图片流式生成")
                        bridge.push(item)
            except BaseException as exc:
                bridge.fail(exc)
                raise
            finally:
                bridge.finish()

        item = self._enqueue(produce, (), {}, deadline=active_deadline, weight=weight, name=name)
        return ImageStreamJob(self, item, bridge)

    def _expire_queued_locked(self, now: float) -> list[tuple[_WorkItem[Any], BaseException]]:
        if not self._queue:
            return []
        kept: deque[_WorkItem[Any]] = deque()
        expired: list[tuple[_WorkItem[Any], BaseException]] = []
        while self._queue:
            item = self._queue.popleft()
            error: BaseException | None = None
            if item.deadline.cancelled:
                error = ImageTaskCancelled(item.deadline.cancel_reason or "图片任务已取消。")
                self._stats["cancelled"] += 1
            elif now >= item.deadline.expires_at:
                error = ImageTaskDeadlineExceeded(item.deadline.timeout_secs)
                self._stats["task_timeouts"] += 1
            elif now >= item.queue_deadline:
                error = ImageTaskQueueTimeoutError()
                self._stats["queue_timeouts"] += 1
            if error is None:
                kept.append(item)
                continue
            # Future 完成会同步执行调用方回调；这里只记录结果，离开运行时锁后再完成。
            expired.append((item, error))
        self._queue = kept
        return expired

    def _next_runnable_index_locked(self, max_concurrency: int) -> int | None:
        available = max_concurrency - self._active_units
        if available <= 0:
            return None
        for index, item in enumerate(self._queue):
            if item.weight <= available:
                return index
        return None

    def _dispatch_loop(self) -> None:
        while True:
            try:
                if self._dispatch_once():
                    return
            except BaseException as exc:
                # 唯一调度线程绝不能因单个 Future/线程创建异常永久死亡。
                logger.error({
                    "event": "image_runtime_dispatcher_error",
                    "error_type": exc.__class__.__name__,
                    "error": str(exc)[:300],
                })
                with self._condition:
                    if self._stopping:
                        return
                    self._condition.wait(timeout=0.1)

    def _dispatch_once(self) -> bool:
        """执行一轮调度；返回 True 表示运行时已停止。"""

        expired: list[tuple[_WorkItem[Any], BaseException]] = []
        submit_failures: list[tuple[_WorkItem[Any], BaseException]] = []
        started_items: list[_WorkItem[Any]] = []
        with self._condition:
            if self._stopping:
                return True
            now = time.monotonic()
            expired = self._expire_queued_locked(now)
            settings = self.settings()
            dispatched = False
            while True:
                index = self._next_runnable_index_locked(settings["max_concurrency"])
                if index is None:
                    break
                item = self._queue[index]
                del self._queue[index]
                item.started_at = time.monotonic()
                self._running[item.sequence] = item
                self._active_units += item.weight
                self._stats["max_active_units"] = max(
                    self._stats["max_active_units"], self._active_units
                )
                try:
                    self._executor.submit(self._run_item, item)
                except BaseException as exc:
                    self._running.pop(item.sequence, None)
                    self._active_units = max(0, self._active_units - item.weight)
                    self._stats["failed"] += 1
                    submit_failures.append((item, exc))
                    continue
                started_items.append(item)
                dispatched = True
            if not (dispatched or expired or submit_failures):
                next_expiry = min(
                    (min(item.queue_deadline, item.deadline.expires_at) for item in self._queue),
                    default=now + 0.5,
                )
                self._condition.wait(timeout=max(0.01, min(0.5, next_expiry - now)))

        # 先释放 Condition，再完成 Future，避免任意调用方回调反向卡住调度和健康检查。
        for item, error in (*expired, *submit_failures):
            _try_set_future_exception(item.started, error)
            _try_set_future_exception(item.future, error)
        for item, error in submit_failures:
            logger.error({
                "event": "image_runtime_submit_failed",
                "task_name": item.name,
                "error_type": error.__class__.__name__,
                "error": str(error)[:300],
            })
        for item in started_items:
            _try_set_future_result(item.started, True)
        return False

    def _run_item(self, item: _WorkItem[Any]) -> None:
        error: BaseException | None = None
        result: Any = None
        try:
            item.deadline.check("图片任务执行")
            result = item.func(*item.args, **item.kwargs)
            item.deadline.check("图片任务收尾")
        except BaseException as exc:
            error = exc
        finally:
            with self._condition:
                self._running.pop(item.sequence, None)
                self._active_units = max(0, self._active_units - item.weight)
                if isinstance(error, ImageTaskDeadlineExceeded):
                    self._stats["task_timeouts"] += 1
                elif isinstance(error, ImageTaskCancelled):
                    self._stats["cancelled"] += 1
                elif error is not None:
                    self._stats["failed"] += 1
                else:
                    self._stats["completed"] += 1
                self._condition.notify_all()

            # 必须先释放全局工作许可再唤醒等待者。这样 max_queue_size=0 时，
            # 调用方收到前一个任务结果后立刻提交下一个任务也不会被误判为过载。
            if error is None:
                _try_set_future_result(item.future, result)
            else:
                _try_set_future_exception(item.future, error)
            if error is not None:
                logger.warning({
                    "event": "image_runtime_task_failed",
                    "task_name": item.name,
                    "error_type": error.__class__.__name__,
                    "error": str(error)[:300],
                })

    def status(self) -> dict[str, object]:
        settings = self.settings()
        now = time.monotonic()
        # /healthz 和进程看门狗不能反过来永久等待运行时锁。即使未来代码误在
        # 临界区执行了慢操作，也要在 50ms 内返回不健康状态，让容器自愈生效。
        if not self._condition.acquire(timeout=0.05):
            return {
                "healthy": False,
                "status_lock_busy": True,
                "dispatcher_alive": self._dispatcher.is_alive(),
                "active_jobs": -1,
                "active_units": -1,
                "queued_jobs": -1,
                "oldest_running_secs": 0.0,
                "overdue_jobs": -1,
                "oldest_overdue_secs": 0.0,
                "settings": settings,
                "counters": dict(self._stats),
            }
        try:
            running = list(self._running.values())
            oldest_running = max(
                (now - item.started_at for item in running if item.started_at is not None),
                default=0.0,
            )
            overdue = sum(1 for item in running if now >= item.deadline.expires_at)
            oldest_overdue = max(
                (now - item.deadline.expires_at for item in running if now >= item.deadline.expires_at),
                default=0.0,
            )
            dispatcher_alive = self._dispatcher.is_alive()
            return {
                "healthy": overdue == 0 and dispatcher_alive and not self._stopping,
                "status_lock_busy": False,
                "dispatcher_alive": dispatcher_alive,
                "active_jobs": len(running),
                "active_units": self._active_units,
                "queued_jobs": len(self._queue),
                "oldest_running_secs": round(oldest_running, 1),
                "overdue_jobs": overdue,
                "oldest_overdue_secs": round(oldest_overdue, 1),
                "settings": settings,
                "counters": dict(self._stats),
            }
        finally:
            self._condition.release()


image_task_runtime = ImageTaskRuntime()


def image_deadline_from_payload(payload: object) -> ImageTaskDeadline | None:
    """从内部请求载荷读取截止线；外部 JSON 无法伪造有效对象。"""

    if not isinstance(payload, dict):
        return None
    value = payload.get("_image_task_deadline")
    return value if isinstance(value, ImageTaskDeadline) else None

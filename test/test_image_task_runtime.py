from __future__ import annotations

import asyncio
import threading
import time
import unittest

from services.image_task_runtime import (
    ImageTaskCancelled,
    ImageTaskDeadline,
    ImageTaskDeadlineExceeded,
    ImageTaskOverloadedError,
    ImageTaskQueueTimeoutError,
    ImageTaskRuntime,
)


def _settings(
    *,
    total_timeout_secs: float = 2,
    max_concurrency: int = 2,
    max_queue_size: int = 8,
    queue_timeout_secs: float = 1,
) -> dict[str, int | float]:
    """为单元测试提供不经过生产配置下限裁剪的短时限。"""

    return {
        "total_timeout_secs": total_timeout_secs,
        "max_concurrency": max_concurrency,
        "max_queue_size": max_queue_size,
        "queue_timeout_secs": queue_timeout_secs,
    }


class ImageTaskDeadlineTests(unittest.TestCase):
    def test_runtime_deadline_can_include_time_before_handler_entry(self) -> None:
        runtime = ImageTaskRuntime(lambda: _settings(total_timeout_secs=1))  # type: ignore[arg-type]
        started_at = time.monotonic() - 0.4

        deadline = runtime.new_deadline(started_at=started_at)

        self.assertGreater(deadline.remaining(), 0.4)
        self.assertLess(deadline.remaining(), 0.7)

    def test_remaining_network_timeout_and_sleep_share_one_deadline(self) -> None:
        deadline = ImageTaskDeadline(0.12)
        initial_remaining = deadline.remaining()

        self.assertGreater(initial_remaining, 0)
        self.assertLessEqual(initial_remaining, 0.12)
        self.assertLessEqual(deadline.network_timeout(30), initial_remaining)

        started_at = time.monotonic()
        with self.assertRaises(ImageTaskDeadlineExceeded) as raised:
            deadline.sleep(1, "测试总截止时间")
        elapsed = time.monotonic() - started_at

        self.assertGreaterEqual(elapsed, 0.08)
        self.assertLess(elapsed, 0.5)
        self.assertEqual(raised.exception.phase, "测试总截止时间")

    def test_cancel_interrupts_sleep_and_preserves_reason(self) -> None:
        deadline = ImageTaskDeadline(2)
        timer = threading.Timer(0.05, deadline.cancel, args=("测试主动取消",))
        timer.start()
        try:
            started_at = time.monotonic()
            with self.assertRaises(ImageTaskCancelled) as raised:
                deadline.sleep(1)
            self.assertLess(time.monotonic() - started_at, 0.5)
            self.assertIn("测试主动取消", str(raised.exception))
        finally:
            timer.cancel()


class ImageTaskRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_cleanup_callback_never_holds_runtime_status_lock(self) -> None:
        runtime = ImageTaskRuntime(
            lambda: _settings(
                total_timeout_secs=2,
                max_concurrency=1,
                max_queue_size=1,
                queue_timeout_secs=0.08,
            )  # type: ignore[arg-type]
        )
        release_running = threading.Event()
        cleanup_started = threading.Event()
        release_cleanup = threading.Event()

        def blocked_work() -> str:
            release_running.wait(2)
            return "done"

        def slow_cleanup() -> None:
            cleanup_started.set()
            release_cleanup.wait(2)

        running = runtime.submit(blocked_work)
        await running.wait_started()
        queued = runtime.submit(lambda: "must-not-run")
        queued.add_cleanup_callback(slow_cleanup)
        result_task = asyncio.create_task(queued.result())
        try:
            self.assertTrue(await asyncio.to_thread(cleanup_started.wait, 1))
            started_at = time.monotonic()
            status = await asyncio.wait_for(asyncio.to_thread(runtime.status), timeout=0.3)
            self.assertLess(time.monotonic() - started_at, 0.2)
            self.assertFalse(status["status_lock_busy"])
            with self.assertRaises(ImageTaskQueueTimeoutError):
                await result_task
        finally:
            release_cleanup.set()
            release_running.set()
        self.assertEqual(await running.result(), "done")

    async def test_status_fails_fast_if_internal_lock_is_unexpectedly_busy(self) -> None:
        runtime = ImageTaskRuntime(lambda: _settings())  # type: ignore[arg-type]
        lock_held = threading.Event()
        release_lock = threading.Event()

        def hold_runtime_lock() -> None:
            with runtime._condition:
                lock_held.set()
                release_lock.wait(1)

        holder = threading.Thread(target=hold_runtime_lock)
        holder.start()
        self.assertTrue(await asyncio.to_thread(lock_held.wait, 1))
        try:
            started_at = time.monotonic()
            status = runtime.status()
            self.assertLess(time.monotonic() - started_at, 0.2)
            self.assertFalse(status["healthy"])
            self.assertTrue(status["status_lock_busy"])
        finally:
            release_lock.set()
            holder.join(timeout=1)

    async def test_global_concurrency_never_exceeds_configured_limit(self) -> None:
        runtime = ImageTaskRuntime(
            lambda: _settings(max_concurrency=2, max_queue_size=8)  # type: ignore[arg-type]
        )
        release = threading.Event()
        two_are_running = threading.Event()
        counter_lock = threading.Lock()
        active = 0
        maximum_active = 0

        def blocked_work(index: int) -> int:
            nonlocal active, maximum_active
            with counter_lock:
                active += 1
                maximum_active = max(maximum_active, active)
                if active == 2:
                    two_are_running.set()
            try:
                if not release.wait(2):
                    raise AssertionError("测试任务未按预期解除阻塞")
                return index
            finally:
                with counter_lock:
                    active -= 1

        jobs = [runtime.submit(blocked_work, index) for index in range(6)]
        self.assertTrue(await asyncio.to_thread(two_are_running.wait, 1))
        await asyncio.sleep(0.05)

        status = runtime.status()
        self.assertEqual(status["active_jobs"], 2)
        self.assertEqual(status["active_units"], 2)
        self.assertEqual(maximum_active, 2)

        release.set()
        results = await asyncio.gather(*(job.result() for job in jobs))
        self.assertEqual(results, list(range(6)))
        self.assertEqual(maximum_active, 2)
        self.assertLessEqual(runtime.status()["counters"]["max_active_units"], 2)  # type: ignore[index]

    async def test_fifty_short_tasks_stay_within_global_limit(self) -> None:
        runtime = ImageTaskRuntime(
            lambda: _settings(max_concurrency=4, max_queue_size=50)  # type: ignore[arg-type]
        )
        counter_lock = threading.Lock()
        active = 0
        maximum_active = 0

        def short_work(index: int) -> int:
            nonlocal active, maximum_active
            with counter_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                time.sleep(0.01)
                return index
            finally:
                with counter_lock:
                    active -= 1

        jobs = [runtime.submit(short_work, index) for index in range(50)]
        results = await asyncio.gather(*(job.result() for job in jobs))

        self.assertEqual(results, list(range(50)))
        self.assertLessEqual(maximum_active, 4)
        self.assertEqual(runtime.status()["counters"]["completed"], 50)  # type: ignore[index]

    async def test_weighted_jobs_respect_unit_limit(self) -> None:
        runtime = ImageTaskRuntime(
            lambda: _settings(max_concurrency=3, max_queue_size=4)  # type: ignore[arg-type]
        )
        release = threading.Event()

        def blocked_work(label: str) -> str:
            if not release.wait(2):
                raise AssertionError("测试任务未按预期解除阻塞")
            return label

        heavy = runtime.submit(blocked_work, "heavy", weight=2)
        await heavy.wait_started()
        light = runtime.submit(blocked_work, "light", weight=1)
        await light.wait_started()
        queued = runtime.submit(blocked_work, "queued", weight=2)
        await asyncio.sleep(0.05)

        self.assertFalse(queued.item.started.done())
        self.assertEqual(runtime.status()["active_units"], 3)
        release.set()
        self.assertEqual(
            await asyncio.gather(heavy.result(), light.result(), queued.result()),
            ["heavy", "light", "queued"],
        )

    async def test_full_queue_rejects_immediately(self) -> None:
        runtime = ImageTaskRuntime(
            lambda: _settings(max_concurrency=1, max_queue_size=1, queue_timeout_secs=1)  # type: ignore[arg-type]
        )
        release = threading.Event()

        def blocked_work() -> str:
            if not release.wait(2):
                raise AssertionError("测试任务未按预期解除阻塞")
            return "done"

        running = runtime.submit(blocked_work)
        await running.wait_started()
        queued = runtime.submit(lambda: "queued")

        with self.assertRaises(ImageTaskOverloadedError) as raised:
            runtime.submit(lambda: "must-not-run")
        self.assertEqual(raised.exception.status_code, 429)
        self.assertEqual(raised.exception.code, "image_queue_full")
        self.assertEqual(runtime.status()["queued_jobs"], 1)

        release.set()
        self.assertEqual(await running.result(), "done")
        self.assertEqual(await queued.result(), "queued")

    async def test_weighted_running_job_does_not_expand_explicit_queue_limit(self) -> None:
        runtime = ImageTaskRuntime(
            lambda: _settings(max_concurrency=4, max_queue_size=2, queue_timeout_secs=1)  # type: ignore[arg-type]
        )
        release = threading.Event()

        def occupies_all_units() -> str:
            if not release.wait(2):
                raise AssertionError("测试任务未按预期解除阻塞")
            return "done"

        running = runtime.submit(occupies_all_units, weight=4)
        await running.wait_started()
        queued_one = runtime.submit(lambda: 1)
        queued_two = runtime.submit(lambda: 2)

        # max_queue_size 表示真实等待任务数，不能因运行任务的 weight 较大而膨胀。
        try:
            with self.assertRaises(ImageTaskOverloadedError):
                runtime.submit(lambda: 3)
        finally:
            release.set()
        self.assertEqual(await running.result(), "done")
        self.assertEqual(await asyncio.gather(queued_one.result(), queued_two.result()), [1, 2])

    async def test_queued_job_fails_with_queue_timeout(self) -> None:
        runtime = ImageTaskRuntime(
            lambda: _settings(
                total_timeout_secs=2,
                max_concurrency=1,
                max_queue_size=1,
                queue_timeout_secs=0.12,
            )  # type: ignore[arg-type]
        )
        release = threading.Event()

        def blocked_work() -> str:
            if not release.wait(2):
                raise AssertionError("测试任务未按预期解除阻塞")
            return "done"

        running = runtime.submit(blocked_work)
        await running.wait_started()
        queued = runtime.submit(lambda: "must-not-run")

        with self.assertRaises(ImageTaskQueueTimeoutError) as started_error:
            await queued.wait_started()
        with self.assertRaises(ImageTaskQueueTimeoutError) as result_error:
            await queued.result()
        self.assertEqual(started_error.exception.code, "image_queue_timeout")
        self.assertEqual(result_error.exception.status_code, 429)
        self.assertEqual(runtime.status()["counters"]["queue_timeouts"], 1)  # type: ignore[index]

        release.set()
        self.assertEqual(await running.result(), "done")

    async def test_total_deadline_includes_time_spent_in_queue(self) -> None:
        runtime = ImageTaskRuntime(
            lambda: _settings(
                total_timeout_secs=2,
                max_concurrency=1,
                max_queue_size=1,
                queue_timeout_secs=1,
            )  # type: ignore[arg-type]
        )
        release = threading.Event()
        executed = threading.Event()

        def blocked_work() -> str:
            if not release.wait(2):
                raise AssertionError("测试任务未按预期解除阻塞")
            return "done"

        running = runtime.submit(blocked_work)
        await running.wait_started()
        queued = runtime.submit(
            lambda: executed.set(),
            deadline=ImageTaskDeadline(0.12),
        )
        try:
            with self.assertRaises(ImageTaskDeadlineExceeded):
                await queued.result()
            self.assertFalse(executed.is_set())
            self.assertEqual(runtime.status()["counters"]["task_timeouts"], 1)  # type: ignore[index]
        finally:
            release.set()
        self.assertEqual(await running.result(), "done")

    async def test_cooperative_task_obeys_total_deadline(self) -> None:
        runtime = ImageTaskRuntime(
            lambda: _settings(total_timeout_secs=1, max_concurrency=1)  # type: ignore[arg-type]
        )
        deadline = ImageTaskDeadline(0.12)

        def work() -> None:
            deadline.sleep(5, "测试图片生成")

        started_at = time.monotonic()
        job = runtime.submit(work, deadline=deadline)
        with self.assertRaises(ImageTaskDeadlineExceeded):
            await job.result()
        elapsed = time.monotonic() - started_at

        self.assertGreaterEqual(elapsed, 0.08)
        self.assertLess(elapsed, 0.5)
        self.assertEqual(runtime.status()["counters"]["task_timeouts"], 1)  # type: ignore[index]

    async def test_client_timeout_does_not_release_blocked_worker_slot_early(self) -> None:
        runtime = ImageTaskRuntime(
            lambda: _settings(
                total_timeout_secs=2,
                max_concurrency=1,
                max_queue_size=1,
                queue_timeout_secs=1,
            )  # type: ignore[arg-type]
        )
        release_stubborn_call = threading.Event()

        def ignores_cancellation() -> str:
            # 模拟无法被 Python 强制终止、也不检查 deadline 的第三方 SDK 调用。
            if not release_stubborn_call.wait(2):
                raise AssertionError("测试任务未按预期解除阻塞")
            return "late-result"

        stubborn = runtime.submit(
            ignores_cancellation,
            deadline=ImageTaskDeadline(0.05),
            name="stubborn-sdk-call",
        )
        await stubborn.wait_started()

        with self.assertRaises(ImageTaskDeadlineExceeded):
            await stubborn.result()

        # 客户端已经收到超时，但底层线程仍在运行；此时许可绝不能被伪释放。
        self.assertEqual(runtime.status()["active_units"], 1)
        following = runtime.submit(
            lambda: "following-result",
            deadline=ImageTaskDeadline(1.5),
        )
        await asyncio.sleep(0.08)
        self.assertFalse(following.item.started.done())
        self.assertEqual(runtime.status()["active_units"], 1)

        release_stubborn_call.set()
        self.assertEqual(await following.result(), "following-result")
        self.assertEqual(runtime.status()["active_units"], 0)

    async def test_sync_stream_is_bridged_from_dedicated_worker_thread(self) -> None:
        runtime = ImageTaskRuntime(
            lambda: _settings(total_timeout_secs=2, max_concurrency=1)  # type: ignore[arg-type]
        )
        worker_names: list[str] = []

        def generate():
            for index in range(3):
                worker_names.append(threading.current_thread().name)
                yield {"index": index}

        stream = runtime.submit_stream(generate)
        values = [item async for item in stream]
        await stream.result()

        self.assertEqual(values, [{"index": 0}, {"index": 1}, {"index": 2}])
        self.assertEqual(len(worker_names), 3)
        self.assertTrue(all(name.startswith("image-runtime") for name in worker_names))
        self.assertFalse(any(name == threading.current_thread().name for name in worker_names))

    async def test_stream_propagates_generator_error(self) -> None:
        runtime = ImageTaskRuntime(
            lambda: _settings(total_timeout_secs=2, max_concurrency=1)  # type: ignore[arg-type]
        )

        def generate():
            yield "first"
            raise ValueError("stream failed")

        stream = runtime.submit_stream(generate)
        self.assertEqual(await anext(stream), "first")
        with self.assertRaisesRegex(ValueError, "stream failed"):
            await anext(stream)
        with self.assertRaisesRegex(ValueError, "stream failed"):
            await stream.result()


if __name__ == "__main__":
    unittest.main()

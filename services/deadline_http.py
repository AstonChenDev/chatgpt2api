from __future__ import annotations

import threading
from typing import Any

from curl_cffi import CurlOpt, requests

from services.image_task_runtime import ImageTaskDeadline


def _bounded_timeout(value: object, deadline: ImageTaskDeadline) -> object:
    """把 curl_cffi 的数字/二元组超时压缩到当前剩余总预算。"""

    remaining = deadline.check("上游网络请求")
    if isinstance(value, tuple) and len(value) == 2:
        return tuple(max(0.05, min(float(item), remaining)) for item in value)
    if isinstance(value, (int, float)):
        return max(0.05, min(float(value), remaining))
    return max(0.05, remaining)


class DeadlineSession(requests.Session):
    """自动遵守图片任务总截止线的 curl_cffi Session。

    curl_cffi 在 `stream=True` 时，普通数字 timeout 主要约束连接和低速状态，
    并不等同于端到端总时限。因此流式请求还必须设置 libcurl 的
    `CURLOPT_TIMEOUT_MS`，确保上游持续滴流也无法突破总预算。
    """

    def __init__(self, *args: Any, image_deadline: ImageTaskDeadline | None = None, **kwargs: Any) -> None:
        self.image_deadline = image_deadline
        self._curl_options_lock = threading.RLock()
        super().__init__(*args, **kwargs)

    def request(self, method: str, url: str, *args: Any, **kwargs: Any):  # type: ignore[override]
        deadline = self.image_deadline
        if deadline is None:
            return super().request(method, url, *args, **kwargs)

        with self._curl_options_lock:
            # timeout 计算和父类读取 curl_options 必须在同一把锁内；否则并发请求
            # 等锁期间会消耗预算，普通请求也可能读到流请求的临时 TIMEOUT_MS。
            deadline.check("上游网络请求")
            positional = list(args)
            # curl_cffi 的 timeout 是 method/url 之后第 8 个可选位置参数。保留完整父类
            # 调用兼容性，同时内部代码仍推荐显式关键字。
            if "timeout" not in kwargs and len(positional) > 7:
                positional[7] = _bounded_timeout(positional[7], deadline)
            else:
                kwargs["timeout"] = _bounded_timeout(kwargs.get("timeout"), deadline)
            stream_enabled = bool(kwargs.get("stream"))
            if "stream" not in kwargs and len(positional) > 28:
                stream_enabled = bool(positional[28])
            if not stream_enabled:
                response = super().request(method, url, *positional, **kwargs)
                deadline.check("上游网络请求")
                return response

            hard_timeout_ms = max(50, int(deadline.check("上游流式请求") * 1000))
            marker = object()
            previous = self.curl_options.get(CurlOpt.TIMEOUT_MS, marker)
            self.curl_options[CurlOpt.TIMEOUT_MS] = hard_timeout_ms
            try:
                # curl_cffi 在此方法返回前已把 curl_options 应用到克隆的流式句柄，
                # 因此恢复 Session 字典不会取消当前请求的硬时限。
                return super().request(method, url, *positional, **kwargs)
            finally:
                if previous is marker:
                    self.curl_options.pop(CurlOpt.TIMEOUT_MS, None)
                else:
                    self.curl_options[CurlOpt.TIMEOUT_MS] = previous

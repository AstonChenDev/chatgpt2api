from __future__ import annotations

import os
from typing import Mapping


# 图片任务运行时使用独立配置块，避免与“单次轮询超时”等旧配置混淆。
# total_timeout_secs 同时覆盖图片与 PPT/PSD，从任务进入服务到下载/落盘完成的总预算。
DEFAULT_IMAGE_TASK_RUNTIME: dict[str, int] = {
    "total_timeout_secs": 300,
    "max_concurrency": 8,
    "max_queue_size": 16,
    "queue_timeout_secs": 5,
}

IMAGE_TASK_RUNTIME_ENV_VARS = {
    "total_timeout_secs": "CHATGPT2API_IMAGE_TASK_TOTAL_TIMEOUT_SECS",
    "max_concurrency": "CHATGPT2API_IMAGE_TASK_MAX_CONCURRENCY",
    "max_queue_size": "CHATGPT2API_IMAGE_TASK_MAX_QUEUE_SIZE",
    "queue_timeout_secs": "CHATGPT2API_IMAGE_TASK_QUEUE_TIMEOUT_SECS",
}

# 这些边界既防止误配置导致服务重新失去响应，也保留足够的生产调节空间。
IMAGE_TASK_RUNTIME_LIMITS = {
    "total_timeout_secs": (30, 1800),
    "max_concurrency": (1, 32),
    "max_queue_size": (0, 256),
    "queue_timeout_secs": (0, 60),
}


def _bounded_int(value: object, default: int, minimum: int, maximum: int) -> int:
    try:
        normalized = int(value)
    except (OverflowError, TypeError, ValueError):
        normalized = default
    return min(maximum, max(minimum, normalized))


def normalize_image_task_runtime_settings(value: object) -> dict[str, int]:
    """规范化图片任务运行时配置，并把所有字段限制在安全边界内。"""

    source = value if isinstance(value, Mapping) else {}
    normalized: dict[str, int] = {}
    for key, default in DEFAULT_IMAGE_TASK_RUNTIME.items():
        minimum, maximum = IMAGE_TASK_RUNTIME_LIMITS[key]
        normalized[key] = _bounded_int(source.get(key), default, minimum, maximum)
    return normalized


def image_task_runtime_settings_with_env(value: object) -> dict[str, int]:
    """返回最终生效配置；非空环境变量优先，空值继续回退配置文件。"""

    source = dict(value) if isinstance(value, Mapping) else {}
    for key, env_name in IMAGE_TASK_RUNTIME_ENV_VARS.items():
        env_value = os.getenv(env_name)
        if env_value is not None and env_value.strip():
            source[key] = env_value
    return normalize_image_task_runtime_settings(source)

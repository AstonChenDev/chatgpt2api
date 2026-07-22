from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import BinaryIO

from services.config import config


MEBIBYTE = 1024 * 1024


class InsufficientDiskSpaceError(RuntimeError):
    """写入后无法保留最低磁盘余量。"""


# 图片与可编辑文件会写入同一个 data 卷。进程内统一串行执行“检查余量 + 写入”，
# 避免多个并发任务都看到同一份剩余空间后同时写满磁盘。
_DISK_WRITE_LOCK = threading.Lock()


def _existing_parent(path: Path) -> Path:
    """返回可供 disk_usage 检查的最近已存在父目录。"""

    current = path.resolve()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def minimum_free_bytes() -> int:
    return max(0, int(config.image_min_free_mb)) * MEBIBYTE


def _ensure_space_unlocked(path: Path, incoming_bytes: int = 0) -> None:
    required = max(0, int(incoming_bytes))
    reserve = minimum_free_bytes()
    usage = shutil.disk_usage(_existing_parent(path))
    if usage.free - required < reserve:
        free_mb = usage.free // MEBIBYTE
        reserve_mb = reserve // MEBIBYTE
        raise InsufficientDiskSpaceError(
            f"本地磁盘空间不足（可用 {free_mb} MB，必须保留 {reserve_mb} MB），已停止写入"
        )


def ensure_disk_space(path: Path, incoming_bytes: int = 0) -> None:
    """只检查空间；适合在接收大响应前做快速预检。"""

    with _DISK_WRITE_LOCK:
        _ensure_space_unlocked(path, incoming_bytes)


def guarded_write_chunk(output: BinaryIO, path: Path, chunk: bytes) -> int:
    """在统一磁盘锁内复查余量并写入一个流式分块。"""

    with _DISK_WRITE_LOCK:
        _ensure_space_unlocked(path, len(chunk))
        return output.write(chunk)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """检查磁盘下限后原子落盘，异常时不留下半文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{threading.get_ident()}.{time.time_ns()}.part")
    with _DISK_WRITE_LOCK:
        _ensure_space_unlocked(path.parent, len(payload))
        try:
            with temp_path.open("xb") as output:
                output.write(payload)
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)

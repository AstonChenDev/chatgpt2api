from __future__ import annotations

import asyncio
import hashlib
import json
import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

from services.config import DATA_DIR
from services.protocol.error_response import anthropic_error_response, openai_error_response
from utils.helper import anthropic_sse_stream, sse_json_stream

LOG_TYPE_CALL = "call"
LOG_TYPE_ACCOUNT = "account"
INTERNAL_RESPONSE_KEYS = {"_account_email", "_conversation_id"}
MAX_LOG_FILE_BYTES = 20 * 1024 * 1024
LOG_TRIM_TARGET_BYTES = 10 * 1024 * 1024
_SSE_ENCODER_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="image-sse-encode")


class LogService:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _trim_locked(self) -> None:
        """把无限增长的 JSONL 日志裁到最近约 10MB，避免后台查询最终 OOM。"""

        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return
        if size < MAX_LOG_FILE_BYTES:
            return
        with self.path.open("rb") as source:
            source.seek(max(0, size - LOG_TRIM_TARGET_BYTES))
            tail = source.read(LOG_TRIM_TARGET_BYTES)
        # 从完整下一行开始，避免保留半段 JSON。
        newline = tail.find(b"\n")
        if newline >= 0:
            tail = tail[newline + 1 :]
        temporary = self.path.with_suffix(self.path.suffix + ".trim.tmp")
        try:
            temporary.write_bytes(tail)
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _legacy_id(raw_line: str, line_number: int) -> str:
        payload = f"{line_number}:{raw_line}".encode("utf-8", errors="ignore")
        return hashlib.sha1(payload).hexdigest()[:24]

    def _parse_line(self, raw_line: str, line_number: int) -> dict[str, Any] | None:
        try:
            item = json.loads(raw_line)
        except Exception:
            return None
        if not isinstance(item, dict):
            return None
        parsed = dict(item)
        parsed["id"] = str(parsed.get("id") or self._legacy_id(raw_line, line_number))
        return parsed

    @staticmethod
    def _serialize_item(item: dict[str, Any]) -> str:
        return json.dumps(item, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _matches_filters(item: dict[str, Any], *, type: str = "", start_date: str = "", end_date: str = "") -> bool:
        t = str(item.get("time") or "")
        day = t[:10]
        if type and item.get("type") != type:
            return False
        if start_date and day < start_date:
            return False
        if end_date and day > end_date:
            return False
        return True

    def add(self, type: str, summary: str = "", detail: dict[str, Any] | None = None, **data: Any) -> None:
        item = {
            "id": uuid4().hex,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": type,
            "summary": summary,
            "detail": detail or data,
        }
        with self._lock:
            self._trim_locked()
            with self.path.open("a", encoding="utf-8") as file:
                file.write(self._serialize_item(item) + "\n")

    def list(self, type: str = "", start_date: str = "", end_date: str = "", limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            if not self.path.exists():
                return []
            # 兼容升级前已过大的日志：首次查询也先裁剪，内存读取始终有硬上限。
            self._trim_locked()
            lines = self.path.read_text(encoding="utf-8").splitlines()
        items: list[dict[str, Any]] = []
        for line_number in range(len(lines) - 1, -1, -1):
            item = self._parse_line(lines[line_number], line_number)
            if item is None:
                continue
            if not self._matches_filters(item, type=type, start_date=start_date, end_date=end_date):
                continue
            items.append(item)
            if len(items) >= limit:
                break
        return items

    def delete(self, ids: list[str]) -> dict[str, int]:
        target_ids = {str(item or "").strip() for item in ids if str(item or "").strip()}
        with self._lock:
            if not self.path.exists() or not target_ids:
                return {"removed": 0}
            self._trim_locked()
            lines = self.path.read_text(encoding="utf-8").splitlines()
            kept_lines: list[str] = []
            removed = 0
            for line_number, raw_line in enumerate(lines):
                item = self._parse_line(raw_line, line_number)
                if item is None:
                    kept_lines.append(raw_line)
                    continue
                if str(item.get("id") or "") in target_ids:
                    removed += 1
                    continue
                kept_lines.append(self._serialize_item(item))
            content = "\n".join(kept_lines)
            if content:
                content += "\n"
            self.path.write_text(content, encoding="utf-8")
            return {"removed": removed}


log_service = LogService(DATA_DIR / "logs.jsonl")


def _collect_urls(value: object) -> list[str]:
    urls: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "url" and isinstance(item, str):
                urls.append(item)
            elif key == "urls" and isinstance(item, list):
                urls.extend(str(url) for url in item if isinstance(url, str))
            else:
                urls.extend(_collect_urls(item))
    elif isinstance(value, list):
        for item in value:
            urls.extend(_collect_urls(item))
    return urls


def _collect_account_emails(value: object) -> list[str]:
    emails: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"_account_email", "account_email"} and isinstance(item, str) and item.strip():
                emails.append(item.strip())
            else:
                emails.extend(_collect_account_emails(item))
    elif isinstance(value, list):
        for item in value:
            emails.extend(_collect_account_emails(item))
    return emails


def _collect_conversation_ids(value: object) -> list[str]:
    ids: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "_conversation_id" and isinstance(item, str) and item.strip():
                ids.append(item.strip())
            else:
                ids.extend(_collect_conversation_ids(item))
    elif isinstance(value, list):
        for item in value:
            ids.extend(_collect_conversation_ids(item))
    return ids


def _strip_internal_response_fields(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _strip_internal_response_fields(item)
            for key, item in value.items()
            if key not in INTERNAL_RESPONSE_KEYS
        }
    if isinstance(value, list):
        return [_strip_internal_response_fields(item) for item in value]
    return value


def _request_excerpt(text: object, limit: int = 1000) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _image_error_response(exc: Exception) -> JSONResponse:
    from services.protocol.conversation import public_image_error_message

    message = public_image_error_message(str(exc))
    if "no available image quota" in message.lower():
        return openai_error_response(
            {
                "error": {
                    "message": "no available image quota",
                    "type": "insufficient_quota",
                    "param": None,
                    "code": "insufficient_quota",
                }
            },
            429,
        )
    if hasattr(exc, "to_openai_error") and hasattr(exc, "status_code"):
        status_code = int(exc.status_code)
        headers = {"Retry-After": "5"} if status_code == 429 else None
        return JSONResponse(status_code=status_code, content=exc.to_openai_error(), headers=headers)
    return openai_error_response(message, 502)


def _protocol_error_response(exc: Exception, status_code: int, sse: str) -> JSONResponse:
    message = str(exc)
    if sse == "anthropic":
        return anthropic_error_response(message, status_code)
    return openai_error_response(message, status_code)


async def _async_sse_json_stream(items, *, image_job=None, sse: str = "openai"):
    """异步版 OpenAI SSE 编码器，避免同步迭代器回落到 AnyIO 公共线程池。

    image_job 由图片流传入：客户端可能在第一条 stream-open 注释发送时就断开，
    此时内层 async generator 尚未开始，只有最外层 finally 能可靠取消工作线程。
    """

    try:
        yield ": stream-open\n\n"
        iterator = items.__aiter__()
        pending_next: asyncio.Task[Any] | None = None
        try:
            while True:
                if pending_next is None:
                    pending_next = asyncio.create_task(anext(iterator))
                done, _ = await asyncio.wait({pending_next}, timeout=15.0)
                if not done:
                    # 上游可能数分钟才出第一张图；周期心跳防止客户端/代理把连接误判为卡死。
                    yield ": ping\n\n"
                    continue
                try:
                    item = pending_next.result()
                except StopAsyncIteration:
                    pending_next = None
                    break
                pending_next = None
                loop = asyncio.get_running_loop()
                encoded = await loop.run_in_executor(
                    _SSE_ENCODER_EXECUTOR,
                    lambda value=item: json.dumps(value, ensure_ascii=False),
                )
                if sse == "anthropic":
                    event = str(item.get("type") or "message_delta") if isinstance(item, dict) else "message_delta"
                    yield f"event: {event}\ndata: {encoded}\n\n"
                else:
                    yield f"data: {encoded}\n\n"
        except Exception as exc:
            from utils.log import logger

            logger.warning({
                "event": "sse_stream_error",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            })
            if sse == "anthropic":
                error = {"type": "error", "error": {"type": exc.__class__.__name__, "message": str(exc)}}
                yield f"event: error\ndata: {json.dumps(error, ensure_ascii=False)}\n\n"
            else:
                error = exc.to_openai_error() if hasattr(exc, "to_openai_error") else {
                    "error": {"message": str(exc), "type": exc.__class__.__name__}
                }
                yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
        finally:
            if pending_next is not None and not pending_next.done():
                pending_next.cancel()
        if sse != "anthropic":
            yield "data: [DONE]\n\n"
    finally:
        if image_job is not None and not image_job.bridge.closed.is_set():
            await image_job.aclose()


def _next_item(items):
    try:
        return True, next(items)
    except StopIteration:
        return False, None


@dataclass
class LoggedCall:
    identity: dict[str, object]
    endpoint: str
    model: str
    summary: str
    started: float = field(default_factory=time.time)
    request_text: str = ""
    request_shape: dict[str, int] | None = None

    async def run(self, handler, *args, sse: str = "openai"):
        from services.protocol.conversation import ImageGenerationError

        try:
            result = await run_in_threadpool(handler, *args)
        except ImageGenerationError as exc:
            self.log("调用失败", status="failed", error=str(exc), account_email=getattr(exc, "account_email", ""),
                     conversation_id=getattr(exc, "conversation_id", ""))
            return _image_error_response(exc)
        except HTTPException as exc:
            self.log("调用失败", status="failed", error=str(exc.detail))
            raise
        except Exception as exc:
            self.log("调用失败", status="failed", error=str(exc), account_email=getattr(exc, "account_email", ""))
            if self.endpoint.startswith("/v1/images"):
                return _image_error_response(exc)
            return _protocol_error_response(exc, 502, sse)

        if isinstance(result, dict):
            self.log("调用完成", result)
            response = dict(result)
            response.pop("_account_email", None)
            return response

        sender = anthropic_sse_stream if sse == "anthropic" else sse_json_stream
        try:
            has_first, first = await run_in_threadpool(_next_item, result)
        except ImageGenerationError as exc:
            self.log("调用失败", status="failed", error=str(exc), account_email=getattr(exc, "account_email", ""),
                     conversation_id=getattr(exc, "conversation_id", ""))
            return _image_error_response(exc)
        except HTTPException as exc:
            self.log("调用失败", status="failed", error=str(exc.detail))
            raise
        except Exception as exc:
            self.log("调用失败", status="failed", error=str(exc), account_email=getattr(exc, "account_email", ""))
            if self.endpoint.startswith("/v1/images"):
                return _image_error_response(exc)
            return _protocol_error_response(exc, 502, sse)
        if not has_first:
            self.log("流式调用结束")
            return StreamingResponse(sender(()), media_type="text/event-stream")
        return StreamingResponse(sender(self.stream(itertools.chain([first], result))), media_type="text/event-stream")

    async def run_image(
        self,
        handler,
        *args,
        stream: bool = False,
        deadline=None,
        weight: int = 1,
        sse: str = "openai",
        cleanup: Callable[[], object] | None = None,
    ):
        """通过图片专用有界执行器运行请求，完全隔离 AnyIO 公共线程池。

        非流式 handler 在专用线程中执行到结束；流式 handler 的创建和每次迭代也都
        在同一个受控图片任务中执行，避免 `stream=true` 绕过隔离。
        """

        from services.image_task_runtime import ImageTaskRuntimeError, image_task_runtime
        from services.protocol.conversation import ImageGenerationError

        cleanup_lock = threading.Lock()
        cleanup_done = False

        def cleanup_once() -> None:
            nonlocal cleanup_done
            if cleanup is None:
                return
            with cleanup_lock:
                if cleanup_done:
                    return
                cleanup_done = True
            cleanup()

        job = None
        try:
            if stream:
                job = image_task_runtime.submit_stream(
                    handler,
                    *args,
                    deadline=deadline,
                    weight=weight,
                    name=self.endpoint,
                )
                if cleanup is not None:
                    job.add_cleanup_callback(cleanup_once)
                # 在发送 HTTP 200 之前确认任务已离开队列；排队超时可正常返回 429。
                await job.wait_started()
                first = await job.wait_ready()
                if first is not None and first[0] == job.bridge._ERROR:
                    first_value = first[1]
                    job.bridge.closed.set()
                    if isinstance(first_value, BaseException):
                        raise first_value
                    raise RuntimeError(str(first_value))
                return StreamingResponse(
                    _async_sse_json_stream(self.stream_image(job), image_job=job, sse=sse),
                    media_type="text/event-stream",
                )

            job = image_task_runtime.submit(
                handler,
                *args,
                deadline=deadline,
                weight=weight,
                name=self.endpoint,
            )
            if cleanup is not None:
                job.add_cleanup_callback(cleanup_once)
            result = await job.result()
        except (ImageGenerationError, ImageTaskRuntimeError) as exc:
            self.log(
                "调用失败",
                status="failed",
                error=str(exc),
                account_email=getattr(exc, "account_email", ""),
                conversation_id=getattr(exc, "conversation_id", ""),
            )
            if sse == "anthropic":
                return _protocol_error_response(exc, int(getattr(exc, "status_code", 502)), sse)
            return _image_error_response(exc)
        except HTTPException as exc:
            self.log("调用失败", status="failed", error=str(exc.detail))
            raise
        except Exception as exc:
            self.log(
                "调用失败",
                status="failed",
                error=str(exc),
                account_email=getattr(exc, "account_email", ""),
            )
            return _protocol_error_response(exc, 502, sse)
        finally:
            # submit 在队列已满时会在返回 job 前抛错，此时没有 Future 回调清理资源。
            if job is None:
                cleanup_once()

        if not isinstance(result, dict):
            exc = RuntimeError("image handler returned an unexpected streaming result")
            self.log("调用失败", status="failed", error=str(exc))
            return _protocol_error_response(exc, 502, "openai")
        self.log("调用完成", result)
        response = dict(result)
        response.pop("_account_email", None)
        return response

    async def stream_image(self, job):
        """记录受控图片流，并在客户端断开时向底层任务发出协作式取消信号。"""

        urls: list[str] = []
        account_emails: list[str] = []
        conversation_ids: list[str] = []
        failed = False
        completed = False
        try:
            async for item in job:
                urls.extend(_collect_urls(item))
                account_emails.extend(_collect_account_emails(item))
                conversation_ids.extend(_collect_conversation_ids(item))
                yield _strip_internal_response_fields(item)
            completed = True
        except asyncio.CancelledError:
            failed = True
            self.log("流式调用取消", status="cancelled", error="客户端已断开连接", urls=urls)
            raise
        except Exception as exc:
            failed = True
            self.log(
                "流式调用失败",
                status="failed",
                error=str(exc),
                urls=urls,
                account_email=(account_emails[0] if account_emails else getattr(exc, "account_email", "")),
                conversation_id=(conversation_ids[0] if conversation_ids else getattr(exc, "conversation_id", "")),
            )
            raise
        finally:
            if completed:
                job.bridge.closed.set()
            else:
                await job.aclose()
            if not failed:
                self.log(
                    "流式调用结束",
                    urls=urls,
                    account_email=account_emails[0] if account_emails else "",
                    conversation_id=conversation_ids[0] if conversation_ids else "",
                )

    def stream(self, items):
        urls: list[str] = []
        account_emails: list[str] = []
        conversation_ids: list[str] = []
        failed = False
        try:
            for item in items:
                urls.extend(_collect_urls(item))
                account_emails.extend(_collect_account_emails(item))
                conversation_ids.extend(_collect_conversation_ids(item))
                yield _strip_internal_response_fields(item)
        except Exception as exc:
            failed = True
            self.log(
                "流式调用失败",
                status="failed",
                error=str(exc),
                urls=urls,
                account_email=(account_emails[0] if account_emails else getattr(exc, "account_email", "")),
                conversation_id=(conversation_ids[0] if conversation_ids else getattr(exc, "conversation_id", "")),
            )
            if self.endpoint.startswith("/v1/images") and not hasattr(exc, "to_openai_error"):
                from services.protocol.conversation import ImageGenerationError, public_image_error_message

                raise ImageGenerationError(public_image_error_message(str(exc))) from exc
            raise
        finally:
            if not failed:
                self.log("流式调用结束", urls=urls, account_email=account_emails[0] if account_emails else "",
                         conversation_id=conversation_ids[0] if conversation_ids else "")

    def log(self, suffix: str, result: object = None, status: str = "success", error: str = "",
            urls: list[str] | None = None, account_email: str = "", conversation_id: str = "") -> None:
        detail = {
            "key_id": self.identity.get("id"),
            "key_name": self.identity.get("name"),
            "role": self.identity.get("role"),
            "endpoint": self.endpoint,
            "model": self.model,
            "started_at": datetime.fromtimestamp(self.started).strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_ms": int((time.time() - self.started) * 1000),
            "status": status,
        }
        request_excerpt = _request_excerpt(self.request_text)
        if request_excerpt:
            detail["request_text"] = request_excerpt
        if self.request_shape:
            detail["request_shape"] = self.request_shape
        if error:
            detail["error"] = error
        email = str(account_email or "").strip()
        if not email:
            emails = _collect_account_emails(result)
            email = emails[0] if emails else ""
        if email:
            detail["account_email"] = email
        conv_id = str(conversation_id or "").strip()
        if not conv_id:
            conv_ids = _collect_conversation_ids(result)
            conv_id = conv_ids[0] if conv_ids else ""
        if conv_id:
            detail["conversation_id"] = conv_id
        collected_urls = [*(urls or []), *_collect_urls(result)]
        if collected_urls and not self.endpoint.startswith("/v1/search"):
            detail["urls"] = list(dict.fromkeys(collected_urls))
        log_service.add(LOG_TYPE_CALL, f"{self.summary}{suffix}", detail)

from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from urllib.parse import quote, urlparse

from curl_cffi import requests
from fastapi import HTTPException
from PIL import Image
from qcloud_cos import CosConfig, CosS3Client

from services.config import DATA_DIR, config
from services.disk_space_guard import InsufficientDiskSpaceError, atomic_write_bytes
from services.image_task_runtime import ImageTaskDeadline, ImageTaskRuntimeError
from utils.log import logger

IMAGE_INDEX_FILE = DATA_DIR / "image_index.json"
IMAGE_INDEX_LOCK = Lock()
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_STORED_IMAGE_BYTES = 25 * 1024 * 1024


class ImageStorageError(RuntimeError):
    pass


@dataclass(frozen=True)
class StoredImage:
    rel: str
    url: str
    storage: str
    size: int


def _clean(value: object) -> str:
    return str(value or "").strip()


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_relative_path(path: str) -> str:
    value = str(path or "").strip().replace("\\", "/").lstrip("/")
    if not value:
        raise HTTPException(status_code=404, detail="image not found")
    parts = Path(value).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise HTTPException(status_code=404, detail="image not found")
    return Path(*parts).as_posix()


def _image_dimensions(payload: bytes) -> tuple[int, int] | None:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            return image.size
    except Exception:
        return None


def _is_image_rel(path: str) -> bool:
    try:
        safe_rel = _safe_relative_path(path)
    except HTTPException:
        return False
    return Path(safe_rel).suffix.lower() in IMAGE_EXTENSIONS


def _local_image_path(relative_path: str) -> Path:
    rel = _safe_relative_path(relative_path)
    root = config.images_dir.resolve()
    path = (root / rel).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="image not found") from exc
    return path


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_json_object(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


class WebDAVClient:
    def __init__(self, settings: dict[str, object], deadline: ImageTaskDeadline | None = None):
        self.url = _clean(settings.get("webdav_url")).rstrip("/")
        self.username = _clean(settings.get("webdav_username"))
        self.password = _clean(settings.get("webdav_password"))
        self.root_path = _clean(settings.get("webdav_root_path")).strip("/")
        self.deadline = deadline
        self.session = requests.Session()

    def _auth_kwargs(self) -> dict[str, object]:
        return {"auth": (self.username, self.password)} if self.username or self.password else {}

    def _request(self, method: str, url: str, **kwargs):
        timeout = self.deadline.network_timeout(30, "WebDAV 图片存储") if self.deadline else 30
        response = self.session.request(method, url, timeout=timeout, **self._auth_kwargs(), **kwargs)
        if self.deadline:
            self.deadline.check("WebDAV 图片存储")
        if response.status_code >= 400 and not (method == "MKCOL" and response.status_code in {405}):
            raise ImageStorageError(f"WebDAV {method} failed: HTTP {response.status_code}")
        return response

    def remote_url(self, rel: str = "") -> str:
        parts = [part for part in [self.root_path, _safe_relative_path(rel) if rel else ""] if part]
        encoded = "/".join(quote(part, safe="") for item in parts for part in item.split("/") if part)
        return f"{self.url}/{encoded}" if encoded else self.url

    def ensure_dirs(self, rel: str) -> None:
        parts = [part for part in [self.root_path, Path(_safe_relative_path(rel)).parent.as_posix()] if part and part != "."]
        current = self.url
        for item in "/".join(parts).split("/"):
            if not item:
                continue
            current = f"{current}/{quote(item, safe='')}"
            timeout = self.deadline.network_timeout(30, "WebDAV 目录创建") if self.deadline else 30
            response = self.session.request("MKCOL", current, timeout=timeout, **self._auth_kwargs())
            if response.status_code in {201, 405}:
                continue
            if response.status_code >= 400:
                raise ImageStorageError(f"WebDAV MKCOL failed: HTTP {response.status_code}")

    def put(self, rel: str, payload: bytes, content_type: str = "image/png") -> str:
        self.ensure_dirs(rel)
        url = self.remote_url(rel)
        self._request("PUT", url, data=payload, headers={"Content-Type": content_type})
        return url

    def get(self, rel: str) -> bytes:
        response = self._request("GET", self.remote_url(rel), stream=True)
        chunks: list[bytes] = []
        total = 0
        try:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if self.deadline:
                    self.deadline.check("WebDAV 图片读取")
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_STORED_IMAGE_BYTES:
                    raise ImageStorageError("stored image exceeds 25MB limit")
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            response.close()

    def delete(self, rel: str) -> bool:
        response = self.session.request("DELETE", self.remote_url(rel), timeout=30, **self._auth_kwargs())
        if response.status_code in {200, 202, 204, 404}:
            return response.status_code != 404
        raise ImageStorageError(f"WebDAV DELETE failed: HTTP {response.status_code}")

    def test(self) -> dict[str, object]:
        if not self.url:
            return {"ok": False, "status": 0, "error": "WebDAV URL is required"}
        if urlparse(self.url).scheme not in {"http", "https"}:
            return {"ok": False, "status": 0, "error": "invalid WebDAV URL"}
        test_rel = ".chatgpt2api_webdav_test.txt"
        try:
            self.put(test_rel, b"chatgpt2api webdav test\n", content_type="text/plain")
            self.delete(test_rel)
            return {"ok": True, "status": 200, "error": None}
        except ImageStorageError as exc:
            return {"ok": False, "status": 0, "error": str(exc)}
        except Exception as exc:
            return {"ok": False, "status": 0, "error": str(exc) or exc.__class__.__name__}
        finally:
            self.session.close()


class COSClient:
    def __init__(self, settings: dict[str, object], deadline: ImageTaskDeadline | None = None):
        self.secret_id = _clean(settings.get("cos_secret_id"))
        self.secret_key = _clean(settings.get("cos_secret_key"))
        self.region = _clean(settings.get("cos_region"))
        self.bucket = _clean(settings.get("cos_bucket"))
        self.path_prefix = _clean(settings.get("cos_path_prefix")).strip("/")
        self.deadline = deadline
        self._client = None

    @property
    def client(self) -> CosS3Client:
        if self._client is None:
            timeout = self.deadline.network_timeout(30, "COS 图片存储") if self.deadline else 30
            config_cos = CosConfig(
                Region=self.region,
                SecretId=self.secret_id,
                SecretKey=self.secret_key,
                Timeout=max(1, int(timeout)),
            )
            # SDK 默认会额外重试 3 次，每次都可能耗尽初始化时的 timeout，导致
            # 一次上传突破总截止线数倍。受控图片任务由上层决定回退，因此禁用 SDK 重试。
            self._client = CosS3Client(config_cos, retry=0 if self.deadline is not None else 3)
        return self._client

    def remote_url(self, rel: str) -> str:
        public_base_url = _clean(config.get_image_storage_settings().get("public_base_url"))
        safe_rel = _safe_relative_path(rel)
        prefix = f"{self.path_prefix}/" if self.path_prefix else ""
        key = f"{prefix}{safe_rel}"
        if public_base_url:
            return f"{public_base_url.rstrip('/')}/{key.lstrip('/')}"
        return f"https://{self.bucket}.cos.{self.region}.myqcloud.com/{key.lstrip('/')}"

    def put(self, rel: str, payload: bytes, content_type: str = "image/png") -> str:
        safe_rel = _safe_relative_path(rel)
        prefix = f"{self.path_prefix}/" if self.path_prefix else ""
        key = f"{prefix}{safe_rel}"
        try:
            if self.deadline:
                self.deadline.check("COS 图片上传")
            self.client.put_object(
                Bucket=self.bucket,
                Body=payload,
                Key=key,
                EnableMD5=True,
                ContentType=content_type
            )
            if self.deadline:
                self.deadline.check("COS 图片上传")
        except ImageTaskRuntimeError:
            raise
        except Exception as exc:
            raise ImageStorageError(f"COS put_object failed: {exc}")
        return self.remote_url(rel)

    def get(self, rel: str) -> bytes:
        safe_rel = _safe_relative_path(rel)
        prefix = f"{self.path_prefix}/" if self.path_prefix else ""
        key = f"{prefix}{safe_rel}"
        try:
            if self.deadline:
                self.deadline.check("COS 图片读取")
            response = self.client.get_object(
                Bucket=self.bucket,
                Key=key
            )
            fp = response['Body'].get_raw_stream()
            chunks: list[bytes] = []
            total = 0
            try:
                while True:
                    if self.deadline:
                        self.deadline.check("COS 图片读取")
                    chunk = fp.read(64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_STORED_IMAGE_BYTES:
                        raise ImageStorageError("stored image exceeds 25MB limit")
                    chunks.append(chunk)
                return b"".join(chunks)
            finally:
                close = getattr(fp, "close", None)
                if callable(close):
                    close()
        except ImageTaskRuntimeError:
            raise
        except Exception as exc:
            raise ImageStorageError(f"COS get_object failed: {exc}")

    def delete(self, rel: str) -> bool:
        safe_rel = _safe_relative_path(rel)
        prefix = f"{self.path_prefix}/" if self.path_prefix else ""
        key = f"{prefix}{safe_rel}"
        try:
            self.client.delete_object(
                Bucket=self.bucket,
                Key=key
            )
            return True
        except Exception as exc:
            raise ImageStorageError(f"COS delete_object failed: {exc}")

    def test(self) -> dict[str, object]:
        if not self.secret_id or not self.secret_key or not self.region or not self.bucket:
            return {"ok": False, "status": 0, "error": "COS configuration fields are required"}
        test_rel = ".chatgpt2api_cos_test.txt"
        try:
            self.put(test_rel, b"chatgpt2api cos test\n", content_type="text/plain")
            self.delete(test_rel)
            return {"ok": True, "status": 200, "error": None}
        except Exception as exc:
            return {"ok": False, "status": 0, "error": str(exc) or exc.__class__.__name__}


class ImageStorageService:
    def __init__(self, index_file: Path = IMAGE_INDEX_FILE):
        self.index_file = index_file
        self._index_lock = IMAGE_INDEX_LOCK

    def settings(self) -> dict[str, object]:
        return config.get_image_storage_settings()

    def mode(self) -> str:
        return _clean(self.settings().get("mode")) or "local"

    def _load_index(self) -> dict[str, dict[str, object]]:
        raw = _read_json_object(self.index_file)
        items = raw.get("items")
        if not isinstance(items, dict):
            return {}
        return {str(key): value for key, value in items.items() if isinstance(value, dict)}

    def _load_clean_index(self) -> dict[str, dict[str, object]]:
        items = self._load_index()
        return {rel: item for rel, item in items.items() if _is_image_rel(rel)}

    def _save_index(self, items: dict[str, dict[str, object]]) -> None:
        _write_json_object(self.index_file, {"items": items})

    def _public_url(self, rel: str, base_url: str | None = None, storage_mode: str | None = None) -> str:
        settings = self.settings()
        mode = storage_mode or _clean(settings.get("mode"))
        if mode == "local":
            # 网络存储失败回退本地后，不能继续返回 COS/CDN 域名下不存在的链接。
            return f"{(base_url or config.base_url).rstrip('/')}/images/{_safe_relative_path(rel)}"
        if mode == "cos":
            return COSClient(settings).remote_url(rel)
        public_base_url = _clean(settings.get("public_base_url"))
        if public_base_url:
            return f"{public_base_url.rstrip('/')}/{_safe_relative_path(rel)}"
        return f"{(base_url or config.base_url).rstrip('/')}/images/{_safe_relative_path(rel)}"

    def make_relative_path(self, image_data: bytes) -> str:
        file_hash = hashlib.md5(image_data).hexdigest()
        filename = f"{int(time.time())}_{file_hash}.png"
        relative_dir = Path(time.strftime("%Y"), time.strftime("%m"), time.strftime("%d"))
        return f"{relative_dir.as_posix()}/{filename}"

    def save(
        self,
        image_data: bytes,
        base_url: str | None = None,
        *,
        deadline: ImageTaskDeadline | None = None,
    ) -> StoredImage:
        # 在线请求不执行全目录清理，避免文件数量增长后把清理耗时算进客户端请求；
        # 后台清理调度器仍会按原逻辑执行。
        if deadline is None:
            config.cleanup_old_images()
        else:
            deadline.check("图片结果保存")
        rel = self.make_relative_path(image_data)
        mode = self.mode()
        if mode not in {"local", "webdav", "both", "cos"}:
            mode = "local"
        stored_local = False
        stored_webdav = False
        stored_cos = False
        remote_url = ""

        if mode in {"local", "both"}:
            if deadline:
                deadline.check("本地图片保存")
            path = _local_image_path(rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                atomic_write_bytes(path, image_data)
            except InsufficientDiskSpaceError as exc:
                raise ImageStorageError(str(exc)) from exc
            stored_local = True

        if mode in {"webdav", "both"}:
            webdav = None
            try:
                # deadline=None 时保持旧构造方式，兼容已有扩展/测试中的单参数客户端。
                webdav = (
                    WebDAVClient(self.settings(), deadline=deadline)
                    if deadline is not None
                    else WebDAVClient(self.settings())
                )
                remote_url = webdav.put(rel, image_data)
                stored_webdav = True
            except ImageTaskRuntimeError:
                raise
            except Exception as exc:
                if mode == "both":
                    logger.warning(f"WebDAV upload failed: {exc}")
                else:
                    logger.warning(f"WebDAV upload failed, fallback to local storage: {exc}")
                    if deadline:
                        deadline.check("WebDAV 失败后本地保存")
                    path = _local_image_path(rel)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        atomic_write_bytes(path, image_data)
                    except InsufficientDiskSpaceError as disk_exc:
                        raise ImageStorageError(str(disk_exc)) from disk_exc
                    stored_local = True
            finally:
                session = getattr(webdav, "session", None)
                if session is not None:
                    session.close()

        if mode == "cos":
            try:
                cos = (
                    COSClient(self.settings(), deadline=deadline)
                    if deadline is not None
                    else COSClient(self.settings())
                )
                remote_url = cos.put(rel, image_data)
                stored_cos = True
            except ImageTaskRuntimeError:
                raise
            except Exception as exc:
                logger.warning(f"COS upload failed, fallback to local storage: {exc}")
                if deadline:
                    deadline.check("COS 失败后本地保存")
                path = _local_image_path(rel)
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    atomic_write_bytes(path, image_data)
                except InsufficientDiskSpaceError as disk_exc:
                    raise ImageStorageError(str(disk_exc)) from disk_exc
                stored_local = True

        dimensions = _image_dimensions(image_data)
        item = {
            "rel": rel,
            "path": rel,
            "name": Path(rel).name,
            "date": "-".join(rel.split("/")[:3]),
            "size": len(image_data),
            "created_at": _now_iso(),
            "storage": "cos" if stored_cos else ("both" if stored_local and stored_webdav else ("webdav" if stored_webdav else "local")),
            "local": stored_local,
            "webdav": stored_webdav,
            "cos": stored_cos,
            "remote_url": remote_url,
        }
        if dimensions:
            item["width"], item["height"] = dimensions
        if deadline is not None:
            acquired = self._index_lock.acquire(timeout=deadline.network_timeout(10, "图片索引写入"))
            if not acquired:
                deadline.check("图片索引写入")
                raise ImageStorageError("image index lock timeout")
        else:
            self._index_lock.acquire()
        try:
            items = self._load_clean_index()
            items[rel] = item
            self._save_index(items)
        finally:
            self._index_lock.release()
        if deadline:
            deadline.check("图片结果保存")
        return StoredImage(rel=rel, url=self._public_url(rel, base_url, storage_mode=str(item["storage"])), storage=str(item["storage"]), size=len(image_data))

    def get_bytes(self, rel: str, deadline: ImageTaskDeadline | None = None) -> bytes:
        safe_rel = _safe_relative_path(rel)
        if not _is_image_rel(safe_rel):
            raise HTTPException(status_code=404, detail="image not found")
        path = _local_image_path(safe_rel)
        if path.is_file():
            if deadline:
                deadline.check("本地图片读取")
            payload = path.read_bytes()
            if len(payload) > MAX_STORED_IMAGE_BYTES:
                raise ImageStorageError("stored image exceeds 25MB limit")
            if deadline:
                deadline.check("本地图片读取")
            return payload
        item = self._load_clean_index().get(safe_rel, {})
        if item.get("webdav"):
            client = (
                WebDAVClient(self.settings(), deadline=deadline)
                if deadline is not None
                else WebDAVClient(self.settings())
            )
            try:
                return client.get(safe_rel)
            finally:
                session = getattr(client, "session", None)
                close = getattr(session, "close", None)
                if callable(close):
                    close()
        if item.get("cos"):
            client = (
                COSClient(self.settings(), deadline=deadline)
                if deadline is not None
                else COSClient(self.settings())
            )
            return client.get(safe_rel)
        raise HTTPException(status_code=404, detail="image not found")

    def exists(self, rel: str) -> bool:
        safe_rel = _safe_relative_path(rel)
        if not _is_image_rel(safe_rel):
            return False
        if _local_image_path(safe_rel).is_file():
            return True
        item = self._load_clean_index().get(safe_rel, {})
        return bool(item.get("webdav")) or bool(item.get("cos"))

    def has_local(self, rel: str) -> bool:
        safe_rel = _safe_relative_path(rel)
        return _is_image_rel(safe_rel) and _local_image_path(safe_rel).is_file()

    def list_items(self, base_url: str, start_date: str = "", end_date: str = "") -> list[dict[str, object]]:
        with self._index_lock:
            indexed = self._load_clean_index()
            root = config.images_dir
            changed = False
            for path in root.rglob("*"):
                if not path.is_file() or not _is_image_rel(path.name):
                    continue
                rel = path.relative_to(root).as_posix()
                if rel in indexed:
                    continue
                dimensions = None
                try:
                    dimensions = _image_dimensions(path.read_bytes())
                except Exception:
                    dimensions = None
                indexed[rel] = {
                    "rel": rel,
                    "path": rel,
                    "name": path.name,
                    "date": "-".join(rel.split("/")[:3]) if len(rel.split("/")) >= 4 else datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d"),
                    "size": path.stat().st_size,
                    "created_at": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "storage": "local",
                    "local": True,
                    "webdav": False,
                    "cos": False,
                    **({"width": dimensions[0], "height": dimensions[1]} if dimensions else {}),
                }
                changed = True

            items: list[dict[str, object]] = []
            for rel, item in list(indexed.items()):
                if not _is_image_rel(rel):
                    indexed.pop(rel, None)
                    changed = True
                    continue
                local = _local_image_path(rel).is_file()
                webdav = bool(item.get("webdav"))
                cos = bool(item.get("cos"))
                if not local and not webdav and not cos:
                    indexed.pop(rel, None)
                    changed = True
                    continue
                
                if local and webdav:
                    storage = "both"
                elif cos:
                    storage = "cos"
                elif webdav:
                    storage = "webdav"
                else:
                    storage = "local"

                if item.get("local") != local or item.get("storage") != storage or item.get("cos") != cos:
                    item = {
                        **item,
                        "local": local,
                        "cos": cos,
                        "storage": storage,
                    }
                    indexed[rel] = item
                    changed = True
                day = str(item.get("date") or "")
                if start_date and day < start_date:
                    continue
                if end_date and day > end_date:
                    continue
                items.append({
                    **item,
                    "rel": rel,
                    "path": rel,
                    "url": self._public_url(rel, base_url, storage_mode=str(item.get("storage"))),
                })
            if changed:
                self._save_index(indexed)
        items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return items

    def delete(self, rel: str) -> bool:
        safe_rel = _safe_relative_path(rel)
        removed = False
        path = _local_image_path(safe_rel)
        if path.is_file():
            path.unlink()
            removed = True
        with self._index_lock:
            items = self._load_clean_index()
            item = items.get(safe_rel, {})
            if item.get("webdav"):
                try:
                    removed = WebDAVClient(self.settings()).delete(safe_rel) or removed
                except ImageStorageError:
                    if not removed:
                        raise
            if item.get("cos"):
                try:
                    removed = COSClient(self.settings()).delete(safe_rel) or removed
                except ImageStorageError:
                    if not removed:
                        raise
            if safe_rel in items:
                items.pop(safe_rel, None)
                self._save_index(items)
        return removed

    def sync_all(self) -> dict[str, int]:
        settings = self.settings()
        mode = self.mode()
        if mode not in {"webdav", "both", "cos"}:
            raise ImageStorageError("网络图片存储（WebDAV/COS）未启用")
        uploaded = 0
        skipped = 0
        failed = 0
        with self._index_lock:
            items = self._load_clean_index()
            if mode == "cos":
                client = COSClient(settings)
                for path in sorted(config.images_dir.rglob("*")):
                    if not path.is_file() or not _is_image_rel(path.name):
                        continue
                    rel = path.relative_to(config.images_dir).as_posix()
                    item = items.get(rel, {})
                    if item.get("cos"):
                        skipped += 1
                        continue
                    try:
                        payload = path.read_bytes()
                        remote_url = client.put(rel, payload)
                        dimensions = _image_dimensions(payload)
                        items[rel] = {
                            **item,
                            "rel": rel,
                            "path": rel,
                            "name": path.name,
                            "date": "-".join(rel.split("/")[:3]) if len(rel.split("/")) >= 4 else datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d"),
                            "size": len(payload),
                            "created_at": str(item.get("created_at") or datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")),
                            "storage": "cos",
                            "local": True,
                            "webdav": False,
                            "cos": True,
                            "remote_url": remote_url,
                            **({"width": dimensions[0], "height": dimensions[1]} if dimensions else {}),
                        }
                        uploaded += 1
                    except Exception:
                        failed += 1
            else:
                client = WebDAVClient(settings)
                for path in sorted(config.images_dir.rglob("*")):
                    if not path.is_file() or not _is_image_rel(path.name):
                        continue
                    rel = path.relative_to(config.images_dir).as_posix()
                    item = items.get(rel, {})
                    if item.get("webdav"):
                        skipped += 1
                        continue
                    try:
                        payload = path.read_bytes()
                        remote_url = client.put(rel, payload)
                        dimensions = _image_dimensions(payload)
                        items[rel] = {
                            **item,
                            "rel": rel,
                            "path": rel,
                            "name": path.name,
                            "date": "-".join(rel.split("/")[:3]) if len(rel.split("/")) >= 4 else datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d"),
                            "size": len(payload),
                            "created_at": str(item.get("created_at") or datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")),
                            "storage": "both" if item.get("local") else "webdav",
                            "local": bool(item.get("local")),
                            "webdav": True,
                            "remote_url": remote_url,
                            **({"width": dimensions[0], "height": dimensions[1]} if dimensions else {}),
                        }
                        uploaded += 1
                    except Exception:
                        failed += 1
            self._save_index(items)
        return {"uploaded": uploaded, "skipped": skipped, "failed": failed}

    def test_webdav(self) -> dict[str, object]:
        return WebDAVClient(self.settings()).test()

    def test_cos(self) -> dict[str, object]:
        return COSClient(self.settings()).test()

    def test_connection(self) -> dict[str, object]:
        mode = self.mode()
        if mode == "cos":
            return self.test_cos()
        return self.test_webdav()


image_storage_service = ImageStorageService()

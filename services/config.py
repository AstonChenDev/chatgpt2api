from __future__ import annotations

import copy
from dataclasses import dataclass
import errno
import json
import os
import sys
import threading
from pathlib import Path
import time

from services.storage.base import StorageBackend
from services.image_task_runtime_config import (
    image_task_runtime_settings_with_env,
    normalize_image_task_runtime_settings,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CONFIG_FILE = BASE_DIR / "config.json"
VERSION_FILE = BASE_DIR / "VERSION"
BACKUP_STATE_FILE = DATA_DIR / "backup_state.json"
RUNTIME_CONFIG_FILE_NAME = "runtime_config.json"
_CONFIG_WRITE_LOCK = threading.RLock()

DEFAULT_BACKUP_INCLUDE = {
    "config": True,
    "cpa": True,
    "sub2api": True,
    "logs": True,
    "image_tasks": True,
    "accounts_snapshot": True,
    "auth_keys_snapshot": True,
    "images": False,
}

DEFAULT_IMAGE_STORAGE = {
    "enabled": False,
    "mode": "local",
    "webdav_url": "",
    "webdav_username": "",
    "webdav_password": "",
    "webdav_root_path": "chatgpt2api/images",
    "cos_secret_id": "",
    "cos_secret_key": "",
    "cos_region": "",
    "cos_bucket": "",
    "cos_path_prefix": "generated/",
    "public_base_url": "",
}

IMAGE_STORAGE_ENV_VARS = {
    "enabled": "CHATGPT2API_IMAGE_STORAGE_ENABLED",
    "mode": "CHATGPT2API_IMAGE_STORAGE_MODE",
    "cos_secret_id": "CHATGPT2API_COS_SECRET_ID",
    "cos_secret_key": "CHATGPT2API_COS_SECRET_KEY",
    "cos_region": "CHATGPT2API_COS_REGION",
    "cos_bucket": "CHATGPT2API_COS_BUCKET",
    "cos_path_prefix": "CHATGPT2API_COS_PATH_PREFIX",
    "public_base_url": "CHATGPT2API_IMAGE_PUBLIC_BASE_URL",
}

# 这些字段属于凭据，设置接口只能返回“是否已配置”，绝不能返回真实值。
IMAGE_STORAGE_SECRET_FIELDS = {
    "webdav_password",
    "cos_secret_id",
    "cos_secret_key",
}

DEFAULT_CHAT_COMPLETION_CACHE = {
    "enabled": True,
    "ttl_seconds": 60,
    "max_entries": 256,
    "dedupe_inflight": True,
    "stream_cache": True,
    "normalize_messages": True,
    "drop_adjacent_duplicates": True,
    "drop_assistant_history": False,
}

DEFAULT_PROXY_RUNTIME_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/145.0.0.0 Safari/537.36"
)

DEFAULT_PROXY_RUNTIME = {
    "enabled": False,
    "egress_mode": "direct",
    "proxy_url": "",
    "resource_proxy_url": "",
    "skip_ssl_verify": False,
    "reset_session_status_codes": [403],
    "clearance": {
        "enabled": False,
        "mode": "none",
        "cf_cookies": "",
        "cf_clearance": "",
        "user_agent": DEFAULT_PROXY_RUNTIME_USER_AGENT,
        "browser": "chrome",
        "flaresolverr_url": "",
        "timeout_sec": 60,
        "refresh_interval": 3600,
        "warm_up_on_start": False,
    },
}

DEFAULT_THIRD_PARTY_APPS = {
    "infinite_canvas": {
        "enabled": False,
        "url": "https://canvas.best",
    },
}


def _normalize_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        return default
    if value is None:
        return default
    return bool(value)


def _normalize_positive_int(value: object, default: int, minimum: int = 0) -> int:
    try:
        normalized = int(value)
    except (OverflowError, TypeError, ValueError):
        normalized = default
    return max(minimum, normalized)


def _normalize_backup_include(value: object) -> dict[str, bool]:
    source = value if isinstance(value, dict) else {}
    normalized = dict(DEFAULT_BACKUP_INCLUDE)
    for key in normalized:
        normalized[key] = _normalize_bool(source.get(key), normalized[key])
    return normalized


def _normalize_backup_settings(value: object) -> dict[str, object]:
    source = value if isinstance(value, dict) else {}
    return {
        "enabled": _normalize_bool(source.get("enabled"), False),
        "provider": "cloudflare_r2",
        "account_id": str(source.get("account_id") or "").strip(),
        "access_key_id": str(source.get("access_key_id") or "").strip(),
        "secret_access_key": str(source.get("secret_access_key") or "").strip(),
        "bucket": str(source.get("bucket") or "").strip(),
        "prefix": str(source.get("prefix") or "backups").strip().strip("/") or "backups",
        "interval_minutes": _normalize_positive_int(source.get("interval_minutes"), 360, 1),
        "rotation_keep": _normalize_positive_int(source.get("rotation_keep"), 10, 0),
        "encrypt": _normalize_bool(source.get("encrypt"), False),
        "passphrase": str(source.get("passphrase") or "").strip(),
        "include": _normalize_backup_include(source.get("include")),
    }


def _normalize_backup_state(value: object) -> dict[str, object]:
    source = value if isinstance(value, dict) else {}
    return {
        "last_started_at": str(source.get("last_started_at") or "").strip() or None,
        "last_finished_at": str(source.get("last_finished_at") or "").strip() or None,
        "last_status": str(source.get("last_status") or "idle").strip() or "idle",
        "last_error": str(source.get("last_error") or "").strip() or None,
        "last_object_key": str(source.get("last_object_key") or "").strip() or None,
    }


def _normalize_image_storage_settings(value: object) -> dict[str, object]:
    source = value if isinstance(value, dict) else {}
    mode = str(source.get("mode") or "local").strip().lower()
    if mode not in {"local", "webdav", "both", "cos"}:
        mode = "local"
    enabled = _normalize_bool(source.get("enabled"), False)
    if not enabled:
        mode = "local"
    root_path = str(source.get("webdav_root_path") or DEFAULT_IMAGE_STORAGE["webdav_root_path"]).strip().strip("/")
    cos_path_prefix = str(source.get("cos_path_prefix") or DEFAULT_IMAGE_STORAGE["cos_path_prefix"]).strip().strip("/")
    cos_path_prefix = f"{cos_path_prefix}/" if cos_path_prefix else ""
    return {
        "enabled": enabled,
        "mode": mode,
        "webdav_url": str(source.get("webdav_url") or "").strip().rstrip("/"),
        "webdav_username": str(source.get("webdav_username") or "").strip(),
        "webdav_password": str(source.get("webdav_password") or "").strip(),
        "webdav_root_path": root_path or str(DEFAULT_IMAGE_STORAGE["webdav_root_path"]),
        "cos_secret_id": str(source.get("cos_secret_id") or "").strip(),
        "cos_secret_key": str(source.get("cos_secret_key") or "").strip(),
        "cos_region": str(source.get("cos_region") or "").strip(),
        "cos_bucket": str(source.get("cos_bucket") or "").strip(),
        "cos_path_prefix": cos_path_prefix,
        "public_base_url": str(source.get("public_base_url") or "").strip().rstrip("/"),
    }


def _image_storage_settings_with_env(value: object) -> dict[str, object]:
    """合并图片存储设置；非空环境变量优先，空变量继续使用持久化配置。"""

    source = dict(value) if isinstance(value, dict) else {}
    for key, env_name in IMAGE_STORAGE_ENV_VARS.items():
        env_value = os.getenv(env_name)
        if env_value is not None and env_value.strip():
            source[key] = env_value
    return _normalize_image_storage_settings(source)


def _image_storage_environment_overrides() -> set[str]:
    """返回明确由环境变量接管的字段名，不包含任何变量值。"""

    return {
        key
        for key, env_name in IMAGE_STORAGE_ENV_VARS.items()
        if (os.getenv(env_name) or "").strip()
    }


def _public_image_storage_settings(value: object) -> dict[str, object]:
    """生成后台可见配置；凭据只暴露存在标记，不返回明文。"""

    effective = _image_storage_settings_with_env(value)
    public = dict(effective)
    for key in IMAGE_STORAGE_SECRET_FIELDS:
        public[key] = ""
        public[f"has_{key}"] = bool(str(effective.get(key) or "").strip())
    public["managed_by_env"] = bool(_image_storage_environment_overrides())
    return public


def _merge_image_storage_update(current: object, incoming: object) -> dict[str, object]:
    """合并后台更新，同时保留空凭据并忽略环境变量接管的字段。"""

    merged = dict(current) if isinstance(current, dict) else {}
    updates = dict(incoming) if isinstance(incoming, dict) else {}
    environment_overrides = _image_storage_environment_overrides()
    # 已由环境变量提供的凭据不应在持久化文件里保留副本。
    for key in IMAGE_STORAGE_SECRET_FIELDS & environment_overrides:
        merged.pop(key, None)
    for key, value in updates.items():
        # `has_*` 和 `managed_by_env` 仅用于后台展示，绝不持久化。
        if key == "managed_by_env" or key.startswith("has_"):
            continue
        # 环境变量的值不能经由后台请求被复制进 config.json。
        if key in environment_overrides:
            continue
        # 凭据输入框留空表示保持原值，避免保存其他设置时误清空密钥。
        if key in IMAGE_STORAGE_SECRET_FIELDS and not str(value or "").strip():
            continue
        merged[key] = value
    return _normalize_image_storage_settings(merged)


def _normalize_chat_completion_cache_settings(value: object) -> dict[str, object]:
    source = value if isinstance(value, dict) else {}
    return {
        "enabled": _normalize_bool(source.get("enabled"), DEFAULT_CHAT_COMPLETION_CACHE["enabled"]),
        "ttl_seconds": _normalize_positive_int(
            source.get("ttl_seconds"),
            int(DEFAULT_CHAT_COMPLETION_CACHE["ttl_seconds"]),
            0,
        ),
        "max_entries": _normalize_positive_int(
            source.get("max_entries"),
            int(DEFAULT_CHAT_COMPLETION_CACHE["max_entries"]),
            1,
        ),
        "dedupe_inflight": _normalize_bool(
            source.get("dedupe_inflight"),
            bool(DEFAULT_CHAT_COMPLETION_CACHE["dedupe_inflight"]),
        ),
        "stream_cache": _normalize_bool(
            source.get("stream_cache"),
            bool(DEFAULT_CHAT_COMPLETION_CACHE["stream_cache"]),
        ),
        "normalize_messages": _normalize_bool(
            source.get("normalize_messages"),
            bool(DEFAULT_CHAT_COMPLETION_CACHE["normalize_messages"]),
        ),
        "drop_adjacent_duplicates": _normalize_bool(
            source.get("drop_adjacent_duplicates"),
            bool(DEFAULT_CHAT_COMPLETION_CACHE["drop_adjacent_duplicates"]),
        ),
        "drop_assistant_history": _normalize_bool(
            source.get("drop_assistant_history"),
            bool(DEFAULT_CHAT_COMPLETION_CACHE["drop_assistant_history"]),
        ),
    }


def _normalize_status_codes(value: object) -> list[int]:
    items = value if isinstance(value, list) else DEFAULT_PROXY_RUNTIME["reset_session_status_codes"]
    normalized: list[int] = []
    for item in items:
        if isinstance(item, bool):
            continue
        try:
            status = int(item)
        except (OverflowError, TypeError, ValueError):
            continue
        if 100 <= status <= 599 and status not in normalized:
            normalized.append(status)
    if not normalized:
        return list(DEFAULT_PROXY_RUNTIME["reset_session_status_codes"])
    return normalized


def _normalize_proxy_runtime_settings(value: object) -> dict[str, object]:
    source = value if isinstance(value, dict) else {}
    default_clearance = DEFAULT_PROXY_RUNTIME["clearance"]
    clearance_source = source.get("clearance") if isinstance(source.get("clearance"), dict) else {}

    egress_mode = str(source.get("egress_mode") or DEFAULT_PROXY_RUNTIME["egress_mode"]).strip().lower()
    if egress_mode not in {"direct", "single_proxy"}:
        egress_mode = str(DEFAULT_PROXY_RUNTIME["egress_mode"])

    clearance_mode = str(clearance_source.get("mode") or default_clearance["mode"]).strip().lower()
    if clearance_mode not in {"none", "manual", "flaresolverr"}:
        clearance_mode = str(default_clearance["mode"])

    user_agent = str(clearance_source.get("user_agent") or default_clearance["user_agent"]).strip()
    browser = str(clearance_source.get("browser") or default_clearance["browser"]).strip()

    existing_clearance_cookies = str(source.get("_existing_cf_cookies") or "").strip()
    existing_cf_clearance = str(source.get("_existing_cf_clearance") or "").strip()
    cf_cookies = str(clearance_source.get("cf_cookies") or "").strip()
    cf_clearance = str(clearance_source.get("cf_clearance") or "").strip()
    if not cf_cookies and _normalize_bool(clearance_source.get("has_cf_cookies"), False):
        cf_cookies = existing_clearance_cookies
    if not cf_clearance and _normalize_bool(clearance_source.get("has_cf_clearance"), False):
        cf_clearance = existing_cf_clearance

    return {
        "enabled": _normalize_bool(source.get("enabled"), bool(DEFAULT_PROXY_RUNTIME["enabled"])),
        "egress_mode": egress_mode,
        "proxy_url": str(source.get("proxy_url") or "").strip(),
        "resource_proxy_url": str(source.get("resource_proxy_url") or "").strip(),
        "skip_ssl_verify": _normalize_bool(
            source.get("skip_ssl_verify"),
            bool(DEFAULT_PROXY_RUNTIME["skip_ssl_verify"]),
        ),
        "reset_session_status_codes": _normalize_status_codes(source.get("reset_session_status_codes")),
        "clearance": {
            "enabled": _normalize_bool(clearance_source.get("enabled"), bool(default_clearance["enabled"])),
            "mode": clearance_mode,
            "cf_cookies": cf_cookies,
            "cf_clearance": cf_clearance,
            "user_agent": user_agent or str(default_clearance["user_agent"]),
            "browser": browser or str(default_clearance["browser"]),
            "flaresolverr_url": str(clearance_source.get("flaresolverr_url") or "").strip(),
            "timeout_sec": _normalize_positive_int(
                clearance_source.get("timeout_sec"),
                int(default_clearance["timeout_sec"]),
                1,
            ),
            "refresh_interval": _normalize_positive_int(
                clearance_source.get("refresh_interval"),
                int(default_clearance["refresh_interval"]),
                60,
            ),
            "warm_up_on_start": _normalize_bool(
                clearance_source.get("warm_up_on_start"),
                bool(default_clearance["warm_up_on_start"]),
            ),
        },
    }


def _normalize_third_party_apps_settings(value: object) -> dict[str, object]:
    source = value if isinstance(value, dict) else {}
    canvas_source = source.get("infinite_canvas") if isinstance(source.get("infinite_canvas"), dict) else {}
    return {
        "infinite_canvas": {
            "enabled": _normalize_bool(canvas_source.get("enabled"), False),
            "url": str(canvas_source.get("url") or DEFAULT_THIRD_PARTY_APPS["infinite_canvas"]["url"]).strip(),
        },
    }


def _validate_image_storage_settings(settings: dict[str, object]) -> None:
    if not _normalize_bool(settings.get("enabled"), False):
        return
    mode = str(settings.get("mode") or "local").strip().lower()
    if mode in {"webdav", "both"}:
        if not str(settings.get("webdav_url") or "").strip():
            raise ValueError("启用 WebDAV 图片存储后必须填写 WebDAV URL")
        if not str(settings.get("webdav_password") or "").strip():
            raise ValueError("启用 WebDAV 图片存储后必须填写 WebDAV 密码")
    elif mode == "cos":
        if not str(settings.get("cos_secret_id") or "").strip():
            raise ValueError("启用 COS 图片存储后必须填写 Secret ID")
        if not str(settings.get("cos_secret_key") or "").strip():
            raise ValueError("启用 COS 图片存储后必须填写 Secret Key")
        if not str(settings.get("cos_region") or "").strip():
            raise ValueError("启用 COS 图片存储后必须填写 Region（存储桶地域）")
        if not str(settings.get("cos_bucket") or "").strip():
            raise ValueError("启用 COS 图片存储后必须填写 Bucket（存储桶名称）")


@dataclass(frozen=True)
class LoadedSettings:
    auth_key: str
    refresh_account_interval_minute: int


def _normalize_auth_key(value: object) -> str:
    return str(value or "").strip()


def _is_invalid_auth_key(value: object) -> bool:
    return _normalize_auth_key(value) == ""


def _read_json_object(path: Path, *, name: str) -> dict[str, object]:
    if not path.exists():
        return {}
    if path.is_dir():
        print(
            f"Warning: {name} at '{path}' is a directory, ignoring it and falling back to other configuration sources.",
            file=sys.stderr,
        )
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_json_object(path: Path, data: dict[str, object]) -> None:
    """以同目录原子替换方式写入 JSON，避免进程中断留下半个配置文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    with _CONFIG_WRITE_LOCK:
        try:
            temporary_path.write_text(payload, encoding="utf-8")
            # 配置可能包含凭据，默认仅允许文件所有者读取。
            temporary_path.chmod(0o600)
            try:
                temporary_path.replace(path)
            except OSError as exc:
                if exc.errno not in {errno.EBUSY, errno.EXDEV}:
                    raise
                # Docker 将 config.json 作为单文件 bind mount 时不能替换挂载点；
                # 退回原位覆盖并 fsync。data 目录内的动态配置仍使用原子替换。
                with path.open("w", encoding="utf-8") as target:
                    target.write(payload)
                    target.flush()
                    os.fsync(target.fileno())
                path.chmod(0o600)
        finally:
            temporary_path.unlink(missing_ok=True)


def _load_settings() -> LoadedSettings:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_config = _read_json_object(CONFIG_FILE, name="config.json")
    auth_key = _normalize_auth_key(os.getenv("CHATGPT2API_AUTH_KEY") or raw_config.get("auth-key"))
    if _is_invalid_auth_key(auth_key):
        raise ValueError(
            "❌ auth-key 未设置！\n"
            "请在环境变量 CHATGPT2API_AUTH_KEY 中设置，或者在 config.json 中填写 auth-key。"
        )

    try:
        refresh_interval = int(raw_config.get("refresh_account_interval_minute", 5))
    except (TypeError, ValueError):
        refresh_interval = 5

    return LoadedSettings(
        auth_key=auth_key,
        refresh_account_interval_minute=refresh_interval,
    )


class ConfigStore:
    def __init__(self, path: Path, runtime_config_path: Path | None = None):
        self.path = path
        # 动态图片运行参数放在 data 中，避免发布时 git reset 覆盖后台设置。
        self.runtime_config_path = runtime_config_path or path.parent / "data" / RUNTIME_CONFIG_FILE_NAME
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.data = self._load()
        self.runtime_data = _read_json_object(self.runtime_config_path, name=RUNTIME_CONFIG_FILE_NAME)
        self._storage_backend: StorageBackend | None = None
        if _is_invalid_auth_key(self.auth_key):
            raise ValueError(
                "❌ auth-key 未设置！\n"
                "请按以下任意一种方式解决：\n"
                "1. 在 Render 的 Environment 变量中添加：\n"
                "   CHATGPT2API_AUTH_KEY = your_real_auth_key\n"
                "2. 或者在 config.json 中填写：\n"
                '   "auth-key": "your_real_auth_key"'
            )

    def _load(self) -> dict[str, object]:
        return _read_json_object(self.path, name="config.json")

    @property
    def auth_key(self) -> str:
        return _normalize_auth_key(os.getenv("CHATGPT2API_AUTH_KEY") or self.data.get("auth-key"))

    @property
    def accounts_file(self) -> Path:
        return DATA_DIR / "accounts.json"

    @property
    def refresh_account_interval_minute(self) -> int:
        try:
            return int(self.data.get("refresh_account_interval_minute", 5))
        except (TypeError, ValueError):
            return 5

    @property
    def image_retention_days(self) -> int:
        try:
            return max(1, int(self.data.get("image_retention_days", 30)))
        except (TypeError, ValueError):
            return 30

    @property
    def image_min_free_mb(self) -> int:
        """图片与可编辑文件写盘后必须保留的空间，默认 500 MB。"""

        value = os.getenv("CHATGPT2API_IMAGE_MIN_FREE_MB")
        if value is None or not value.strip():
            value = self.data.get("image_min_free_mb", 500)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 500

    @property
    def image_poll_timeout_secs(self) -> int:
        try:
            return max(1, int(self.data.get("image_poll_timeout_secs", 120)))
        except (TypeError, ValueError):
            return 120

    @property
    def image_poll_interval_secs(self) -> float:
        try:
            return max(0.5, float(self.data.get("image_poll_interval_secs", 10.0)))
        except (TypeError, ValueError):
            return 10.0

    @property
    def image_poll_initial_wait_secs(self) -> float:
        """Image generation upstream takes ~30s; polling immediately wastes requests
        and trips a transient 429. Default 10s gives the conversation document time
        to commit before the first poll."""
        try:
            return max(0.0, float(self.data.get("image_poll_initial_wait_secs", 10.0)))
        except (TypeError, ValueError):
            return 10.0

    @property
    def image_account_concurrency(self) -> int:
        try:
            return max(1, int(self.data.get("image_account_concurrency", 3)))
        except (TypeError, ValueError):
            return 3

    @property
    def image_parallel_generation(self) -> bool:
        value = self.data.get("image_parallel_generation", True)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @property
    def image_settle_enabled(self) -> bool:
        """图片二次确认机制：找到 file_ids 后等待一段时间再次确认。"""
        value = self.data.get("image_settle_enabled", True)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @property
    def image_check_before_hit_enabled(self) -> bool:
        """先check再hit：通过轮询确认 file_ids 存在后再返回，而非仅依赖 SSE 事件。"""
        value = self.data.get("image_check_before_hit_enabled", True)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @property
    def image_remove_conversation_after_result(self) -> bool:
        """出图成功后异步隐藏 ChatGPT 本地对话记录。"""
        value = self.data.get("image_remove_conversation_after_result", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @property
    def image_settle_secs(self) -> float:
        """二次确认等待时间（秒）。"""
        try:
            return max(0.5, float(self.data.get("image_settle_secs", 2.0)))
        except (TypeError, ValueError):
            return 2.0

    @property
    def auto_remove_invalid_accounts(self) -> bool:
        value = self.data.get("auto_remove_invalid_accounts", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @property
    def auto_remove_rate_limited_accounts(self) -> bool:
        value = self.data.get("auto_remove_rate_limited_accounts", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @property
    def auto_relogin_after_refresh(self) -> bool:
        value = self.data.get("auto_relogin_after_refresh", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @property
    def log_levels(self) -> list[str]:
        levels = self.data.get("log_levels")
        if not isinstance(levels, list):
            return []
        allowed = {"debug", "info", "warning", "error"}
        return [level for item in levels if (level := str(item or "").strip().lower()) in allowed]

    @property
    def sensitive_words(self) -> list[str]:
        words = self.data.get("sensitive_words")
        return [word for item in words if (word := str(item or "").strip())] if isinstance(words, list) else []

    @property
    def ai_review(self) -> dict[str, object]:
        value = self.data.get("ai_review")
        return value if isinstance(value, dict) else {}

    @property
    def global_system_prompt(self) -> str:
        return str(self.data.get("global_system_prompt") or "").strip()

    @property
    def images_dir(self) -> Path:
        path = DATA_DIR / "images"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def image_thumbnails_dir(self) -> Path:
        path = DATA_DIR / "image_thumbnails"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cleanup_old_images(self) -> int:
        cutoff = time.time() - self.image_retention_days * 86400
        removed = 0
        for path in self.images_dir.rglob("*"):
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        for path in sorted((p for p in self.images_dir.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            try:
                path.rmdir()
            except OSError:
                pass
        return removed

    @property
    def base_url(self) -> str:
        return str(
            os.getenv("CHATGPT2API_BASE_URL")
            or self.data.get("base_url")
            or ""
        ).strip().rstrip("/")

    @property
    def app_version(self) -> str:
        try:
            value = VERSION_FILE.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return "0.0.0"
        return value or "0.0.0"

    def get(self) -> dict[str, object]:
        data = dict(self.data)
        data["refresh_account_interval_minute"] = self.refresh_account_interval_minute
        data["image_retention_days"] = self.image_retention_days
        data["image_min_free_mb"] = self.image_min_free_mb
        data["image_poll_timeout_secs"] = self.image_poll_timeout_secs
        data["image_poll_interval_secs"] = self.image_poll_interval_secs
        data["image_poll_initial_wait_secs"] = self.image_poll_initial_wait_secs
        data["image_account_concurrency"] = self.image_account_concurrency
        data["image_parallel_generation"] = self.image_parallel_generation
        data["image_remove_conversation_after_result"] = self.image_remove_conversation_after_result
        data["auto_remove_invalid_accounts"] = self.auto_remove_invalid_accounts
        data["auto_remove_rate_limited_accounts"] = self.auto_remove_rate_limited_accounts
        data["auto_relogin_after_refresh"] = self.auto_relogin_after_refresh
        data["log_levels"] = self.log_levels
        data["sensitive_words"] = self.sensitive_words
        data["ai_review"] = self.ai_review
        data["global_system_prompt"] = self.global_system_prompt
        data["backup"] = self.get_backup_settings()
        data["image_storage"] = self.get_public_image_storage_settings()
        data["image_task_runtime"] = self.get_image_task_runtime_settings()
        data["chat_completion_cache"] = self.get_chat_completion_cache_settings()
        data["proxy_runtime"] = self.get_public_proxy_runtime_settings()
        data["third_party_apps"] = self.get_third_party_apps_settings()
        data.pop("auth-key", None)
        return data

    def get_proxy_settings(self) -> str:
        return str(self.data.get("proxy") or "").strip()

    def get_proxy_runtime_settings(self) -> dict[str, object]:
        return _normalize_proxy_runtime_settings(self.data.get("proxy_runtime"))

    def get_public_proxy_runtime_settings(self) -> dict[str, object]:
        runtime = copy.deepcopy(self.get_proxy_runtime_settings())
        clearance = runtime.get("clearance") if isinstance(runtime.get("clearance"), dict) else {}
        if isinstance(clearance, dict):
            cf_cookies = str(clearance.get("cf_cookies") or "").strip()
            cf_clearance = str(clearance.get("cf_clearance") or "").strip()
            clearance["cf_cookies"] = ""
            clearance["cf_clearance"] = ""
            clearance["has_cf_cookies"] = bool(cf_cookies)
            clearance["has_cf_clearance"] = bool(cf_clearance)
        return runtime

    def get_third_party_apps_settings(self) -> dict[str, object]:
        return _normalize_third_party_apps_settings(self.data.get("third_party_apps"))

    def update(self, data: dict[str, object]) -> dict[str, object]:
        incoming_data = dict(data or {})
        incoming_auth_key = incoming_data.pop("auth-key", None)
        runtime_update_marker = object()
        incoming_runtime = incoming_data.pop("image_task_runtime", runtime_update_marker)

        with _CONFIG_WRITE_LOCK:
            next_data = dict(self.data)
            next_data.update(incoming_data)

            # 后台 GET 从不返回认证密钥；空值表示保留。若环境变量已接管，
            # 即使旧客户端主动提交，也不能把环境变量密钥复制回 config.json。
            auth_managed_by_env = bool((os.getenv("CHATGPT2API_AUTH_KEY") or "").strip())
            auth_removed_from_file = auth_managed_by_env and "auth-key" in next_data
            if auth_managed_by_env:
                next_data.pop("auth-key", None)
            elif str(incoming_auth_key or "").strip():
                next_data["auth-key"] = str(incoming_auth_key).strip()

            if "backup" in next_data:
                next_data["backup"] = _normalize_backup_settings(next_data.get("backup"))
            if "image_storage" in incoming_data:
                persisted_storage = _merge_image_storage_update(
                    self.data.get("image_storage"),
                    incoming_data.get("image_storage"),
                )
                # 校验最终有效配置，但只持久化不含环境变量值的部分。
                _validate_image_storage_settings(_image_storage_settings_with_env(persisted_storage))
                next_data["image_storage"] = persisted_storage
            elif "image_storage" in next_data:
                next_data["image_storage"] = _merge_image_storage_update(
                    next_data.get("image_storage"),
                    {},
                )

            next_runtime_data = dict(self.runtime_data)
            runtime_changed = incoming_runtime is not runtime_update_marker
            if runtime_changed:
                # 后台允许只提交一个字段；其余字段从 data 中的动态配置继承。
                previous_runtime = self.runtime_data.get("image_task_runtime")
                if not isinstance(previous_runtime, dict):
                    # 兼容旧版本：首次保存前仍可读取 config.json 中的旧配置块。
                    previous_runtime = self.data.get("image_task_runtime")
                previous_runtime = dict(previous_runtime) if isinstance(previous_runtime, dict) else {}
                if isinstance(incoming_runtime, dict):
                    previous_runtime.update(incoming_runtime)
                next_runtime_data["image_task_runtime"] = normalize_image_task_runtime_settings(previous_runtime)

            if "chat_completion_cache" in next_data:
                next_data["chat_completion_cache"] = _normalize_chat_completion_cache_settings(
                    next_data.get("chat_completion_cache")
                )
            if "third_party_apps" in next_data:
                next_data["third_party_apps"] = _normalize_third_party_apps_settings(next_data.get("third_party_apps"))
            if "proxy_runtime" in next_data:
                incoming_proxy_runtime = next_data.get("proxy_runtime")
                if isinstance(incoming_proxy_runtime, dict):
                    previous_clearance = self.get_proxy_runtime_settings().get("clearance")
                    if isinstance(previous_clearance, dict):
                        incoming_proxy_runtime = dict(incoming_proxy_runtime)
                        incoming_proxy_runtime["_existing_cf_cookies"] = previous_clearance.get("cf_cookies")
                        incoming_proxy_runtime["_existing_cf_clearance"] = previous_clearance.get("cf_clearance")
                next_data["proxy_runtime"] = _normalize_proxy_runtime_settings(incoming_proxy_runtime)
            next_data.pop("backup_state", None)

            # 动态运行参数只写入 data/runtime_config.json，避免发布覆盖。
            legacy_runtime_removed = next_data.pop("image_task_runtime", None) is not None
            if runtime_changed:
                _write_json_object(self.runtime_config_path, next_runtime_data)
            if incoming_data or legacy_runtime_removed or auth_removed_from_file or (
                not auth_managed_by_env and str(incoming_auth_key or "").strip()
            ):
                _write_json_object(self.path, next_data)
            self.data = next_data
            self.runtime_data = next_runtime_data
        return self.get()

    def get_backup_settings(self) -> dict[str, object]:
        return _normalize_backup_settings(self.data.get("backup"))

    def get_image_storage_settings(self) -> dict[str, object]:
        return _image_storage_settings_with_env(self.data.get("image_storage"))

    def get_public_image_storage_settings(self) -> dict[str, object]:
        return _public_image_storage_settings(self.data.get("image_storage"))

    def get_image_task_runtime_settings(self) -> dict[str, int]:
        persisted = self.runtime_data.get("image_task_runtime")
        if not isinstance(persisted, dict):
            # 读取旧配置作为一次性兼容回退；下次后台保存会迁移到 data 文件。
            persisted = self.data.get("image_task_runtime")
        return image_task_runtime_settings_with_env(persisted)

    def get_chat_completion_cache_settings(self) -> dict[str, object]:
        return _normalize_chat_completion_cache_settings(self.data.get("chat_completion_cache"))

    def get_storage_backend(self) -> StorageBackend:
        """获取存储后端实例（单例）"""
        if self._storage_backend is None:
            from services.storage.factory import create_storage_backend
            self._storage_backend = create_storage_backend(DATA_DIR)
        return self._storage_backend


def load_backup_state() -> dict[str, object]:
    return _normalize_backup_state(_read_json_object(BACKUP_STATE_FILE, name="backup_state.json"))


def save_backup_state(state: dict[str, object]) -> dict[str, object]:
    normalized = _normalize_backup_state(state)
    BACKUP_STATE_FILE.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return normalized


config = ConfigStore(CONFIG_FILE)

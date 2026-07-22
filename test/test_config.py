import errno
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_CONFIG_FILE = ROOT_DIR / "config.json"


class ConfigLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._created_root_config = False
        if not ROOT_CONFIG_FILE.exists():
            ROOT_CONFIG_FILE.write_text(json.dumps({"auth-key": "test-auth"}), encoding="utf-8")
            cls._created_root_config = True

        from services import config as config_module

        cls.config_module = config_module

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._created_root_config and ROOT_CONFIG_FILE.exists():
            ROOT_CONFIG_FILE.unlink()

    def test_load_settings_ignores_directory_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            data_dir = base_dir / "data"
            config_dir = base_dir / "config.json"
            os_auth_key = "env-auth"

            config_dir.mkdir()

            module = self.config_module
            old_base_dir = module.BASE_DIR
            old_data_dir = module.DATA_DIR
            old_config_file = module.CONFIG_FILE
            old_env_auth_key = module.os.environ.get("CHATGPT2API_AUTH_KEY")
            try:
                module.BASE_DIR = base_dir
                module.DATA_DIR = data_dir
                module.CONFIG_FILE = config_dir
                module.os.environ["CHATGPT2API_AUTH_KEY"] = os_auth_key

                settings = module._load_settings()

                self.assertEqual(settings.auth_key, os_auth_key)
                self.assertEqual(settings.refresh_account_interval_minute, 5)
            finally:
                module.BASE_DIR = old_base_dir
                module.DATA_DIR = old_data_dir
                module.CONFIG_FILE = old_config_file
                if old_env_auth_key is None:
                    module.os.environ.pop("CHATGPT2API_AUTH_KEY", None)
                else:
                    module.os.environ["CHATGPT2API_AUTH_KEY"] = old_env_auth_key

    def test_image_storage_environment_overrides_config_file(self) -> None:
        module = self.config_module
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps({
                "auth-key": "test-auth",
                "image_storage": {
                    "enabled": False,
                    "mode": "local",
                    "cos_region": "from-json",
                    "cos_bucket": "json-bucket",
                },
            }), encoding="utf-8")
            env = {
                "CHATGPT2API_IMAGE_STORAGE_ENABLED": "true",
                "CHATGPT2API_IMAGE_STORAGE_MODE": "cos",
                "CHATGPT2API_COS_SECRET_ID": "env-secret-id",
                "CHATGPT2API_COS_SECRET_KEY": "env-secret-key",
                "CHATGPT2API_COS_REGION": "ap-shanghai",
                "CHATGPT2API_COS_BUCKET": "env-bucket-123",
                "CHATGPT2API_COS_PATH_PREFIX": "production/images",
                "CHATGPT2API_IMAGE_PUBLIC_BASE_URL": "https://cdn.example.test/images/",
            }

            with mock.patch.dict(module.os.environ, env, clear=False):
                store = module.ConfigStore(config_path)
                settings = store.get_image_storage_settings()

            self.assertTrue(settings["enabled"])
            self.assertEqual(settings["mode"], "cos")
            self.assertEqual(settings["cos_secret_id"], "env-secret-id")
            self.assertEqual(settings["cos_secret_key"], "env-secret-key")
            self.assertEqual(settings["cos_region"], "ap-shanghai")
            self.assertEqual(settings["cos_bucket"], "env-bucket-123")
            self.assertEqual(settings["cos_path_prefix"], "production/images/")
            self.assertEqual(settings["public_base_url"], "https://cdn.example.test/images")

    def test_blank_image_storage_environment_keeps_config_fallback(self) -> None:
        module = self.config_module
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps({
                "auth-key": "test-auth",
                "image_storage": {
                    "enabled": True,
                    "mode": "cos",
                    "cos_secret_id": "json-secret-id",
                    "cos_secret_key": "json-secret-key",
                    "cos_region": "ap-guangzhou",
                    "cos_bucket": "json-bucket-123",
                },
            }), encoding="utf-8")
            blank_env = {
                env_name: ""
                for env_name in module.IMAGE_STORAGE_ENV_VARS.values()
            }

            with mock.patch.dict(module.os.environ, blank_env, clear=False):
                settings = module.ConfigStore(config_path).get_image_storage_settings()

            self.assertTrue(settings["enabled"])
            self.assertEqual(settings["mode"], "cos")
            self.assertEqual(settings["cos_secret_id"], "json-secret-id")
            self.assertEqual(settings["cos_region"], "ap-guangzhou")

    def test_minimum_free_space_is_configurable_with_environment_priority(self) -> None:
        module = self.config_module
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(
                json.dumps({"auth-key": "test-auth", "image_min_free_mb": 750}),
                encoding="utf-8",
            )
            with mock.patch.dict(
                module.os.environ,
                {"CHATGPT2API_IMAGE_MIN_FREE_MB": ""},
                clear=False,
            ):
                self.assertEqual(module.ConfigStore(config_path).image_min_free_mb, 750)
            with mock.patch.dict(
                module.os.environ,
                {"CHATGPT2API_IMAGE_MIN_FREE_MB": "900"},
                clear=False,
            ):
                self.assertEqual(module.ConfigStore(config_path).image_min_free_mb, 900)

    def test_public_settings_redact_environment_secrets_and_never_persist_them(self) -> None:
        module = self.config_module
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(
                json.dumps({
                    "auth-key": "file-admin-key",
                    "image_storage": {
                        "enabled": False,
                        "mode": "local",
                        "cos_secret_id": "legacy-file-secret-id",
                        "cos_secret_key": "legacy-file-secret-key",
                    },
                }),
                encoding="utf-8",
            )
            environment = {
                "CHATGPT2API_AUTH_KEY": "environment-admin-key",
                "CHATGPT2API_IMAGE_STORAGE_ENABLED": "true",
                "CHATGPT2API_IMAGE_STORAGE_MODE": "cos",
                "CHATGPT2API_COS_SECRET_ID": "environment-secret-id",
                "CHATGPT2API_COS_SECRET_KEY": "environment-secret-key",
                "CHATGPT2API_COS_REGION": "ap-shanghai",
                "CHATGPT2API_COS_BUCKET": "environment-bucket",
            }

            with mock.patch.dict(module.os.environ, environment, clear=True):
                store = module.ConfigStore(config_path)
                public = store.get()
                serialized_public = json.dumps(public)
                self.assertNotIn("environment-admin-key", serialized_public)
                self.assertNotIn("environment-secret-id", serialized_public)
                self.assertNotIn("environment-secret-key", serialized_public)
                self.assertEqual(public["image_storage"]["cos_secret_id"], "")
                self.assertEqual(public["image_storage"]["cos_secret_key"], "")
                self.assertTrue(public["image_storage"]["has_cos_secret_id"])
                self.assertTrue(public["image_storage"]["has_cos_secret_key"])
                self.assertTrue(public["image_storage"]["managed_by_env"])

                # 模拟后台保存整份公开配置，也不能把环境变量中的凭据反写到文件。
                store.update({
                    "auth-key": "environment-admin-key",
                    "image_storage": public["image_storage"],
                })

            persisted = config_path.read_text(encoding="utf-8")
            self.assertNotIn("environment-admin-key", persisted)
            self.assertNotIn("environment-secret-id", persisted)
            self.assertNotIn("environment-secret-key", persisted)
            self.assertNotIn("legacy-file-secret-id", persisted)
            self.assertNotIn("legacy-file-secret-key", persisted)
            self.assertNotIn("has_cos_secret_id", persisted)
            self.assertNotIn("managed_by_env", persisted)
            self.assertNotIn("auth-key", json.loads(persisted))

    def test_blank_secret_updates_preserve_existing_credentials(self) -> None:
        module = self.config_module
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(
                json.dumps({
                    "auth-key": "file-admin-key",
                    "image_storage": {
                        "enabled": True,
                        "mode": "cos",
                        "cos_secret_id": "persisted-secret-id",
                        "cos_secret_key": "persisted-secret-key",
                        "cos_region": "ap-guangzhou",
                        "cos_bucket": "persisted-bucket",
                    },
                }),
                encoding="utf-8",
            )
            blank_environment = {
                name: ""
                for name in module.IMAGE_STORAGE_ENV_VARS.values()
            }
            blank_environment["CHATGPT2API_AUTH_KEY"] = ""

            with mock.patch.dict(module.os.environ, blank_environment, clear=False):
                store = module.ConfigStore(config_path)
                public = store.get()
                self.assertEqual(public["image_storage"]["cos_secret_id"], "")
                self.assertEqual(public["image_storage"]["cos_secret_key"], "")
                self.assertTrue(public["image_storage"]["has_cos_secret_id"])
                self.assertTrue(public["image_storage"]["has_cos_secret_key"])

                store.update({
                    "auth-key": "",
                    "image_storage": {
                        **public["image_storage"],
                        "cos_secret_id": "",
                        "cos_secret_key": "",
                    },
                })

            persisted = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["auth-key"], "file-admin-key")
            self.assertEqual(persisted["image_storage"]["cos_secret_id"], "persisted-secret-id")
            self.assertEqual(persisted["image_storage"]["cos_secret_key"], "persisted-secret-key")

    def test_missing_config_file_can_start_with_environment_auth(self) -> None:
        module = self.config_module
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            runtime_path = Path(tmp_dir) / "data" / "runtime_config.json"
            with mock.patch.dict(
                module.os.environ,
                {"CHATGPT2API_AUTH_KEY": "environment-only-key"},
                clear=True,
            ):
                store = module.ConfigStore(config_path, runtime_config_path=runtime_path)
                self.assertNotIn("auth-key", store.get())
                store.update({"image_task_runtime": {"total_timeout_secs": 420}})
                self.assertFalse(config_path.exists())
                self.assertTrue(runtime_path.exists())
                self.assertEqual(
                    module.ConfigStore(config_path, runtime_config_path=runtime_path)
                    .get_image_task_runtime_settings()["total_timeout_secs"],
                    420,
                )

    def test_settings_api_get_never_returns_storage_credentials(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api import system as system_module

        module = self.config_module
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(
                json.dumps({"auth-key": "file-admin-key"}),
                encoding="utf-8",
            )
            environment = {
                "CHATGPT2API_AUTH_KEY": "environment-admin-key",
                "CHATGPT2API_IMAGE_STORAGE_ENABLED": "true",
                "CHATGPT2API_IMAGE_STORAGE_MODE": "cos",
                "CHATGPT2API_COS_SECRET_ID": "api-secret-id",
                "CHATGPT2API_COS_SECRET_KEY": "api-secret-key",
                "CHATGPT2API_COS_REGION": "ap-shanghai",
                "CHATGPT2API_COS_BUCKET": "api-bucket",
            }
            with mock.patch.dict(module.os.environ, environment, clear=True):
                store = module.ConfigStore(config_path)
                app = FastAPI()
                app.include_router(system_module.create_router("test"))
                with (
                    mock.patch.object(system_module, "config", store),
                    mock.patch.object(system_module, "require_admin", return_value={}),
                ):
                    response = TestClient(app).get("/api/settings")

            self.assertEqual(response.status_code, 200, response.text)
            serialized = response.text
            self.assertNotIn("environment-admin-key", serialized)
            self.assertNotIn("api-secret-id", serialized)
            self.assertNotIn("api-secret-key", serialized)
            storage = response.json()["config"]["image_storage"]
            self.assertEqual(storage["cos_secret_id"], "")
            self.assertEqual(storage["cos_secret_key"], "")
            self.assertTrue(storage["has_cos_secret_id"])
            self.assertTrue(storage["has_cos_secret_key"])

    def test_config_write_supports_single_file_bind_mount_fallback(self) -> None:
        module = self.config_module
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(
                Path,
                "replace",
                side_effect=OSError(errno.EBUSY, "bind mount target is busy"),
            ):
                module._write_json_object(path, {"safe": True})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"safe": True})


if __name__ == "__main__":
    unittest.main()

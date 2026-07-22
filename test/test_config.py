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


if __name__ == "__main__":
    unittest.main()

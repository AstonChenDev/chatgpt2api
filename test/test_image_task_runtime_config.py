from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from services.config import ConfigStore
from services.image_task_runtime_config import (
    DEFAULT_IMAGE_TASK_RUNTIME,
    IMAGE_TASK_RUNTIME_ENV_VARS,
    image_task_runtime_settings_with_env,
    normalize_image_task_runtime_settings,
)


class ImageTaskRuntimeConfigTests(unittest.TestCase):
    def test_missing_or_invalid_config_uses_safe_defaults(self) -> None:
        self.assertEqual(normalize_image_task_runtime_settings(None), DEFAULT_IMAGE_TASK_RUNTIME)
        self.assertEqual(normalize_image_task_runtime_settings("invalid"), DEFAULT_IMAGE_TASK_RUNTIME)
        self.assertEqual(
            normalize_image_task_runtime_settings(
                {
                    "total_timeout_secs": "invalid",
                    "max_concurrency": None,
                    "max_queue_size": object(),
                    "queue_timeout_secs": [],
                }
            ),
            DEFAULT_IMAGE_TASK_RUNTIME,
        )

    def test_values_are_converted_and_clamped_to_safety_limits(self) -> None:
        self.assertEqual(
            normalize_image_task_runtime_settings(
                {
                    "total_timeout_secs": "99999",
                    "max_concurrency": 0,
                    "max_queue_size": -100,
                    "queue_timeout_secs": 999,
                }
            ),
            {
                "total_timeout_secs": 1800,
                "max_concurrency": 1,
                "max_queue_size": 0,
                "queue_timeout_secs": 60,
            },
        )
        self.assertEqual(
            normalize_image_task_runtime_settings(
                {
                    "total_timeout_secs": 1,
                    "max_concurrency": 100,
                    "max_queue_size": 999,
                    "queue_timeout_secs": -1,
                }
            ),
            {
                "total_timeout_secs": 30,
                "max_concurrency": 32,
                "max_queue_size": 256,
                "queue_timeout_secs": 0,
            },
        )

    def test_non_blank_environment_values_override_config(self) -> None:
        environment = {
            "CHATGPT2API_IMAGE_TASK_TOTAL_TIMEOUT_SECS": "420",
            "CHATGPT2API_IMAGE_TASK_MAX_CONCURRENCY": "12",
            "CHATGPT2API_IMAGE_TASK_MAX_QUEUE_SIZE": "24",
            "CHATGPT2API_IMAGE_TASK_QUEUE_TIMEOUT_SECS": "9",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            actual = image_task_runtime_settings_with_env(
                {
                    "total_timeout_secs": 100,
                    "max_concurrency": 2,
                    "max_queue_size": 3,
                    "queue_timeout_secs": 4,
                }
            )

        self.assertEqual(
            actual,
            {
                "total_timeout_secs": 420,
                "max_concurrency": 12,
                "max_queue_size": 24,
                "queue_timeout_secs": 9,
            },
        )

    def test_blank_environment_values_keep_config_fallback(self) -> None:
        configured = {
            "total_timeout_secs": 360,
            "max_concurrency": 6,
            "max_queue_size": 10,
            "queue_timeout_secs": 7,
        }
        blank_environment = {name: "  " for name in IMAGE_TASK_RUNTIME_ENV_VARS.values()}
        with mock.patch.dict(os.environ, blank_environment, clear=True):
            self.assertEqual(image_task_runtime_settings_with_env(configured), configured)

    def test_config_store_persists_partial_updates_outside_tracked_config(self) -> None:
        configured = {
            "total_timeout_secs": 360,
            "max_concurrency": 6,
            "max_queue_size": 10,
            "queue_timeout_secs": 7,
        }
        blank_environment = {name: "" for name in IMAGE_TASK_RUNTIME_ENV_VARS.values()}
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(
                json.dumps({"auth-key": "test-only", "image_task_runtime": configured}),
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, blank_environment, clear=False):
                store = ConfigStore(path)
                self.assertEqual(store.get()["image_task_runtime"], configured)
                public = store.update({"image_task_runtime": {"max_concurrency": 9}})

            expected = {**configured, "max_concurrency": 9}
            self.assertEqual(public["image_task_runtime"], expected)
            persisted_config = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("image_task_runtime", persisted_config)
            self.assertEqual(
                json.loads(store.runtime_config_path.read_text(encoding="utf-8"))["image_task_runtime"],
                expected,
            )
            with mock.patch.dict(os.environ, blank_environment, clear=False):
                reloaded = ConfigStore(path)
                self.assertEqual(reloaded.get_image_task_runtime_settings(), expected)
                with mock.patch.dict(
                    os.environ,
                    {"CHATGPT2API_IMAGE_TASK_MAX_CONCURRENCY": "12"},
                    clear=False,
                ):
                    self.assertEqual(reloaded.get_image_task_runtime_settings()["max_concurrency"], 12)
            self.assertEqual(
                json.loads(store.runtime_config_path.read_text(encoding="utf-8"))["image_task_runtime"],
                expected,
            )


if __name__ == "__main__":
    unittest.main()

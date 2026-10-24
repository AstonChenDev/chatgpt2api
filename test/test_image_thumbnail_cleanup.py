from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from services.image_service import cleanup_image_thumbnails


class ImageThumbnailCleanupTests(unittest.TestCase):
    def test_cleanup_uses_one_batch_and_keeps_existing_thumbnails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keep = root / "keep.png.png"
            orphan = root / "orphan.jpg.png"
            other = root / "unexpected.txt"
            for path in (keep, orphan, other):
                path.write_bytes(b"thumbnail")
            with (mock.patch("services.image_service.config") as config,
                  mock.patch("services.image_service.image_storage_service") as storage):
                config.image_thumbnails_dir = root
                storage.existing_relatives.return_value = {"keep.png"}
                removed = cleanup_image_thumbnails()
                storage.existing_relatives.assert_called_once()
                self.assertCountEqual(storage.existing_relatives.call_args.args[0], ["keep.png", "orphan.jpg"])
                storage.exists.assert_not_called()
            self.assertEqual(removed, 2)
            self.assertTrue(keep.is_file())
            self.assertFalse(orphan.exists())
            self.assertFalse(other.exists())

    def test_stopped_worker_does_not_read_index_or_delete_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            thumbnail = root / "cached.png.png"
            thumbnail.write_bytes(b"thumbnail")
            stop = threading.Event()
            stop.set()
            with (mock.patch("services.image_service.config") as config,
                  mock.patch("services.image_service.image_storage_service") as storage):
                config.image_thumbnails_dir = root
                self.assertEqual(cleanup_image_thumbnails(stop), 0)
                storage.existing_relatives.assert_not_called()
            self.assertTrue(thumbnail.is_file())

    def test_stop_after_index_read_preserves_all_remaining_cache_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            thumbnail = root / "cached.png.png"
            thumbnail.write_bytes(b"thumbnail")
            stop = threading.Event()
            with (mock.patch("services.image_service.config") as config,
                  mock.patch("services.image_service.image_storage_service") as storage):
                config.image_thumbnails_dir = root
                storage.existing_relatives.side_effect = lambda _: stop.set() or set()
                self.assertEqual(cleanup_image_thumbnails(stop), 0)
                storage.existing_relatives.assert_called_once()
            self.assertTrue(thumbnail.is_file())


if __name__ == "__main__":
    unittest.main()

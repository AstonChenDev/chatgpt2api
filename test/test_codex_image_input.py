import base64
import unittest
from io import BytesIO
from unittest.mock import Mock

from PIL import Image

from services.openai_backend_api import OpenAIBackendAPI


class CodexImageInputTest(unittest.TestCase):
    @staticmethod
    def _image_bytes(image_format: str) -> bytes:
        output = BytesIO()
        Image.new("RGB", (8, 12), "red").save(output, format=image_format)
        return output.getvalue()

    def test_remote_image_is_downloaded_and_encoded_as_data_url(self) -> None:
        backend = object.__new__(OpenAIBackendAPI)
        image_bytes = self._image_bytes("JPEG")
        backend._decode_image_base64 = Mock(return_value=image_bytes)

        result = backend._codex_image_input(
            "edit the reference",
            ["https://example.com/reference.jpg"],
        )

        backend._decode_image_base64.assert_called_once_with("https://example.com/reference.jpg")
        image_url = result[0]["content"][1]["image_url"]
        self.assertTrue(image_url.startswith("data:image/jpeg;base64,"))
        self.assertEqual(base64.b64decode(image_url.split(",", 1)[1]), image_bytes)


if __name__ == "__main__":
    unittest.main()

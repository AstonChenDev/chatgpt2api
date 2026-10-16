import base64
import unittest
from io import BytesIO
from unittest.mock import Mock, patch

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

    def test_codex_requests_use_supported_base_model_and_preserve_image_tool(self) -> None:
        """模型退役后仍可调用图片工具，且 S/F、旧 Image-2 和参考图模式不会串用。"""
        image_bytes = self._image_bytes("PNG")
        for public_model, tool_model in (
            ("gpt-image-2.5-sunburst", "gpt-image-2.5-sunburst"),
            ("gpt-image-2.5-flare", "gpt-image-2.5-flare"),
            ("codex-gpt-image-2", "gpt-image-2"),
        ):
            for references in ([], ["https://example.com/reference.png"]):
                with self.subTest(model=public_model, has_reference=bool(references)):
                    backend = object.__new__(OpenAIBackendAPI)
                    backend.access_token = "test-codex-token"
                    backend.base_url = "https://chatgpt.com"
                    backend.session = Mock()
                    response = backend.session.post.return_value
                    response.status_code = 200
                    backend._ensure_codex_source_account = Mock()
                    backend._decode_image_base64 = Mock(return_value=image_bytes)
                    event = {"type": "response.completed"}
                    backend._iter_codex_response_events = Mock(return_value=iter([event]))

                    with (
                        patch("services.openai_backend_api.account_service.get_account", return_value={}),
                        patch("services.openai_backend_api.logger.info"),
                    ):
                        events = list(backend.iter_codex_image_response_events(
                            "create a portrait illustration",
                            images=references,
                            size="768x1024",
                            quality="low",
                            model=public_model,
                        ))

                    self.assertEqual(events, [event])
                    backend.session.post.assert_called_once()
                    self.assertEqual(
                        backend.session.post.call_args.args[0],
                        "https://chatgpt.com/backend-api/codex/responses",
                    )
                    payload = backend.session.post.call_args.kwargs["json"]
                    self.assertEqual(payload["model"], "gpt-5.6-sol")
                    self.assertEqual(payload["tools"], [{
                        "type": "image_generation",
                        "model": tool_model,
                        "action": "edit" if references else "generate",
                        "size": "768x1024",
                        "quality": "low",
                        "output_format": "png",
                    }])
                    self.assertEqual(payload["tool_choice"], {"type": "image_generation"})
                    self.assertFalse(payload["store"])
                    self.assertTrue(payload["stream"])
                    content = payload["input"][0]["content"]
                    self.assertEqual(len(content), 1 + len(references))
                    if references:
                        self.assertEqual(content[1]["type"], "input_image")
                        self.assertEqual(
                            content[1]["image_url"],
                            "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii"),
                        )
                    response.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

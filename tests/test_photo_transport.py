import base64
import io
import json
import unittest
from unittest.mock import patch

from PIL import Image

import decision
from test_decision import REPORT, model_response


def image_bytes(size, mode="RGB", image_format="JPEG", **options):
    with Image.new(mode, size) as image, io.BytesIO() as output:
        image.save(output, image_format, **options)
        return output.getvalue()


class PhotoTransportTest(unittest.TestCase):
    def test_photos_at_or_below_pixel_budget_keep_original_bytes(self):
        for size, image_format in (((320, 512), "PNG"), ((448, 448), "JPEG")):
            with self.subTest(size=size):
                original = image_bytes(size, image_format=image_format)
                self.assertEqual(decision.prepare_photo(original), original)

    def test_preserves_zoro_grid_for_landscape_portrait_and_rounding(self):
        examples = (
            ((1200, 600), (608, 288)),
            ((600, 1200), (288, 608)),
            ((451, 447), (448, 448)),
            ((768, 768), (416, 416)),
            ((7960, 40), (6272, 32)),
            ((8000, 32), (7083, 28)),
        )
        for size, expected in examples:
            with self.subTest(size=size):
                original = image_bytes(size, mode="RGBA", image_format="PNG")
                with Image.open(io.BytesIO(decision.prepare_photo(original))) as sent:
                    self.assertEqual(sent.size, expected)
                    self.assertLessEqual(sent.width * sent.height, 200_704)
                    self.assertEqual(sent.format, "JPEG")
                    self.assertEqual(sent.mode, "RGB")

    def test_oversized_jpeg_uses_zoro_orientation_and_jpeg_quality_95(self):
        exif = Image.Exif()
        exif[274] = 6
        original = image_bytes((1200, 600), exif=exif)
        jpeg_95 = image_bytes((608, 288), quality=95)
        with (
            Image.open(io.BytesIO(decision.prepare_photo(original))) as sent,
            Image.open(io.BytesIO(jpeg_95)) as reference,
        ):
            self.assertEqual(sent.size, (608, 288))
            self.assertEqual(sent.quantization, reference.quantization)

    def test_photo_and_combined_rule_reuse_one_transform(self):
        original = image_bytes((1200, 600))
        with (
            patch.object(decision, "prepare_photo", wraps=decision.prepare_photo) as prepare,
            patch.object(decision, "urlopen", side_effect=model_response(rule_1=0.8)) as send,
        ):
            result = decision.decide(REPORT, original, ["보호자가 다쳤고 주택이 침수됨"])

        self.assertEqual(result["mode"], "zoro")
        prepare.assert_called_once_with(original)
        self.assertEqual(send.call_count, 3)
        bodies = [json.loads(call.args[0].data) for call in send.call_args_list]
        image_payloads = [body["images"] for body in bodies if "images" in body]
        self.assertEqual(len(image_payloads), 2)
        self.assertEqual(image_payloads[0], image_payloads[1])
        self.assertEqual(len(image_payloads[0]), 1)
        sent_bytes = base64.b64decode(image_payloads[0][0], validate=True)
        with Image.open(io.BytesIO(sent_bytes)) as sent:
            self.assertEqual(sent.size, (608, 288))

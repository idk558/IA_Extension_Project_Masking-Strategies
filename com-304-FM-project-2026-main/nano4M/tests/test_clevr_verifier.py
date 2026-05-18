import unittest
from pathlib import Path

from PIL import Image

from nanofm.eval.clevr_verifier import CLEVRCaptionParser, compute_fidelity_score


class MockVerifier(object):
    def __init__(self, detections):
        self.detections = detections

    def detect(self, image, expected_categories):
        return self.detections


class CLEVRVerifierTest(unittest.TestCase):
    def test_parser_handles_examples_and_special_tokens(self):
        parser = CLEVRCaptionParser()
        caption = (
            "[SOS]Object 1 - Position: x=77 y=51 Shape: cube Color: blue "
            "Material: metal. Object 2 - Position: x=57 y=34 Shape: sphere "
            "Color: cyan Material: rubber. [EOS]"
        )
        objects = parser.parse(caption)

        self.assertEqual(len(objects), 2)
        self.assertEqual(objects[0]["shape"], "cube")
        self.assertEqual(objects[0]["color"], "blue")
        self.assertEqual(objects[0]["material"], "metal")
        self.assertEqual(objects[0]["position"], {"x": 77, "y": 51})
        self.assertEqual(objects[1]["shape"], "sphere")
        self.assertEqual(objects[1]["color"], "cyan")

    def test_parser_skips_malformed_entries_with_pad_tokens(self):
        parser = CLEVRCaptionParser()
        caption = (
            "[SOS]Object 1 - Position: x=10 y=20 Shape: cylinder [PAD] "
            "Color: red Material: rubber. Object 2 - Position: x=1 y=2 "
            "Shape: pyramid Color: blue Material: metal. Object 3 - "
            "Shape: sphere Color: green [PAD] Material: metal. [EOS] [PAD]"
        )
        objects = parser.parse(caption)

        self.assertEqual(len(objects), 2)
        self.assertEqual(objects[0]["shape"], "cylinder")
        self.assertEqual(objects[0]["color"], "red")
        self.assertEqual(objects[1]["shape"], "sphere")
        self.assertEqual(objects[1]["color"], "green")

    def test_parser_handles_truncated_caption(self):
        parser = CLEVRCaptionParser()
        caption = "[SOS]Object 1 - Position: x=5 y=6 Shape: cube Color: yellow"

        objects = parser.parse(caption)

        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["shape"], "cube")
        self.assertEqual(objects[0]["color"], "yellow")
        self.assertIsNone(objects[0]["material"])

    def test_score_is_one_when_expected_equals_detected(self):
        caption = (
            "Object 1 - Position: x=77 y=51 Shape: cube Color: blue Material: metal. "
            "Object 2 - Position: x=57 y=34 Shape: sphere Color: cyan Material: rubber."
        )
        detections = [
            {"shape": "cube", "color": "blue", "label": "blue cube", "confidence": 0.9, "bbox": [0, 0, 1, 1]},
            {"shape": "sphere", "color": "cyan", "label": "cyan sphere", "confidence": 0.8, "bbox": [1, 1, 2, 2]},
        ]

        result = compute_fidelity_score(caption, Image.new("RGB", (8, 8)), verifier=MockVerifier(detections))

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["category_score"], 1.0)
        self.assertEqual(result["hallucination_penalty"], 1.0)

    def test_score_decreases_monotonically_as_objects_are_missed(self):
        caption = (
            "Object 1 - Position: x=1 y=1 Shape: cube Color: blue Material: metal. "
            "Object 2 - Position: x=2 y=2 Shape: cube Color: blue Material: rubber. "
            "Object 3 - Position: x=3 y=3 Shape: sphere Color: cyan Material: metal."
        )
        image = Image.new("RGB", (8, 8))
        all_detected = [
            {"shape": "cube", "color": "blue", "label": "blue cube", "confidence": 0.9, "bbox": [0, 0, 1, 1]},
            {"shape": "cube", "color": "blue", "label": "blue cube", "confidence": 0.8, "bbox": [1, 1, 2, 2]},
            {"shape": "sphere", "color": "cyan", "label": "cyan sphere", "confidence": 0.7, "bbox": [2, 2, 3, 3]},
        ]
        one_missing = all_detected[:2]
        two_missing = all_detected[:1]

        full = compute_fidelity_score(caption, image, verifier=MockVerifier(all_detected))["score"]
        partial = compute_fidelity_score(caption, image, verifier=MockVerifier(one_missing))["score"]
        worse = compute_fidelity_score(caption, image, verifier=MockVerifier(two_missing))["score"]

        self.assertGreater(full, partial)
        self.assertGreater(partial, worse)

    def test_real_clevr_image_if_available(self):
        candidates = [
            Path("/work/com-304/datasets/clevr_com_304/val/rgb"),
            Path("/work/com-304/datasets/clevr_com_304/val/images"),
            Path(__file__).resolve().parents[1] / "data" / "clevr" / "val" / "rgb",
        ]
        image_path = None
        for directory in candidates:
            if directory.exists():
                image_path = next(directory.glob("*.png"), None) or next(directory.glob("*.jpg"), None)
                if image_path is not None:
                    break
        if image_path is None:
            self.skipTest("No local CLEVR image found")

        caption = "Object 1 - Position: x=77 y=51 Shape: cube Color: blue Material: metal."
        result = compute_fidelity_score(caption, Image.open(image_path).convert("RGB"))

        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)


if __name__ == "__main__":
    unittest.main()

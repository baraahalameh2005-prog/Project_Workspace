"""Body style option: validation only (no Anny model needed)."""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError

import api
from body_style import ATHLETIC_SHAPE, DEFAULT_BODY_STYLE, normalize_body_style

BODY = {"height": 170, "weight": 58, "chest": 86, "waist": 64, "hip": 92}


class BodyStyleTest(unittest.TestCase):
    def test_default_is_standard(self):
        self.assertEqual(DEFAULT_BODY_STYLE, "standard")
        self.assertEqual(api.AvatarRequest(**BODY).body_style, "standard")

    def test_case_insensitive(self):
        self.assertEqual(normalize_body_style(" Athletic "), "athletic")
        self.assertEqual(api.AvatarRequest(**BODY, body_style="ATHLETIC").body_style, "athletic")

    def test_unknown_style_rejected(self):
        with self.assertRaises(ValidationError):
            api.AvatarRequest(**BODY, body_style="sporty")

    def test_athletic_female_only(self):
        with self.assertRaises(ValidationError):
            api.AvatarRequest(**BODY, gender="male", body_style="athletic")
        api.AvatarRequest(**BODY, gender="male", body_style="standard")

    def test_athletic_controls_exist_in_face_model(self):
        from appearance import FACE_MODEL_LABELS

        for label in ATHLETIC_SHAPE:
            self.assertIn(label, FACE_MODEL_LABELS)

    def test_options_list_body_styles(self):
        options = api.avatar_options()["body_style"]
        self.assertEqual(options["values"], ["standard", "athletic"])
        self.assertEqual(options["default"], "standard")


if __name__ == "__main__":
    unittest.main()

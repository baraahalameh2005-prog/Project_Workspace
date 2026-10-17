"""measure_photo.py on drawn masks of known size (no rembg): each kind's
measurements come out within a centimetre or two, and one tape measurement
rescales them."""
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import measure_photo as mp  # noqa: E402

P = 10.0  # px per cm in the drawings


def drawn_mask(w_cm, h_cm, polygon_cm):
    img = Image.new("L", (int(w_cm * P), int(h_cm * P)), 0)
    ImageDraw.Draw(img).polygon([(x * P, y * P) for x, y in polygon_cm], fill=255)
    return np.asarray(img) > 0


class Scale(unittest.TestCase):
    def test_one_tape_measurement(self):
        cm = {"length": 60.0, "waist": 70.0}
        self.assertAlmostEqual(mp.rescale(cm, {"length": 72.0}), 1.2)
        with self.assertRaises(ValueError):
            mp.rescale(cm, {"hem": 100.0})


class Kinds(unittest.TestCase):
    def test_skirt(self):
        mask = drawn_mask(90, 90, [(22, 10), (58, 10), (70, 80), (10, 80)])
        r = mp.measure_mask(mask, P)
        self.assertEqual(r.kind, "skirt")
        self.assertAlmostEqual(r.cm["waist"], 72, delta=1.5)
        self.assertAlmostEqual(r.cm["length"], 70, delta=1)
        self.assertAlmostEqual(r.cm["hem"], 120, delta=1.5)

    def test_pants(self):
        mask = drawn_mask(90, 120, [(20, 8), (58, 8), (62, 36), (72, 106), (52, 108), (39, 38), (26, 108), (6, 106), (16, 36)])
        r = mp.measure_mask(mask, P)
        self.assertEqual(r.kind, "pants")
        self.assertAlmostEqual(r.cm["waist"], 76, delta=1.5)
        self.assertAlmostEqual(r.cm["inseam"], np.hypot(70, 13), delta=2)
        self.assertAlmostEqual(r.cm["opening"], 40, delta=2)

    def test_scarf(self):
        mask = drawn_mask(200, 80, [(8, 8), (188, 8), (188, 68), (8, 68)])
        r = mp.measure_mask(mask, P)
        self.assertEqual(r.kind, "scarf")
        self.assertAlmostEqual(r.cm["length"], 180, delta=1)
        self.assertAlmostEqual(r.cm["width"], 60, delta=1)

    def test_accessory(self):
        mask = drawn_mask(60, 50, [(10, 10), (42, 10), (42, 34), (10, 34)])
        r = mp.measure_mask(mask, P, "accessory")
        self.assertAlmostEqual(r.cm["width"], 32, delta=0.5)
        self.assertAlmostEqual(r.cm["height"], 24, delta=0.5)


if __name__ == "__main__":
    unittest.main()

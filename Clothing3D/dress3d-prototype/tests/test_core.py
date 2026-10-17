"""Core regression tests (stdlib unittest; no extra installs).

Run from the repository root:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v

These tests check *construction consistency* (the mesh reproduces the numbers it was given)
and pipeline behaviour on the bundled SYNTHETIC test images. They do not prove accuracy on real
photos or real garments.
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dress3d.schema import parse_spec  # noqa: E402
from dress3d.generator import DressGenerator  # noqa: E402
from dress3d import export as ex  # noqa: E402
from dress3d.measure import measure_dress, compare  # noqa: E402


def boundary_loop_count(V, F):
    m = trimesh.Trimesh(V, F, process=False)
    be = m.edges[trimesh.grouping.group_rows(m.edges_sorted, require_count=1)]
    return len(trimesh.graph.connected_components(be)), m


class SchemaTests(unittest.TestCase):
    def test_inches_and_flat(self):
        s = parse_spec({"units": "in", "measurements": {"dress_length": 38, "bust": 36,
                                                        "waist": {"value": 15, "method": "flat"}}})
        self.assertAlmostEqual(s.m["dress_length"], 38 * 2.54)
        self.assertAlmostEqual(s.m["waist"], 15 * 2 * 2.54)
        self.assertIn("hip", s.m)                       # filled by a documented default
        self.assertNotIn("hip", s.provided)

    def test_required_and_invalid(self):
        with self.assertRaises(ValueError):
            parse_spec({"measurements": {"bust": 90}})  # dress_length missing
        with self.assertRaises(ValueError):
            parse_spec({"measurements": {"dress_length": 90, "bust": -1}})
        with self.assertRaises(ValueError):
            parse_spec({"measurements": {"dress_length": 90, "bust": 90, "unknown_field": 3}})


class GeneratorRoundTripTests(unittest.TestCase):
    def _roundtrip(self, sleeve):
        spec = parse_spec({"style": {"silhouette": "a-line", "sleeve": sleeve, "neckline": "v"},
                           "measurements": {"dress_length": 95, "bust": 92, "waist": 78, "hip": 100, "hem": 150},
                           "options": {"fold_count": 8, "fold_amplitude": 1.5}})
        dm = DressGenerator(spec).build()
        loops, m = boundary_loop_count(dm.vertices, dm.faces)
        self.assertEqual(loops, 4, "expected neck, hem and two armhole/cuff openings")
        self.assertTrue(m.is_winding_consistent)
        self.assertEqual(len(trimesh.grouping.group_rows(m.edges_sorted, require_count=3)), 0)
        with tempfile.TemporaryDirectory() as d:
            glb = os.path.join(d, "t.glb")
            ex.export_glb(ex.to_trimesh(dm, ex.make_print_texture(dm.atlas_size_cm)), glb)
            ex.save_landmarks(dm, os.path.join(d, "t.landmarks.json"))
            meas, _ = measure_dress(glb, os.path.join(d, "t.landmarks.json"), hip_position=spec.m["hip_position"])
        for row in compare(meas, spec.m):
            if row["measurement"] in ("waist_position", "hip_position"):
                continue
            self.assertLess(abs(row["pct_err"]), 0.5, row)

    def test_sleeveless(self):
        self._roundtrip("none")

    def test_long_sleeve(self):
        self._roundtrip("long")


class HybridSyntheticTests(unittest.TestCase):
    def test_t2_style_and_dimensions(self):
        from dress3d.hybrid import reconstruct
        photo = os.path.join(ROOT, "experiments", "inputs", "synthetic", "T2_aline_vneck", "photo_front.png")
        if not os.path.exists(photo):
            self.skipTest("synthetic inputs not present")
        user = {"dress_length": 100, "bust": 90, "waist": 76, "hip": 98, "hem": 170, "shoulder_width": 36,
                "hip_position": 58}
        dm, tex, uv, rep, pa = reconstruct(photo, user)
        self.assertEqual(rep["style"], {"silhouette": "a-line", "neckline": "v", "sleeve": "none"})
        self.assertTrue(np.all((uv >= 0) & (uv <= 1)))
        with tempfile.TemporaryDirectory() as d:
            glb = os.path.join(d, "h.glb")
            ex.export_glb(ex.to_trimesh(dm, tex, uv=uv), glb)
            ex.save_landmarks(dm, os.path.join(d, "h.landmarks.json"))
            meas, _ = measure_dress(glb, os.path.join(d, "h.landmarks.json"), hip_position=58)
        for k in ("dress_length", "bust", "waist", "hip", "hem", "shoulder_width"):
            self.assertLess(abs(meas[k] - user[k]) / user[k], 0.005, k)


if __name__ == "__main__":
    unittest.main()

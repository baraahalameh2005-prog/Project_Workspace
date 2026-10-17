"""Stretch, flat measurements, overrides, validation of garment data, CLI and HTTP API."""

import json
import os
import subprocess
import sys
import threading
import unittest
import urllib.error
import urllib.request

from fit_engine import recommend_size
from fit_engine.rules import target_ease
from fit_engine.server import make_server

from .helpers import body, codes, comp, dress, flat_version, pants, tshirt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIUM = dict(chest=92, waist=80, hip=96, shoulder_width=44)


class TestStretch(unittest.TestCase):
    def test_stretch_allows_negative_ease(self):
        # hip of size M is 2 cm smaller than the body
        b = body(chest=88, waist=76, hip=104, shoulder_width=39)
        woven = recommend_size(b, dress(stretch="none"))
        stretchy = recommend_size(b, dress(stretch="high"))
        self.assertNotEqual(woven.get("recommended_size"), "M")
        self.assertEqual(stretchy["recommended_size"], "M")
        self.assertIn("stretch_note", codes(stretchy["reasons"]))

    def test_stretch_lowers_min_ease(self):
        lo_none, _, _ = target_ease("dress", "hip", "regular", "none", 100)
        lo_high, _, _ = target_ease("dress", "hip", "regular", "high", 100)
        self.assertAlmostEqual(lo_none - lo_high, 12.0)

    def test_pants_waist_not_shifted_by_cut(self):
        self.assertEqual(target_ease("pants", "waist", "loose", "none", 80)[:2],
                         target_ease("pants", "waist", "slim", "none", 80)[:2])


class TestGarmentData(unittest.TestCase):
    def test_flat_measurements_equal_circumference(self):
        b = body(chest=92, waist=80, hip=96, thigh=56, inseam=80)
        a = recommend_size(b, pants())
        f = recommend_size(b, flat_version(pants()))
        self.assertEqual(a["recommended_size"], f["recommended_size"])
        self.assertEqual(comp(a, "hip")["garment_cm"], comp(f, "hip")["garment_cm"])

    def test_half_measurement_without_flat_flag_gets_hint(self):
        g = flat_version(tshirt())
        g["measurement_type"] = "circumference"   # wrong flag on purpose
        r = recommend_size(body(**MEDIUM), g)
        self.assertEqual(r["status"], "no_suitable_size")   # everything looks far too small
        self.assertIn("possible_flat_measurement", codes(r["warnings"]))

    def test_ease_override_is_used(self):
        g = tshirt(ease_overrides={"chest": [0, 4]})
        r = recommend_size(body(**MEDIUM), g)
        self.assertEqual(comp(r, "chest")["target_ease_cm"], [0.0, 4.0])
        self.assertEqual(comp(r, "chest")["ease_source"], "override")
        self.assertIn("ease_override_used", codes(r["reasons"]))

    def test_invalid_ease_override(self):
        r = recommend_size(body(**MEDIUM), tshirt(ease_overrides={"chest": [10, 2]}))
        self.assertIn("invalid_ease_override", codes(r["errors"]))

    def test_duplicate_and_empty_sizes(self):
        g = tshirt()
        g["sizes"].append(dict(g["sizes"][0]))
        self.assertIn("duplicate_size", codes(recommend_size(body(**MEDIUM), g)["errors"]))
        g2 = tshirt()
        g2["sizes"] = []
        self.assertIn("no_sizes", codes(recommend_size(body(**MEDIUM), g2)["errors"]))

    def test_unknown_fit_type_defaults_to_regular(self):
        r = recommend_size(body(**MEDIUM), tshirt(fit_type="baggy"))
        self.assertEqual(r["fit_type"], "regular")
        self.assertIn("unknown_fit_type", codes(r["warnings"]))

    def test_size_order_preserved_in_all_sizes(self):
        r = recommend_size(body(**MEDIUM), tshirt())
        self.assertEqual([s["size"] for s in r["all_sizes"]], ["S", "M", "L", "XL"])

    def test_deterministic(self):
        a = recommend_size(body(**MEDIUM), tshirt())
        b = recommend_size(body(**MEDIUM), tshirt())
        self.assertEqual(a, b)


class TestCLI(unittest.TestCase):
    def _run(self, *args):
        return subprocess.run([sys.executable, "-m", "fit_engine", *args], cwd=ROOT,
                              capture_output=True, text=True, encoding="utf-8")

    def test_cli_json(self):
        p = self._run("recommend", "--body", "examples/body_medium.json",
                      "--garment", "examples/garment_tshirt_regular.json")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout)["recommended_size"], "M")

    def test_cli_invalid_exit_code(self):
        p = self._run("recommend", "--body", "examples/body_invalid.json",
                      "--garment", "examples/garment_tshirt_regular.json", "--summary")
        self.assertEqual(p.returncode, 1)
        self.assertIn("غير صالحة", p.stdout)


class TestHTTPAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = make_server("127.0.0.1", 0)
        cls.port = cls.httpd.server_address[1]
        cls.t = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.t.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _post(self, payload):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/recommend",
                                     data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_health(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5) as r:
            self.assertEqual(json.loads(r.read())["status"], "ok")

    def test_recommend_ok(self):
        code, res = self._post({"body": body(**MEDIUM), "garment": tshirt()})
        self.assertEqual(code, 200)
        self.assertEqual(res["recommended_size"], "M")

    def test_recommend_invalid(self):
        code, res = self._post({"body": body(chest="x"), "garment": tshirt()})
        self.assertEqual(code, 422)
        self.assertEqual(res["status"], "invalid_input")

    def test_bad_request_shape(self):
        code, _ = self._post({"only": "body"})
        self.assertEqual(code, 400)


if __name__ == "__main__":
    unittest.main()

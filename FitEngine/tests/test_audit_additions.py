"""Tests added during the technical audit (v0.2.0).

Expected values are worked out by hand from rules.py (comments show the arithmetic),
so these tests check that results are RIGHT, not only that the code runs.
All data is synthetic; no avatar or 3D garment is needed.
"""

import http.client
import json
import os
import re
import unittest

from fit_engine import recommend_size
from fit_engine.server import UI_PATH, make_server

from .helpers import body, codes, comp, dress, pants, tshirt

MEDIUM = dict(chest=92, waist=80, hip=96, shoulder_width=44)


def ts(sizes, **extra):
    """Minimal t-shirt garment with explicit sizes."""
    g = {"category": "t_shirt", "fit_type": "regular", "stretch": "none", "unit": "cm",
         "sizes": [{"label": l, "measurements": m} for l, m in sizes]}
    g.update(extra)
    return g


class TestBoundaries(unittest.TestCase):
    """t-shirt chest target with override [6, 16]; circumference tolerance = 3 cm."""

    def _status(self, garment_chest):
        g = ts([("M", {"chest": garment_chest, "shoulder_width": 44})],
               ease_overrides={"chest": [6, 16]})
        return comp(recommend_size(body(chest=92, shoulder_width=44), g), "chest")["status"]

    def test_lower_edge_is_good(self):
        self.assertEqual(self._status(98), "good")            # ease +6 == min

    def test_upper_edge_is_good(self):
        self.assertEqual(self._status(108), "good")           # ease +16 == max

    def test_exactly_tolerance_is_slightly(self):
        self.assertEqual(self._status(95), "slightly_tight")  # ease +3, 3 below min
        self.assertEqual(self._status(111), "slightly_loose")  # ease +19, 3 above max

    def test_beyond_tolerance_is_too(self):
        self.assertEqual(self._status(94.9), "too_tight")
        self.assertEqual(self._status(111.1), "too_loose")


class TestBetweenTwoSizes(unittest.TestCase):
    """Body chest 96, shoulder 45 on the standard t-shirt (target chest +6..+16, shoulder -1..+3).

    M: chest +4 (slightly tight), shoulder -1 (good)  -> wearable but tight
    L: chest +12 (good),          shoulder +1 (good)  -> best
    """

    def setUp(self):
        self.r = recommend_size(body(chest=96, waist=82, hip=98, shoulder_width=45), tshirt())

    def test_both_neighbours_evaluated(self):
        st = {s["size"]: s["statuses"]["chest"] for s in self.r["all_sizes"]}
        self.assertEqual(st["M"], "slightly_tight")
        self.assertEqual(st["L"], "good")

    def test_picks_fully_fitting_size_and_offers_other(self):
        self.assertEqual(self.r["recommended_size"], "L")
        self.assertEqual(comp(self.r, "chest")["ease_cm"], 12.0)
        self.assertEqual(self.r["alternative_size"]["size"], "M")
        self.assertEqual(self.r["alternative_size"]["direction"], "smaller")


class TestAreasJudgedSeparately(unittest.TestCase):
    """Dress target (regular, non-stretch): chest +4..+12, waist +3..+12, hip +4..+14."""

    def test_each_area_has_its_own_status(self):
        # size M: chest 96-86=+10 good, waist 80-72=+8 good, hip 102-100=+2 slightly tight
        r = recommend_size(body(chest=86, waist=72, hip=100, shoulder_width=39), dress())
        self.assertEqual(comp(r, "chest")["status"], "good")
        self.assertEqual(comp(r, "waist")["status"], "good")
        self.assertEqual(comp(r, "hip")["status"], "slightly_tight")
        self.assertEqual(r["overall_fit"], "tight")

    def test_tight_and_loose_areas_give_mixed(self):
        # waist 87-72=+15 (3 over max 12 -> slightly_loose), hip 102-100=+2 (2 under min 4 -> slightly_tight)
        g = {"category": "dress", "fit_type": "regular", "stretch": "none", "unit": "cm",
             "sizes": [{"label": "M", "measurements": {"chest": 96, "waist": 87, "hip": 102}}]}
        r = recommend_size(body(chest=86, waist=72, hip=100), g)
        self.assertEqual(comp(r, "waist")["status"], "slightly_loose")
        self.assertEqual(comp(r, "hip")["status"], "slightly_tight")
        self.assertEqual(r["overall_fit"], "mixed")


class TestIncompleteGarmentData(unittest.TestCase):
    """A size missing a key garment measurement must not win by being compared on fewer keys."""

    def test_complete_size_beats_incomplete_one(self):
        # before v0.2.0 L won (score 0.0 on shoulder only) although its chest is unknown
        g = ts([("M", {"chest": 100, "shoulder_width": 44}),
                ("L", {"shoulder_width": 46}),
                ("XL", {"chest": 150, "shoulder_width": 47})])
        r = recommend_size(body(chest=92, shoulder_width=45), g)
        self.assertEqual(r["recommended_size"], "M")
        L = next(s for s in r["all_sizes"] if s["size"] == "L")
        self.assertFalse(L["acceptable"])
        self.assertEqual(L["incomplete_primary"], ["chest"])
        self.assertIn("incomplete_size", codes(r["warnings"]))

    def test_only_incomplete_size_might_fit_gives_no_recommendation(self):
        g = ts([("S", {"chest": 80, "shoulder_width": 40}),
                ("M", {"shoulder_width": 45}),
                ("L", {"chest": 150, "shoulder_width": 47})])
        r = recommend_size(body(chest=92, shoulder_width=45), g)
        self.assertEqual(r["status"], "insufficient_data")
        self.assertEqual(r["situation"], "incomplete_garment_data")
        self.assertIsNone(r["recommended_size"])
        self.assertEqual(r["confidence"], "none")
        self.assertIn("(M)", r["reasons"][0]["message_en"])

    def test_key_missing_in_every_size_lowers_confidence_only(self):
        # no size has chest -> sizes are still comparable on shoulders; recommend with low confidence
        g = ts([("M", {"shoulder_width": 44}), ("L", {"shoulder_width": 46})])
        r = recommend_size(body(chest=92, shoulder_width=46), g)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["recommended_size"], "L")
        self.assertEqual(r["confidence"], "low")
        self.assertEqual(comp(r, "chest")["status"], "missing_garment")
        self.assertIn("missing_primary_partial", codes(r["warnings"]))


class TestNoInventedValues(unittest.TestCase):
    def test_missing_body_values_stay_missing(self):
        r = recommend_size(body(chest=92), tshirt())
        self.assertIsNone(comp(r, "shoulder_width")["body_cm"])
        self.assertIsNone(comp(r, "shoulder_width")["ease_cm"])
        self.assertEqual(r["confidence"], "low")

    def test_height_alone_cannot_size_pants(self):
        r = recommend_size(body(height=172, inseam=80), pants())
        self.assertEqual(r["status"], "insufficient_data")
        self.assertIsNone(r["recommended_size"])
        self.assertIn("insufficient_primary", codes(r["reasons"]))


class TestAssumptionsLowerConfidence(unittest.TestCase):
    def test_missing_fit_type(self):
        g = tshirt()
        del g["fit_type"]
        r = recommend_size(body(**MEDIUM), g)
        self.assertEqual(r["recommended_size"], "M")
        self.assertEqual(r["confidence"], "medium")
        self.assertIn("fit_type_default", codes(r["warnings"]))

    def test_unknown_stretch(self):
        r = recommend_size(body(**MEDIUM), tshirt(stretch="super"))
        self.assertEqual(r["stretch"], "none")
        self.assertEqual(r["confidence"], "medium")

    def test_all_given_is_high(self):
        self.assertEqual(recommend_size(body(**MEDIUM), tshirt())["confidence"], "high")


class TestUnitsAndDefinitions(unittest.TestCase):
    def test_millimetres_sent_as_cm_hint(self):
        r = recommend_size(body(chest=920, shoulder_width=44), tshirt())
        self.assertEqual(r["status"], "invalid_input")
        self.assertIn("بالملم", r["errors"][0]["message_ar"])

    def test_garment_in_inches_body_in_cm(self):
        g = tshirt()
        g["unit"] = "in"
        for s in g["sizes"]:
            s["measurements"] = {k: v / 2.54 for k, v in s["measurements"].items()}
        r = recommend_size(body(**MEDIUM), g)
        self.assertEqual(r["recommended_size"], "M")
        self.assertAlmostEqual(comp(r, "chest")["garment_cm"], 100.0, places=1)

    def test_flat_doubles_circumferences_not_shoulder(self):
        g = ts([("M", {"chest": 50, "shoulder_width": 44})], measurement_type="flat")
        r = recommend_size(body(chest=92, shoulder_width=44), g)
        self.assertEqual(comp(r, "chest")["garment_cm"], 100.0)        # 50 flat -> 100 around
        self.assertEqual(comp(r, "shoulder_width")["garment_cm"], 44.0)  # width stays a width

    def test_null_measurement_type_means_circumference(self):
        g = ts([("M", {"chest": 100, "shoulder_width": 44})], measurement_type=None)
        r = recommend_size(body(chest=92, shoulder_width=44), g)
        self.assertEqual(comp(r, "chest")["garment_cm"], 100.0)

    def test_contradictory_body_is_flagged(self):
        r = recommend_size(body(chest=70, waist=120, hip=60, shoulder_width=40), tshirt())
        self.assertIn("unusual_proportions", codes(r["warnings"]))


class TestRobustness(unittest.TestCase):
    """recommend_size must never raise; bad input -> status invalid_input."""

    GOOD_BODY = body(**MEDIUM)

    def test_malformed_inputs_never_raise(self):
        bad_garments = [
            None, [], "x", 5,
            tshirt(ease_overrides=[1, 2]),
            tshirt(ease_overrides="x"),
            tshirt(ease_overrides={"chest": [float("nan"), 3]}),
            tshirt(ease_overrides={"chest": [0, float("inf")]}),
            tshirt(sizes={"M": 1}),
            tshirt(sizes=[None]),
            tshirt(sizes=[{"label": None, "measurements": {"chest": 100}}]),
            tshirt(sizes=[{"label": True, "measurements": {"chest": 100}}]),
            tshirt(sizes=[{"label": "M", "measurements": None}]),
            tshirt(sizes=[{"label": "M", "measurements": [1, 2]}]),
            tshirt(sizes=[{"label": "M", "unit": ["cm"], "measurements": {"chest": 100}}]),
            tshirt(category=["t_shirt"]),
            tshirt(measurement_type="half"),
        ]
        for g in bad_garments:
            with self.subTest(garment=repr(g)[:60]):
                r = recommend_size(self.GOOD_BODY, g)
                self.assertEqual(r["status"], "invalid_input")
                self.assertTrue(r["errors"])
                json.dumps(r, ensure_ascii=False)

        bad_bodies = [None, 5, {"measurements": None}, {"measurements": [92]},
                      {"unit": ["cm"], "measurements": {"chest": 92}},
                      {"measurements": {"chest": {"v": 1}}}, {"chest": 1e308},
                      {"chest": float("nan")}, {"chest": "92cm"}, {}]
        for b in bad_bodies:
            with self.subTest(body=repr(b)[:60]):
                r = recommend_size(b, tshirt())
                self.assertEqual(r["status"], "invalid_input")

    def test_unknown_override_key_is_warning(self):
        r = recommend_size(self.GOOD_BODY, tshirt(ease_overrides={"neck": [1, 2]}))
        self.assertEqual(r["status"], "ok")
        self.assertIn("unknown_ease_override", codes(r["warnings"]))

    def test_numeric_size_labels(self):
        g = pants()
        for s in g["sizes"]:
            s["label"] = int(s["label"])
        r = recommend_size(body(waist=80, hip=96, thigh=56, inseam=80), g)
        self.assertEqual(r["recommended_size"], "32")


class TestWebUI(unittest.TestCase):
    """The UI must be served by the API and must get its results from POST /recommend."""

    @classmethod
    def setUpClass(cls):
        import threading
        cls.httpd = make_server("127.0.0.1", 0)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        with open(UI_PATH, encoding="utf-8") as f:
            cls.html = f.read()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _get(self, path):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        c.request("GET", path)
        r = c.getresponse()
        return r.status, r.getheader("Content-Type"), r.read()

    def test_ui_served(self):
        status, ctype, data = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        self.assertIn("/recommend", data.decode("utf-8"))

    def test_ui_calls_real_engine_and_schema(self):
        self.assertRegex(self.html, r'fetch\("/recommend"')
        self.assertRegex(self.html, r'fetch\("/schema"')

    def test_ui_has_no_hardcoded_results(self):
        # the page must not contain a canned result object
        for field in ('"recommended_size"', "recommended_size:", '"overall_fit"', "all_sizes:"):
            self.assertNotIn(field, self.html)

    def test_query_string_and_bad_content_length(self):
        status, _, _ = self._get("/health?x=1")
        self.assertEqual(status, 200)
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        c.putrequest("POST", "/recommend")
        c.putheader("Content-Length", "abc")
        c.endheaders()
        self.assertEqual(c.getresponse().status, 400)

    def test_ui_file_exists_in_package(self):
        self.assertTrue(os.path.isfile(UI_PATH))
        self.assertTrue(re.search(r"<html[^>]*dir=\"rtl\"", self.html))


if __name__ == "__main__":
    unittest.main()

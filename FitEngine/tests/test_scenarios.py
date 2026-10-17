"""The required scenarios from the project brief (section 7)."""

import json
import unittest

from fit_engine import recommend_size

from .helpers import body, codes, comp, dress, pants, tshirt

MEDIUM = dict(height=172, chest=92, waist=80, hip=96, shoulder_width=44, arm_length=60)


class TestOneSizeFits(unittest.TestCase):
    """1. A user for whom one of the available sizes fits."""

    def setUp(self):
        self.r = recommend_size(body(**MEDIUM), tshirt())

    def test_recommends_m(self):
        self.assertEqual(self.r["status"], "ok")
        self.assertEqual(self.r["recommended_size"], "M")
        self.assertEqual(self.r["overall_fit"], "good")

    def test_every_primary_measurement_compared(self):
        self.assertEqual(comp(self.r, "chest")["status"], "good")
        self.assertEqual(comp(self.r, "chest")["ease_cm"], 8.0)
        self.assertEqual(comp(self.r, "shoulder_width")["status"], "good")

    def test_has_reason_and_disclaimer(self):
        self.assertIn("reason_all_good", codes(self.r["reasons"]))
        self.assertEqual(self.r["disclaimer"]["code"], "disclaimer")

    def test_confidence_medium_when_stretch_unknown(self):
        g = tshirt()
        del g["stretch"]
        r = recommend_size(body(**MEDIUM), g)
        self.assertEqual(r["recommended_size"], "M")
        self.assertEqual(r["confidence"], "medium")
        self.assertIn("stretch_default", codes(r["warnings"]))

    def test_result_is_json_serialisable(self):
        json.dumps(self.r, ensure_ascii=False)


class TestAllSizesTooSmall(unittest.TestCase):
    """2. Every available size is tighter than needed."""

    def test_all_too_small(self):
        r = recommend_size(body(chest=130, waist=120, hip=128, shoulder_width=52), tshirt())
        self.assertEqual(r["status"], "no_suitable_size")
        self.assertEqual(r["situation"], "all_sizes_too_small")
        self.assertIsNone(r["recommended_size"])
        self.assertEqual(r["closest_size"], "XL")
        self.assertEqual(r["overall_fit"], "tight")
        self.assertEqual(comp(r, "chest")["status"], "too_tight")
        self.assertIn("all_too_small", codes(r["reasons"]))


class TestAllSizesTooLarge(unittest.TestCase):
    """3. Every available size is looser than needed."""

    def test_all_too_large(self):
        r = recommend_size(body(chest=70, waist=60, hip=72, shoulder_width=34), tshirt())
        self.assertEqual(r["status"], "no_suitable_size")
        self.assertEqual(r["situation"], "all_sizes_too_large")
        self.assertEqual(r["closest_size"], "S")
        self.assertEqual(r["overall_fit"], "loose")
        self.assertIn("all_too_large", codes(r["reasons"]))


class TestChestFitsHipDoesNot(unittest.TestCase):
    """4. Chest fits but hip does not — must be reported per area, not hidden."""

    def test_no_single_size_for_pear_shape(self):
        r = recommend_size(body(chest=86, waist=72, hip=108, shoulder_width=38), dress())
        self.assertEqual(r["status"], "no_suitable_size")
        self.assertEqual(r["situation"], "no_single_size_fits")
        msg = r["reasons"][0]
        self.assertEqual(msg["code"], "no_single_size")
        self.assertIn("محيط الصدر", msg["message_ar"])
        self.assertIn("محيط الورك", msg["message_ar"])
        self.assertIn("hip fits: XL", msg["message_en"])

    def test_wearable_size_still_flags_hip(self):
        # hip only slightly tight on M -> wearable, but the hip issue is reported
        r = recommend_size(body(chest=86, waist=72, hip=100, shoulder_width=39), dress())
        self.assertEqual(r["recommended_size"], "M")
        self.assertEqual(comp(r, "chest")["status"], "good")
        self.assertEqual(comp(r, "hip")["status"], "slightly_tight")
        self.assertEqual(r["overall_fit"], "tight")
        issue_keys = [i["measurement"] for i in r["issues"]]
        self.assertEqual(issue_keys, ["hip"])


class TestFitTypeChangesSize(unittest.TestCase):
    """5. Two candidate sizes whose suitability depends on the cut."""

    B = dict(chest=96, waist=82, hip=98, shoulder_width=45)

    def test_regular_prefers_l(self):
        r = recommend_size(body(**self.B), tshirt(fit_type="regular"))
        self.assertEqual(r["recommended_size"], "L")
        self.assertEqual(r["alternative_size"]["size"], "M")
        self.assertEqual(r["alternative_size"]["direction"], "smaller")

    def test_slightly_tight_on_m_regular_moves_to_l(self):
        # same body, normal size grading: L fits fully, so it wins over a slightly tight M
        r = recommend_size(body(chest=95, waist=80, hip=94, shoulder_width=44), tshirt())
        self.assertEqual(r["recommended_size"], "L")
        self.assertEqual(r["alternative_size"]["size"], "M")

    def test_slim_prefers_m(self):
        r = recommend_size(body(**self.B), tshirt(fit_type="slim"))
        self.assertEqual(r["recommended_size"], "M")

    def test_oversized_prefers_xl(self):
        r = recommend_size(body(**self.B), tshirt(fit_type="oversized"))
        self.assertEqual(r["recommended_size"], "XL")

    def test_slightly_tight_m_beats_l_when_l_is_looser(self):
        # chest ease on M = +5 (target starts at +6) -> slightly tight,
        # but L = +18 is further outside on the loose side, so M stays the pick
        # and L is offered for comfort.
        g = {"category": "t_shirt", "fit_type": "regular", "stretch": "none", "unit": "cm",
             "sizes": [{"label": "M", "measurements": {"chest": 100, "shoulder_width": 44}},
                       {"label": "L", "measurements": {"chest": 113, "shoulder_width": 44}}]}
        r = recommend_size(body(chest=95, shoulder_width=44), g)
        self.assertEqual(r["recommended_size"], "M")
        self.assertEqual(comp(r, "chest")["status"], "slightly_tight")
        self.assertEqual(r["alternative_size"]["size"], "L")
        self.assertEqual(r["alternative_size"]["code"], "alt_fixes_tight")
        self.assertIn("between_sizes", codes(r["reasons"]))


class TestMissingMeasurements(unittest.TestCase):
    """6. Some measurements are missing."""

    def test_missing_primary_lowers_confidence(self):
        r = recommend_size(body(chest=92, height=172), tshirt())
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["recommended_size"], "M")
        self.assertEqual(r["confidence"], "low")
        self.assertEqual(comp(r, "shoulder_width")["status"], "missing_body")
        self.assertIn("missing_primary_partial", codes(r["warnings"]))
        self.assertIn("shoulder_width", r["missing_measurements"]["body"])

    def test_no_primary_means_insufficient_data(self):
        r = recommend_size(body(height=172, arm_length=60), tshirt())
        self.assertEqual(r["status"], "insufficient_data")
        self.assertEqual(r["overall_fit"], "insufficient_data")
        self.assertIsNone(r["recommended_size"])
        self.assertEqual(r["confidence"], "none")

    def test_garment_missing_measurement_for_one_size(self):
        g = tshirt()
        del g["sizes"][1]["measurements"]["shoulder_width"]   # M has no shoulder
        r = recommend_size(body(**MEDIUM), g)
        m = next(s for s in r["all_sizes"] if s["size"] == "M")
        self.assertEqual(m["statuses"]["shoulder_width"], "missing_garment")
        self.assertIn("shoulder_width", r["missing_measurements"]["garment"])

    def test_weight_alone_is_rejected(self):
        r = recommend_size({"unit": "kg", "measurements": {"weight": 70}}, tshirt())
        self.assertEqual(r["status"], "invalid_input")

    def test_weight_only_body_in_cm(self):
        r = recommend_size(body(weight=70), tshirt())
        self.assertEqual(r["status"], "invalid_input")
        self.assertIn("no_body_measurements", codes(r["errors"]))
        self.assertIn("weight_ignored", codes(r["warnings"]))


class TestInvalidInputAndUnits(unittest.TestCase):
    """7. Invalid values and different units."""

    def test_inches_equal_cm(self):
        cm = recommend_size(body(**MEDIUM), tshirt())
        inch = recommend_size(body(unit="in", **{k: v / 2.54 for k, v in MEDIUM.items()}), tshirt())
        self.assertEqual(cm["recommended_size"], inch["recommended_size"])
        self.assertAlmostEqual(comp(inch, "chest")["body_cm"], 92.0, places=1)

    def test_mm_and_per_value_unit(self):
        b = {"unit": "mm", "measurements": {"chest": 920, "waist": 800, "hip": 960,
                                             "shoulder_width": {"value": 44, "unit": "cm"}}}
        r = recommend_size(b, tshirt())
        self.assertEqual(r["recommended_size"], "M")
        self.assertEqual(comp(r, "shoulder_width")["body_cm"], 44.0)

    def test_non_numeric_negative_and_out_of_range(self):
        r = recommend_size(body(chest=38, waist="abc", hip=-5), tshirt())
        self.assertEqual(r["status"], "invalid_input")
        c = codes(r["errors"])
        self.assertIn("out_of_range", c)
        self.assertIn("invalid_number", c)
        self.assertIn("non_positive", c)
        chest_err = next(e for e in r["errors"] if e["field"] == "body.measurements.chest")
        self.assertIn("بالإنش", chest_err["message_ar"])

    def test_unknown_unit(self):
        r = recommend_size(body(unit="ft", chest=3), tshirt())
        self.assertEqual(r["status"], "invalid_input")
        self.assertIn("unknown_unit", codes(r["errors"]))

    def test_boolean_is_not_a_number(self):
        r = recommend_size(body(chest=True), tshirt())
        self.assertIn("invalid_number", codes(r["errors"]))

    def test_aliases_mapped(self):
        r = recommend_size({"unit": "cm", "measurements": {"bust": 92, "hips": 96, "shoulders": 44}},
                           tshirt())
        self.assertEqual(r["recommended_size"], "M")
        self.assertIn("alias_used", codes(r["warnings"]))

    def test_unknown_measurement_is_warning(self):
        r = recommend_size(body(chest=92, shoulder_width=44, neck=38), tshirt())
        self.assertEqual(r["status"], "ok")
        self.assertIn("unknown_measurement", codes(r["warnings"]))

    def test_body_not_a_dict(self):
        r = recommend_size([92, 80], tshirt())
        self.assertEqual(r["status"], "invalid_input")


class TestGarmentTypes(unittest.TestCase):
    """8. Different garment types use different key measurements."""

    def test_pants_use_waist_and_hip_not_chest(self):
        r = recommend_size(body(chest=92, waist=80, hip=96, thigh=56, inseam=80), pants())
        self.assertEqual(r["recommended_size"], "32")
        keys = [c["measurement"] for c in r["comparisons"]]
        self.assertNotIn("chest", keys)
        self.assertEqual([c["measurement"] for c in r["comparisons"] if c["importance"] == "primary"],
                         ["waist", "hip"])

    def test_pants_without_waist_or_hip_is_insufficient(self):
        r = recommend_size(body(chest=92, shoulder_width=44, height=172), pants())
        self.assertEqual(r["status"], "insufficient_data")

    def test_tshirt_does_not_compare_sleeve_or_length(self):
        r = recommend_size(body(**MEDIUM), tshirt())
        keys = [c["measurement"] for c in r["comparisons"]]
        self.assertNotIn("sleeve_length", keys)
        info = {i["measurement"]: i["code"] for i in r["not_compared"]}
        self.assertEqual(info["sleeve_length"], "not_relevant")
        self.assertEqual(info["length"], "not_compared")

    def test_shirt_compares_sleeve_against_arm(self):
        g = {"category": "shirt", "fit_type": "regular", "stretch": "none", "unit": "cm",
             "sizes": [{"label": "M", "measurements": {"chest": 102, "shoulder_width": 45,
                                                         "sleeve_length": 56}}]}
        r = recommend_size(body(chest=92, shoulder_width=44, arm_length=62), g)
        c = comp(r, "sleeve_length")
        self.assertEqual(c["body_measurement"], "arm_length")
        self.assertEqual(c["status"], "too_short")
        self.assertEqual(r["recommended_size"], "M")  # short sleeve warns, does not block

    def test_unknown_category_rejected(self):
        r = recommend_size(body(**MEDIUM), {"category": "hat", "sizes": [
            {"label": "M", "measurements": {"chest": 100}}]})
        self.assertEqual(r["status"], "invalid_input")
        self.assertIn("unknown_category", codes(r["errors"]))


if __name__ == "__main__":
    unittest.main()

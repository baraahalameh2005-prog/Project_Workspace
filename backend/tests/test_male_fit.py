"""
Tests for the adult male fit (fit_to_measurements.fit_adult_male) and the
optional male shoulder measurement.

Run from the backend folder:

    ..\\.venv\\Scripts\\python.exe -B -m unittest tests.test_male_fit -v

The fast tests use fakes: the API contract, validation and a synthetic,
linear "body" for the solver. RealMaleFitTests load the Anny face model
(about 1 GB of memory) and run real fits (a few seconds each); set
AVATAR_SKIP_MODEL_TESTS=1 to skip them. The test inputs are synthetic
test data, not measurements of real people.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import fit_to_measurements as ftm
from tests.test_avatar import VALID_REQUEST, fake_fit_body


MALE_REQUEST = {
    "gender": "male",
    "height": 178,
    "weight": 78,
    "waist": 86,
    "chest": 100,
    "hip": 98,
}


# ============================================================
# API CONTRACT (FAKES)
# ============================================================

class MaleApiTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        import api

        cls.api = api
        cls.client = TestClient(api.app)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.temp = Path(temp.name)

        def fake_fit(**kwargs):
            fitted = fake_fit_body(**kwargs)
            if kwargs["gender"] == "male" and kwargs.get("shape_model") is not None:
                fitted["fit_method"] = "adult_male"
                fitted["body_controls"] = {"measure-waist-circ-incr": -0.25}
            return fitted

        def fake_export(output, face, path, appearance):
            Path(path).write_bytes(b"glTF fake")

        self.face = mock.MagicMock()
        patches = [
            mock.patch.object(self.api, "get_model", return_value=("model", "meas")),
            mock.patch.object(self.api, "fit_body", side_effect=fake_fit),
            mock.patch.object(self.api, "export_appearance_avatar", side_effect=fake_export),
            mock.patch.object(self.api, "build"),
            mock.patch.object(self.api, "OUTPUT_DIR", self.temp / "generated"),
            mock.patch.object(self.api, "AVATAR_PATH", self.temp / "generated_avatar.glb"),
            mock.patch.object(self.api.generated_files, "all_directories", [str(self.temp / "generated")]),
            mock.patch.object(self.api, "get_face_model", return_value=self.face),
            # Male bodies are measured again after the build; the build is faked here.
            mock.patch.object(self.api.ftm, "report_male_body",
                              side_effect=lambda output, meas, targets, oor: ({}, {}, list(oor))),
        ]
        mocks = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)
        self.fit_mock = mocks[1]

    def post(self, body):
        return self.client.post("/generate-avatar", json=body)

    def test_male_uses_adult_fit_with_shoulder(self):
        response = self.post({**MALE_REQUEST, "shoulder": 41})

        self.assertEqual(response.status_code, 200, response.text)
        kwargs = self.fit_mock.call_args.kwargs
        self.assertEqual(kwargs["shoulder_cm"], 41)
        self.assertIs(kwargs["shape_model"], self.face.controls)

        data = response.json()
        self.assertEqual(data["fit_method"], "adult_male")
        self.assertEqual(data["body_controls"], {"measure-waist-circ-incr": -0.25})

    def test_body_controls_reach_the_exported_body(self):
        self.post({**MALE_REQUEST, "shoulder": 41})

        # The last controls() call builds the exported body: face preset + body controls.
        exported = self.face.controls.call_args.args[0]
        self.assertEqual(exported["measure-waist-circ-incr"], -0.25)
        self.assertTrue(any(k.startswith(("chin", "mouth", "nose")) for k in exported))

    def test_male_request_without_shoulder_still_works(self):
        response = self.post(MALE_REQUEST)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(self.fit_mock.call_args.kwargs["shoulder_cm"])

    def test_female_keeps_the_standard_fit(self):
        response = self.post(VALID_REQUEST)

        self.assertEqual(response.status_code, 200, response.text)
        kwargs = self.fit_mock.call_args.kwargs
        self.assertIsNone(kwargs["shape_model"])
        self.assertIsNone(kwargs["shoulder_cm"])
        data = response.json()
        self.assertEqual(data["fit_method"], "standard")
        self.assertEqual(data["body_controls"], {})

    def test_female_shoulder_is_ignored_with_a_warning(self):
        response = self.post({**VALID_REQUEST, "gender": "female", "shoulder": 38})

        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(self.fit_mock.call_args.kwargs["shoulder_cm"])
        self.assertTrue(any(w.startswith("shoulder:") for w in response.json()["warnings"]))

    def test_invalid_shoulder_is_rejected(self):
        for value in [29.9, 60.1, -40, "wide", "NaN", "Infinity"]:
            with self.subTest(value=value):
                self.fit_mock.reset_mock()
                response = self.client.post(
                    "/generate-avatar",
                    content=json.dumps({**MALE_REQUEST, "shoulder": value})
                    if value not in ("NaN", "Infinity")
                    else '{"gender":"male","height":178,"weight":78,"waist":86,"chest":100,"hip":98,'
                    f'"shoulder":{value}}}',
                    headers={"Content-Type": "application/json"},
                )
                self.assertEqual(response.status_code, 422, response.text)
                self.fit_mock.assert_not_called()

    def test_options_list_the_optional_shoulder(self):
        data = self.client.get("/avatar-options").json()

        shoulder = data["optional_measurements"]["shoulder"]
        self.assertEqual(
            shoulder,
            {"required": False, "minimum": 30.0, "maximum": 60.0, "unit": "cm", "genders": ["male"]},
        )
        # Required measurements are unchanged.
        self.assertEqual(set(data["measurements"]), {"height", "weight", "waist", "chest", "hip"})


# ============================================================
# PURE FUNCTIONS
# ============================================================

class MaleReportTests(unittest.TestCase):

    def test_shoulder_accuracy(self):
        report = ftm.accuracy_report(
            {"shoulder": 40.0},
            {"shoulder_breadth_m": 0.395},
            controls=ftm.ADULT_MALE_CONTROL,
        )

        self.assertAlmostEqual(report["shoulder"]["actual"], 39.5)
        self.assertAlmostEqual(report["shoulder"]["absolute_error"], 0.5)
        self.assertEqual(report["shoulder"]["control"], ftm.ADULT_MALE_CONTROL["shoulder"])

    def test_default_controls_unchanged(self):
        report = ftm.accuracy_report({"waist": 70.0}, {"waist_m": 0.7})

        self.assertEqual(report["waist"]["control"], ftm.TARGET_CONTROL["waist"])

    def test_density_correction(self):
        corrected = ftm._density_corrected({"mass_kg": 73.5, "bmi": 23.2, "waist_m": 0.86, "chest_m": None})

        factor = ftm.MALE_BODY_DENSITY / ftm.ANNY_DENSITY
        self.assertAlmostEqual(corrected["mass_kg"], 73.5 * factor)
        self.assertAlmostEqual(corrected["bmi"], 23.2 * factor)
        self.assertEqual(corrected["waist_m"], 0.86)
        self.assertIsNone(corrected["chest_m"])

    def test_adult_age_is_adult_by_annys_calibration(self):
        # anny/data/shape_calibration: 0.77 -> 18 years, 0.83 -> 64 years.
        self.assertGreaterEqual(ftm.ADULT_MALE_AGE, 0.77)
        self.assertLessEqual(ftm.ADULT_MALE_AGE, 0.83)


class _LinearBody:
    """
    Stand-in for Anny: girths and mass are linear in the phenotype and the
    controls, with the couplings of the real ones (the shoulder control also
    widens the chest; the weight parameter, body fat, widens every girth;
    controls add or remove mass). evaluate_shape solves the weight parameter
    like the real one; evaluate builds a body without solving anything.
    """

    def __init__(self):
        self.seen = set()

    def shape_model(self, controls):
        self.seen.update(controls)
        return dict(controls)

    @staticmethod
    def girths(c, weight, muscle):
        fat = weight - 0.5
        return {
            "chest_tape_m": 1.00 + 0.14 * c.get("measure-bust-circ-incr", 0.0)
            + 0.085 * c.get("torso-vshape-incr", 0.0) + 0.10 * fat,
            "waist_m": 0.85 + 0.10 * c.get("measure-waist-circ-incr", 0.0) + 0.15 * fat,
            "hip_tape_m": 0.98 + 0.145 * c.get("measure-hips-circ-incr", 0.0) + 0.10 * fat,
            "shoulder_breadth_m": 0.39 + 0.053 * c.get("torso-vshape-incr", 0.0),
        }

    @staticmethod
    def mass(c, weight, muscle):
        controls = sum(c.get(k, 0.0) for k in ("measure-bust-circ-incr", "measure-waist-circ-incr",
                                               "measure-hips-circ-incr"))
        return 45.0 + 40.0 * weight + 25.0 * (muscle - 0.5) + 3.0 * controls

    def evaluate(self, model, measurements, gender, age, height, weight, muscle, proportions):
        result = {**self.girths(model, weight, muscle), "mass_kg": self.mass(model, weight, muscle),
                  "height_m": 1.78, "bmi": 22.0, "chest_m": None, "hip_m": None}
        return {"result": result}, dict(result)

    def evaluate_shape(self, model, measurements, gender, target_height, target_weight, age, muscle,
                       proportions, start=None, stats=None):
        base = self.mass(model, 0.0, muscle)
        weight = float(np.clip((target_weight - base) / 40.0, ftm.PARAM_MIN, ftm.PARAM_MAX))
        output, result = self.evaluate(model, measurements, gender, age, 0.5, weight, muscle, proportions)
        return output, result, 0.5, weight


class _FakeMeasurements:
    @staticmethod
    def tape_measurements(output):
        r = output["result"]
        return {"chest_tape_m": r["chest_tape_m"], "hip_tape_m": r["hip_tape_m"], "chest_tape_z": 0.0,
                "hip_tape_z": 0.0}


class MaleSolverTests(unittest.TestCase):

    def fit(self, **targets):
        body = _LinearBody()
        case = dict(height_cm=178, weight_kg=78, waist_cm=86, chest_cm=100, hip_cm=98)
        case.update(targets)
        stats = {}
        with mock.patch.object(ftm, "evaluate_shape", side_effect=body.evaluate_shape), \
                mock.patch.object(ftm, "evaluate", side_effect=body.evaluate), \
                mock.patch.object(ftm, "solve_height", return_value=0.5):
            _, result, params, controls = ftm.fit_adult_male(
                shape_model=body.shape_model, measurements=_FakeMeasurements(), stats=stats, **case)
        return result, params, controls, stats, body

    def test_reaches_reachable_targets(self):
        result, params, _, stats, _ = self.fit(shoulder_cm=40)

        self.assertAlmostEqual(result["chest_tape_m"], 1.00, delta=0.003)
        self.assertAlmostEqual(result["waist_m"], 0.86, delta=0.003)
        self.assertAlmostEqual(result["hip_tape_m"], 0.98, delta=0.003)
        self.assertAlmostEqual(result["shoulder_breadth_m"], 0.40, delta=0.003)
        # Reported mass is for MALE_BODY_DENSITY.
        self.assertAlmostEqual(result["mass_kg"], 78.0, delta=ftm.MALE_WEIGHT_PASS_TOLERANCE_KG)
        self.assertEqual(stats["out_of_range"], [])
        self.assertEqual(params[2], ftm.ADULT_MALE_AGE)
        self.assertEqual(params[4], ftm.ADULT_MALE_PROPORTIONS)

    def test_controls_do_not_spiral_into_body_fat(self):
        # The phenotype alone already has these girths: the controls must stay
        # small and the weight parameter where the phenotype put it, instead of
        # trading mass for fat (the spiral that the frozen phenotype prevents).
        body = _LinearBody()
        weight = (78.0 * ftm.ANNY_DENSITY / ftm.MALE_BODY_DENSITY - body.mass({}, 0.0, 0.5)) / 40.0
        g = body.girths({}, weight, 0.5)
        _, params, controls, stats, _ = self.fit(
            chest_cm=g["chest_tape_m"] * 100, waist_cm=g["waist_m"] * 100, hip_cm=g["hip_tape_m"] * 100)

        for value in controls.values():
            self.assertLess(abs(value), 0.02)
        self.assertAlmostEqual(params[1], weight, delta=0.01)
        self.assertEqual(stats["out_of_range"], [])

    def test_only_the_measurement_controls_are_used(self):
        # Never Anny's pregnancy shape or any other body control.
        _, _, controls, _, body = self.fit(shoulder_cm=40, waist_cm=110, hip_cm=80)

        allowed = {label for label, _ in ftm.MALE_TARGET_CONTROLS.values()}
        self.assertLessEqual(set(controls), allowed)
        self.assertLessEqual(body.seen, allowed)
        self.assertNotIn("stomach-pregnant-incr", body.seen)

    def test_impossible_shoulder_does_not_move_waist_or_hip(self):
        reference, _, _, _, _ = self.fit()
        result, _, controls, stats, _ = self.fit(shoulder_cm=55)

        self.assertEqual(controls["torso-vshape-incr"], 1.0)
        self.assertIn("shoulder", stats["out_of_range"])
        for key in ("waist_m", "hip_tape_m"):
            self.assertAlmostEqual(result[key], reference[key], delta=0.003)

    def test_chest_never_goes_below_the_bust_floor(self):
        result, _, controls, stats, _ = self.fit(chest_cm=80)

        self.assertEqual(controls["measure-bust-circ-incr"], ftm.MALE_CONTROL_BOUNDS["measure-bust-circ-incr"][0])
        self.assertIn("chest", stats["out_of_range"])
        self.assertGreater(result["chest_tape_m"], 0.80)

    def test_unreachable_weight_is_reported_and_muscle_stays_capped(self):
        # 130 kg is beyond the weight parameter and the muscle cap of this body.
        result, params, _, stats, _ = self.fit(weight_kg=130)

        self.assertLessEqual(params[3], ftm.MALE_MUSCLE_RANGE[1])
        self.assertIn("weight", stats["out_of_range"])
        self.assertLess(result["mass_kg"], 129.0)

    def test_no_shoulder_means_no_shoulder_control(self):
        _, _, controls, _, _ = self.fit()

        self.assertNotIn("torso-vshape-incr", controls)


# ============================================================
# REAL ANNY MODEL
# ============================================================

@unittest.skipIf(
    os.environ.get("AVATAR_SKIP_MODEL_TESTS") == "1",
    "AVATAR_SKIP_MODEL_TESTS=1",
)
class RealMaleFitTests(unittest.TestCase):
    """Synthetic test cases (not real people). Acceptance criteria fixed before the fit was tuned."""

    CIRCUMFERENCE_TOLERANCE_CM = 2.0
    SHOULDER_TOLERANCE_CM = 1.5
    HEIGHT_TOLERANCE_CM = 0.5
    WEIGHT_TOLERANCE_KG = 1.0
    REPORTED_WEIGHT_LIMIT_KG = 2.5

    BASE = dict(height_cm=178, weight_kg=78, waist_cm=86, chest_cm=100, hip_cm=98, shoulder_cm=39)

    @classmethod
    def setUpClass(cls):
        from appearance import FaceModel
        from measurements import BodyMeasurements

        cls.face = FaceModel()
        cls.meas = BodyMeasurements(cls.face.model)
        cls._cache = {}

    def fit(self, **changes):
        case = dict(self.BASE)
        case.update(changes)
        key = tuple(sorted(case.items()))
        if key not in self._cache:
            self._cache[key] = ftm.fit_body(
                model=None,
                measurements=self.meas,
                gender="male",
                shape_model=self.face.controls,
                **case,
            )
        return self._cache[key]

    def assert_within_criteria(self, fitted):
        for name, entry in fitted["accuracy"].items():
            with self.subTest(target=name):
                limit = {
                    "height": self.HEIGHT_TOLERANCE_CM,
                    "weight": self.WEIGHT_TOLERANCE_KG,
                    "shoulder": self.SHOULDER_TOLERANCE_CM,
                }.get(name, self.CIRCUMFERENCE_TOLERANCE_CM)
                # Weight is matched only as far as it keeps the body natural
                # (fit_adult_male, stage C): a larger miss must be reported,
                # and stay within REPORTED_WEIGHT_LIMIT_KG for these cases.
                if name == "weight" and "weight" in fitted["out_of_range"]:
                    limit = self.REPORTED_WEIGHT_LIMIT_KG
                self.assertEqual(entry["status"], "measured")
                self.assertLessEqual(entry["absolute_error"], limit, entry)

    def test_average_male_meets_the_criteria(self):
        fitted = self.fit()

        self.assertEqual(fitted["fit_method"], "adult_male")
        self.assert_within_criteria(fitted)
        for value in fitted["body_controls"].values():
            self.assertTrue(-1.0 <= value <= 1.0)

    def test_average_man_stays_natural(self):
        """
        An average man (175 cm, 75 kg, 98/85/97, shoulder 40; synthetic) is
        mostly the plain adult phenotype: body fat mid-range, small controls,
        normal upper arms. Before the staged fit this case ended at maximum
        body fat with every control compressing it.
        """

        fitted = ftm.fit_body(
            model=None, measurements=self.meas, gender="male", shape_model=self.face.controls,
            height_cm=175, weight_kg=75, waist_cm=85, chest_cm=98, hip_cm=97, shoulder_cm=40,
        )

        self.assert_within_criteria(fitted)
        self.assertLess(fitted["parameters"]["weight"], 0.85)
        self.assertLessEqual(max(abs(v) for v in fitted["body_controls"].values()), 0.5)
        self.assertLess(fitted["result"]["upper_arm_m"] * 100, 34.0)
        self.assertEqual(fitted["out_of_range"], [])

    def test_male_body_is_adult(self):
        params = self.fit()["parameters"]

        self.assertGreaterEqual(params["age"], 0.77)

    def test_shoulder_input_changes_the_geometry(self):
        narrow = self.fit(shoulder_cm=36)
        broad = self.fit(shoulder_cm=43)

        self.assert_within_criteria(narrow)
        # Anny's shoulder control also widens the chest, and the bust control
        # does not go below its floor (a flat chest): broad shoulders with a
        # 100 cm chest leave the chest larger, and that must be reported.
        for name, entry in broad["accuracy"].items():
            if name == "chest" and "chest" in broad["out_of_range"]:
                continue
            with self.subTest(target=name):
                limit = {"height": self.HEIGHT_TOLERANCE_CM, "weight": self.WEIGHT_TOLERANCE_KG,
                         "shoulder": self.SHOULDER_TOLERANCE_CM}.get(name, self.CIRCUMFERENCE_TOLERANCE_CM)
                self.assertLessEqual(entry["absolute_error"], limit, entry)
        if broad["accuracy"]["chest"]["absolute_error"] > self.CIRCUMFERENCE_TOLERANCE_CM:
            self.assertIn("chest", broad["out_of_range"])
        gain = (broad["result"]["shoulder_breadth_m"] - narrow["result"]["shoulder_breadth_m"]) * 100
        self.assertGreater(gain, 0.75 * 7)
        # The joints move with the skin, so arms attach further out.
        self.assertGreater(broad["result"]["shoulder_width_m"], narrow["result"]["shoulder_width_m"] + 0.04)

    def test_each_circumference_moves_alone(self):
        base = self.fit()["result"]

        for name, key in (("waist_cm", "waist_m"), ("chest_cm", "chest_tape_m"), ("hip_cm", "hip_tape_m")):
            with self.subTest(changed=name):
                changed = self.fit(**{name: self.BASE[name] + 8})
                self.assert_within_criteria(changed)
                result = changed["result"]
                self.assertGreater((result[key] - base[key]) * 100, 0.75 * 8)
                for other in ("waist_m", "chest_tape_m", "hip_tape_m"):
                    if other != key:
                        self.assertLessEqual(abs(result[other] - base[other]) * 100, 1.5, other)

    def test_height_changes_length_not_girth(self):
        base = self.fit()["result"]
        tall = self.fit(height_cm=188)

        self.assert_within_criteria(tall)
        self.assertGreater(tall["result"]["leg_length_m"], base["leg_length_m"] + 0.03)
        for key in ("waist_m", "chest_tape_m", "hip_tape_m"):
            self.assertLessEqual(abs(tall["result"][key] - base[key]) * 100, 1.5, key)

    def test_exported_glb(self):
        from appearance import appearance_for, export_appearance_avatar
        from generate_avatar import build

        fitted = self.fit()
        p = fitted["parameters"]
        app = appearance_for(gender="male", skin_tone="#c8997e", eye_color="brown")
        output = build(
            model=self.face.controls({**app.face_controls, **fitted["body_controls"]}),
            gender=0.0, height=p["height"], weight=p["weight"], age=p["age"],
            muscle=p["muscle"], proportions=p["proportions"],
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "male.glb"
            export_appearance_avatar(output=output, face=self.face, path=path, appearance=app)

            import trimesh

            mesh = trimesh.load(path, force="mesh")
            vertices = np.asarray(mesh.vertices)
            self.assertTrue(np.all(np.isfinite(vertices)))
            height = vertices[:, 1].max() - vertices[:, 1].min()
            self.assertAlmostEqual(height * 100, 178, delta=1.0)
            self.assertAlmostEqual(vertices[:, 1].min(), 0.0, delta=0.01)

    @staticmethod
    def belly_lead_cm(output):
        """How far the abdomen's front reaches beyond the chest's front (rest pose, z up, front is -y)."""

        v = output["rest_vertices"][0].detach().cpu().numpy()
        z = (v[:, 2] - v[:, 2].min()) / (v[:, 2].max() - v[:, 2].min())
        torso = np.abs(v[:, 0]) < 0.12

        def front(low, high):
            return -v[torso & (z > low) & (z < high), 1].min()

        return (front(0.52, 0.62) - front(0.70, 0.76)) * 100

    def test_reported_avatar_has_no_hanging_belly(self):
        """
        The avatar the user reported (551c0fd9..., 2026-10-17): a round belly
        10.3 cm in front of the chest. These are the inputs reconstructed from
        that mesh (mean vertex distance 1.45 mm); the 52 cm shoulder is beyond
        what the shoulder control can reach at 170 cm.
        """

        fitted = ftm.fit_body(
            model=None, measurements=self.meas, gender="male", shape_model=self.face.controls,
            height_cm=170, weight_kg=78, waist_cm=94, chest_cm=103, hip_cm=102, shoulder_cm=52,
        )

        self.assertLessEqual(self.belly_lead_cm(fitted["output"]), 4.0)
        self.assertNotIn("stomach-pregnant-incr", fitted["body_controls"])
        self.assertIn("shoulder", fitted["out_of_range"])
        self.assertGreaterEqual(fitted["body_controls"]["measure-bust-circ-incr"], -0.6)
        for name in ("height", "weight", "waist", "hip"):
            limit = {"height": self.HEIGHT_TOLERANCE_CM, "weight": self.WEIGHT_TOLERANCE_KG}.get(
                name, self.CIRCUMFERENCE_TOLERANCE_CM)
            self.assertLessEqual(fitted["accuracy"][name]["absolute_error"], limit, name)

    def test_female_fit_is_unchanged(self):
        """The female path is the recorded production fit (tools/baseline_cases.json)."""

        import anny
        from measurements import BodyMeasurements

        cases = json.loads((ROOT / "tools" / "baseline_cases.json").read_text())["cases"]
        case = next(c for c in cases if c["case"] == "F1_162.6cm_50kg")
        plain = anny.Anny()
        fitted = ftm.fit_body(
            model=plain,
            measurements=BodyMeasurements(plain),
            gender="female",
            shoulder_cm=40,
            shape_model=self.face.controls,
            **{k: case["input"][k] for k in ("height_cm", "weight_kg", "waist_cm", "chest_cm", "hip_cm")},
        )

        self.assertEqual(fitted["fit_method"], "standard")
        self.assertEqual(fitted["body_controls"], {})
        self.assertNotIn("shoulder", fitted["accuracy"])
        for name, value in case["parameters"].items():
            if name != "gender":
                self.assertAlmostEqual(fitted["parameters"][name], value, places=9, msg=name)


if __name__ == "__main__":
    unittest.main()

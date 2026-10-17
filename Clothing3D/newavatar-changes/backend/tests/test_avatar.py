"""
Fast tests for the avatar backend.

Run from the backend folder:

    ..\\.venv\\Scripts\\python.exe -B -m unittest tests.test_avatar -v

The API tests replace the model, the fitting and the GLB export with
fakes, so they check the request/response contract only, not real
measurement accuracy.

RealModelTests load Anny (about 40 s) and run a few single body
evaluations. They never run the full fitting optimizer and never write
the project's generated_avatar.glb. Set AVATAR_SKIP_MODEL_TESTS=1 to
skip them.
"""

import hashlib
import json
import math
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
from generate_avatar import parse_skin_tone, skin_base_color, srgb_to_linear


PROJECT_GLB = ROOT / "generated_avatar.glb"


def file_hash(path):
    if not path.exists():
        return None

    return hashlib.sha256(path.read_bytes()).hexdigest()


VALID_REQUEST = {
    "height": 162.62,
    "weight": 49.99,
    "waist": 71.80,
    "chest": 86.60,
    "hip": 92.15,
}


# ============================================================
# PURE FUNCTIONS
# ============================================================

class PercentageErrorTests(unittest.TestCase):

    def test_formula(self):
        self.assertAlmostEqual(ftm.percentage_error(80.0, 84.0), 5.0)
        self.assertAlmostEqual(ftm.percentage_error(80.0, 76.0), 5.0)
        self.assertAlmostEqual(ftm.percentage_error(-10.0, -11.0), 10.0)
        self.assertEqual(ftm.percentage_error(50.0, 50.0), 0.0)

    def test_invalid_values_give_none(self):
        self.assertIsNone(ftm.percentage_error(0.0, 1.0))
        self.assertIsNone(ftm.percentage_error(None, 1.0))
        self.assertIsNone(ftm.percentage_error(1.0, None))
        self.assertIsNone(ftm.percentage_error(float("nan"), 1.0))
        self.assertIsNone(ftm.percentage_error(1.0, float("inf")))
        self.assertIsNone(ftm.percentage_error("abc", 1.0))


class AccuracyReportTests(unittest.TestCase):

    def test_units_and_errors(self):
        result = {
            "height_m": 1.63,
            "mass_kg": 50.5,
            "waist_m": 0.70,
            "chest_m": 0.90,
            "hip_m": 0.92,
        }

        report = ftm.accuracy_report(
            {"height": 162.0, "weight": 50.0, "chest": 86.6},
            result,
        )

        self.assertEqual(report["height"]["unit"], "cm")
        self.assertAlmostEqual(report["height"]["actual"], 163.0)
        self.assertAlmostEqual(report["height"]["absolute_error"], 1.0)

        self.assertEqual(report["weight"]["unit"], "kg")
        self.assertAlmostEqual(report["weight"]["actual"], 50.5)
        self.assertAlmostEqual(report["weight"]["percentage_error"], 1.0)

        self.assertAlmostEqual(report["chest"]["actual"], 90.0)
        self.assertAlmostEqual(report["chest"]["absolute_error"], 3.4)
        self.assertAlmostEqual(
            report["chest"]["percentage_error"],
            3.4 / 86.6 * 100.0,
        )

        for entry in report.values():
            self.assertEqual(entry["status"], "measured")

    def test_missing_measurement_is_reported_not_dropped(self):
        report = ftm.accuracy_report(
            {"chest": 86.6, "hip": 92.0},
            {"chest_m": None, "hip_m": float("nan")},
        )

        for name in ("chest", "hip"):
            self.assertEqual(report[name]["status"], "measurement_failed")
            self.assertIsNone(report[name]["actual"])
            self.assertIsNone(report[name]["absolute_error"])
            self.assertIsNone(report[name]["percentage_error"])
            self.assertIsNotNone(report[name]["requested"])

    def test_zero_requested_value(self):
        report = ftm.accuracy_report({"waist": 0.0}, {"waist_m": 0.7})

        self.assertEqual(report["waist"]["status"], "measured")
        self.assertIsNone(report["waist"]["percentage_error"])


class GenderTests(unittest.TestCase):

    def test_valid_values(self):
        self.assertEqual(ftm.normalize_gender("female"), "female")
        self.assertEqual(ftm.normalize_gender("Male"), "male")
        self.assertEqual(ftm.normalize_gender("  FEMALE "), "female")

        self.assertEqual(ftm.gender_value("female"), 1.0)
        self.assertEqual(ftm.gender_value("male"), 0.0)

    def test_invalid_values(self):
        for value in ["", "f", "m", "woman", "other", None, 1, 0.0]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ftm.normalize_gender(value)


class SolverTests(unittest.TestCase):

    def run_solver(self, func, target, start, tolerance):
        calls = []

        def wrapped(x):
            calls.append(x)
            return func(x), None

        x, value, _, n = ftm.solve_monotonic(
            wrapped, target, start, tolerance
        )

        self.assertEqual(n, len(calls))

        return x, value, n

    def test_linear(self):
        x, value, n = self.run_solver(
            lambda x: 1.4 + 0.6 * x, 1.70, 0.5, 1e-4
        )

        self.assertLessEqual(abs(value - 1.70), 1e-4)
        self.assertLessEqual(n, 3)

    def test_nonlinear(self):
        x, value, n = self.run_solver(
            lambda x: 30.0 + 90.0 * x ** 3, 70.0, 0.5, 1e-2
        )

        self.assertLessEqual(abs(value - 70.0), 1e-2)
        self.assertLessEqual(n, 10)

    def test_unreachable_target_returns_closest(self):
        x, value, _ = self.run_solver(
            lambda x: x, 2.0, 0.5, 1e-6
        )

        self.assertEqual(x, ftm.PARAM_MAX)
        self.assertAlmostEqual(value, x)

    def test_unreachable_target_checks_bound_early(self):
        _, _, n = self.run_solver(lambda x: 40.0 + 10.0 * x, 65.0, 0.5, 1e-2)

        self.assertLessEqual(n, 3)


class PolygonContainsOriginTests(unittest.TestCase):

    def test_square(self):
        from measurements import _polygon_contains_origin

        square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)

        self.assertTrue(_polygon_contains_origin(square))
        self.assertFalse(_polygon_contains_origin(square + [3, 0]))

    def test_concave_shape_around_origin(self):
        from measurements import _polygon_contains_origin

        # Arm lobe at the origin joined to a large "torso" lobe above it.
        merged = np.array([[-1, -1], [1, -1], [1, 1], [8, 2], [8, 30], [-8, 30], [-8, 2], [-1, 1]], dtype=float)

        self.assertTrue(_polygon_contains_origin(merged))


class SkinToneTests(unittest.TestCase):

    def test_parse(self):
        self.assertEqual(parse_skin_tone("#c8997e"), [200, 153, 126, 255])
        self.assertEqual(parse_skin_tone("FFFFFF"), [255, 255, 255, 255])

    def test_invalid(self):
        for value in ["", "#fff", "#gggggg", "red", None]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_skin_tone(value)

    def test_srgb_to_linear(self):
        self.assertEqual(float(srgb_to_linear(0)), 0.0)
        self.assertAlmostEqual(float(srgb_to_linear(255)), 1.0, places=12)
        # 10/255 is on the linear segment (c / 12.92).
        self.assertAlmostEqual(float(srgb_to_linear(10)), 10 / 255 / 12.92, places=12)
        self.assertAlmostEqual(float(srgb_to_linear(200)), 0.57758, places=4)

        values = srgb_to_linear(np.arange(256))
        self.assertTrue(np.all(np.diff(values) > 0))

    def test_srgb_to_linear_matches_trimesh(self):
        from trimesh.visual import color

        channels = np.arange(256, dtype=np.uint8)

        np.testing.assert_allclose(srgb_to_linear(channels), color.srgb_to_linear(channels), rtol=0, atol=1e-12)

    def test_skin_base_color(self):
        np.testing.assert_allclose(skin_base_color("#c8997e"), [0.57758, 0.318547, 0.208637, 1.0], atol=1e-6)
        np.testing.assert_allclose(skin_base_color("a0705a"), [0.351533, 0.162029, 0.102242, 1.0], atol=1e-6)

        with self.assertRaises(ValueError):
            skin_base_color("pink")


# ============================================================
# API CONTRACT (FAKE MODEL, NO FILE WRITES)
# ============================================================

def fake_fit_body(**kwargs):
    result = {
        "height_m": kwargs["height_cm"] / 100.0,
        "mass_kg": kwargs["weight_kg"],
        "bmi": 20.0,
        "waist_m": kwargs["waist_cm"] / 100.0 * 1.01,
        "chest_m": None,
        "hip_m": kwargs["hip_cm"] / 100.0,
        "hip_z": -0.1,
        "neck_m": 0.33,
    }

    accuracy = ftm.accuracy_report(
        {
            "height": kwargs["height_cm"],
            "weight": kwargs["weight_kg"],
            "waist": kwargs["waist_cm"],
            "chest": kwargs["chest_cm"],
            "hip": kwargs["hip_cm"],
        },
        result,
    )

    return {
        "output": object(),
        "result": result,
        "accuracy": accuracy,
        "fit_info": {},
        "parameters": {
            "gender": ftm.normalize_gender(kwargs["gender"]),
            "height": 0.5,
            "weight": 0.5,
            "age": 0.5,
            "muscle": 0.5,
            "proportions": 0.5,
        },
    }


class ModuleConfigurationTests(unittest.TestCase):
    """The module's api.py takes its folders from app/config.py."""

    def test_api_uses_module_folders(self):
        import api
        from app.config import settings

        self.assertEqual(api.OUTPUT_DIR, settings.output_dir)
        self.assertEqual(
            Path(api.generated_files.directory),
            settings.output_dir,
        )
        self.assertFalse(api.WRITE_LEGACY_AVATAR)

    def test_anny_and_warp_caches_are_inside_the_module(self):
        import anny.paths
        from app import config

        self.assertTrue(
            config.is_inside(anny.paths.get_anny_cache_path(), config.MODULE_ROOT)
        )
        self.assertTrue(
            config.is_inside(os.environ["WARP_CACHE_PATH"], config.MODULE_ROOT)
        )

    def test_torch_inductor_cache_is_inside_the_module(self):
        # The folder PyTorch itself resolves (and has already created when
        # torch was imported by fit_to_measurements above).
        from torch._inductor.runtime.cache_dir_utils import cache_dir

        from app import config

        self.assertTrue(config.is_inside(cache_dir(), config.MODULE_ROOT))


class ApiContractTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        import api

        cls.api = api

        # Not used as a context manager, so the lifespan (which loads
        # Anny) does not run.
        cls.client = TestClient(api.app)

    def setUp(self):
        self.glb_hash = file_hash(PROJECT_GLB)

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.temp = Path(temp.name)

        def fake_export(output, face, path, appearance):
            Path(path).write_bytes(b"glTF fake " + Path(path).stem.encode())

        patches = [
            mock.patch.object(
                self.api, "get_model", return_value=("model", "meas")
            ),
            mock.patch.object(
                self.api, "fit_body", side_effect=fake_fit_body
            ),
            mock.patch.object(
                self.api, "export_appearance_avatar", side_effect=fake_export
            ),
            mock.patch.object(self.api, "build"),
            mock.patch.object(
                self.api, "AVATAR_PATH", self.temp / "generated_avatar.glb"
            ),
            mock.patch.object(
                self.api, "OUTPUT_DIR", self.temp / "generated"
            ),
            mock.patch.object(
                self.api.generated_files,
                "all_directories",
                [str(self.temp / "generated")],
            ),
            # The face model is not loaded; build is mocked anyway.
            mock.patch.object(self.api, "get_face_model"),
        ]

        self.mocks = [p.start() for p in patches]

        for p in patches:
            self.addCleanup(p.stop)

        self.fit_mock = self.mocks[1]
        self.export_mock = self.mocks[2]

    def tearDown(self):
        self.assertEqual(file_hash(PROJECT_GLB), self.glb_hash)

    def post(self, **changes):
        body = dict(VALID_REQUEST)
        body.update(changes)
        body = {k: v for k, v in body.items() if v is not None}

        return self.client.post("/generate-avatar", json=body)

    def test_root(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"message": "Avatar API is running"},
        )

    def test_default_request_is_female(self):
        response = self.post()

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            self.fit_mock.call_args.kwargs["gender"],
            "female",
        )
        self.assertEqual(
            response.json()["parameters"]["gender"],
            "female",
        )

    def test_response_contract(self):
        data = self.post(gender="female").json()

        for key in [
            "message",
            "avatar_url",
            "parameters",
            "measurements",
            "avatar_id",
            "legacy_avatar_url",
            "warnings",
            "accuracy",
            "failed_measurements",
            "fit_time_seconds",
        ]:
            self.assertIn(key, data)

        self.assertEqual(data["message"], "Avatar generated successfully")
        self.assertEqual(
            data["avatar_url"],
            f"/models/generated/{data['avatar_id']}.glb",
        )
        # The module does not write the shared legacy file.
        self.assertIsNone(data["legacy_avatar_url"])

        for key in ["gender", "height", "weight", "age", "muscle", "proportions"]:
            self.assertIn(key, data["parameters"])

        self.assertNotIn("hip_z", data["measurements"])

        for name in ["height", "weight", "waist", "chest", "hip"]:
            self.assertIn(name, data["accuracy"])

    def test_export_goes_to_a_unique_file_only(self):
        data = self.post().json()

        path = self.export_mock.call_args.kwargs["path"]

        self.assertEqual(path, self.api.OUTPUT_DIR / f"{data['avatar_id']}.glb")
        self.assertTrue(path.exists())

        # No legacy copy and no temporary files left behind.
        self.assertFalse(self.api.AVATAR_PATH.exists())
        self.assertEqual(
            sorted(p.name for p in self.temp.iterdir()),
            ["generated"],
        )

    def test_legacy_copy_when_turned_on(self):
        with mock.patch.object(self.api, "WRITE_LEGACY_AVATAR", True):
            data = self.post().json()

        path = self.api.OUTPUT_DIR / f"{data['avatar_id']}.glb"

        self.assertEqual(data["legacy_avatar_url"], "/models/generated_avatar.glb")
        self.assertEqual(self.api.AVATAR_PATH.read_bytes(), path.read_bytes())
        self.assertEqual(
            sorted(p.name for p in self.temp.iterdir()),
            ["generated", "generated_avatar.glb"],
        )

    def test_each_generation_gets_its_own_url_and_file(self):
        first = self.post(weight=50).json()
        second = self.post(weight=60).json()

        self.assertNotEqual(first["avatar_id"], second["avatar_id"])
        self.assertNotEqual(first["avatar_url"], second["avatar_url"])

        first_file = self.api.OUTPUT_DIR / f"{first['avatar_id']}.glb"
        second_file = self.api.OUTPUT_DIR / f"{second['avatar_id']}.glb"

        # The second generation does not change the first user's file.
        self.assertIn(first["avatar_id"].encode(), first_file.read_bytes())
        self.assertIn(second["avatar_id"].encode(), second_file.read_bytes())

        # Each URL serves its own file.
        for data, file in [(first, first_file), (second, second_file)]:
            response = self.client.get(data["avatar_url"])

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, file.read_bytes())

        # No shared legacy file is written.
        self.assertFalse(self.api.AVATAR_PATH.exists())

    def test_legacy_copy_failure_is_a_warning(self):
        with mock.patch.object(
            self.api, "WRITE_LEGACY_AVATAR", True
        ), mock.patch.object(
            self.api, "_copy_file_atomic", side_effect=PermissionError()
        ):
            response = self.post()

        self.assertEqual(response.status_code, 200)

        data = response.json()

        self.assertTrue(
            any("legacy avatar file" in w for w in data["warnings"])
        )
        self.assertEqual(
            self.client.get(data["avatar_url"]).status_code,
            200,
        )

    def test_warnings_follow_the_fitting_tolerances(self):
        data = self.post().json()

        warnings = " ".join(data["warnings"])

        # fake_fit_body: chest missing, waist 1% off, the rest exact.
        self.assertIn("chest: could not be measured", warnings)
        self.assertIn("waist:", warnings)

        for name in ["height:", "weight:", "hip:"]:
            self.assertNotIn(name, warnings)

    def test_accuracy_warnings_thresholds(self):
        def entry(requested, actual, unit="cm"):
            return {
                "status": "measured",
                "unit": unit,
                "requested": requested,
                "actual": actual,
                "absolute_error": abs(actual - requested),
                "percentage_error": ftm.percentage_error(requested, actual),
            }

        warnings = self.api.accuracy_warnings(
            {
                "height": entry(170.0, 170.005),
                "weight": entry(60.0, 60.5, "kg"),
                # Circumference tolerance: 1 % (body_style.STANDARD.good_enough).
                "chest": entry(90.0, 90.5),
                "hip": entry(100.0, 101.5),
            }
        )

        text = " ".join(warnings)

        self.assertNotIn("height:", text)
        self.assertIn("weight:", text)
        self.assertNotIn("chest:", text)
        self.assertIn("hip:", text)

    def test_health_does_not_load_the_model(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.mocks[0].assert_not_called()

    def test_avatar_options_match_validation(self):
        options = self.client.get("/avatar-options").json()

        self.assertEqual(
            set(options["measurements"]),
            {"height", "weight", "waist", "chest", "hip"},
        )
        self.assertEqual(options["gender"]["values"], ["female", "male"])
        self.assertEqual(options["skin_tone"]["format"], "#RRGGBB")
        self.assertEqual(options["eye_color"]["default"], "brown")
        self.assertEqual(set(options["eye_color"]["values"]), {"brown", "hazel", "green", "blue", "gray"})

        for name, spec in options["measurements"].items():
            with self.subTest(name=name):
                low, high = spec["minimum"], spec["maximum"]

                self.assertEqual(self.post(**{name: low}).status_code, 200)
                self.assertEqual(self.post(**{name: high}).status_code, 200)
                self.assertEqual(self.post(**{name: low - 0.01}).status_code, 422)
                self.assertEqual(self.post(**{name: high + 0.01}).status_code, 422)

    def static_mount(self, name):
        for route in self.api.app.routes:
            if getattr(route, "name", None) == name:
                return route.app

        raise AssertionError(f"no mount named {name!r}")

    def test_static_serving_is_limited(self):
        # Serve /models and /test from a fixture folder that looks like the
        # backend folder. Every blocked file below exists in it, so a 404
        # comes from the filter, not from a missing file.
        site = self.temp / "site"
        fixtures = {
            "generated_avatar.glb": b"glTF legacy",
            "shirt_S.glb": b"glTF shirt",
            "test_avatar.html": b"<html>viewer</html>",
            "api.py": b"code",
            "measurements.py": b"code",
            ".gitignore": b"x",
            ".git/config": b"x",
            "tests/test_avatar.py": b"code",
            "profile_baseline.txt": b"x",
            ".generated_avatar.tmp.glb": b"glTF partial",
        }

        for name, content in fixtures.items():
            path = site / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

        for name in ["models", "test"]:
            patch = mock.patch.object(
                self.static_mount(name), "all_directories", [str(site)]
            )
            patch.start()
            self.addCleanup(patch.stop)

        allowed = {
            "/models/generated_avatar.glb": "generated_avatar.glb",
            "/models/shirt_S.glb": "shirt_S.glb",
            "/test/test_avatar.html": "test_avatar.html",
        }

        for url, name in allowed.items():
            with self.subTest(url=url):
                response = self.client.get(url)

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, fixtures[name])

        blocked = [
            "/models/api.py",
            "/models/measurements.py",
            "/models/.gitignore",
            "/models/.git/config",
            "/models/tests/test_avatar.py",
            "/models/test_avatar.html",
            "/models/profile_baseline.txt",
            "/models/.generated_avatar.tmp.glb",
            "/models/../api.py",
            "/test/api.py",
            "/test/generated_avatar.glb",
            "/models/generated/../../api.py",
            "/models/generated/missing.glb",
        ]

        for url in blocked:
            with self.subTest(url=url):
                self.assertEqual(self.client.get(url).status_code, 404)

    def test_failed_measurement_is_reported(self):
        data = self.post().json()

        self.assertNotIn("chest_m", data["measurements"])
        self.assertIn("chest_m", data["failed_measurements"])
        self.assertEqual(
            data["accuracy"]["chest"]["status"],
            "measurement_failed",
        )
        self.assertIsNone(data["accuracy"]["chest"]["percentage_error"])

        waist = data["accuracy"]["waist"]
        self.assertAlmostEqual(waist["percentage_error"], 1.0, places=6)

    def test_gender_values(self):
        for value, expected in [("male", "male"), ("Male", "male"), ("FEMALE", "female")]:
            with self.subTest(value=value):
                response = self.post(gender=value)

                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    self.fit_mock.call_args.kwargs["gender"],
                    expected,
                )

    def test_invalid_gender_is_rejected(self):
        for value in ["f", "", "other", 1]:
            with self.subTest(value=value):
                self.fit_mock.reset_mock()

                response = self.post(gender=value)

                self.assertEqual(response.status_code, 422)
                self.fit_mock.assert_not_called()

    def test_invalid_numbers_are_rejected(self):
        cases = [
            {"height": -170},
            {"height": 0},
            {"height": 500},
            {"weight": 0},
            {"weight": 1000},
            {"waist": -1},
            {"chest": 10},
            {"hip": 900},
            {"weight": "heavy"},
        ]

        for change in cases:
            with self.subTest(change=change):
                self.fit_mock.reset_mock()

                response = self.post(**change)

                self.assertEqual(response.status_code, 422)
                self.fit_mock.assert_not_called()

    def test_non_finite_numbers_are_rejected(self):
        body = (
            '{"height": NaN, "weight": 50, "waist": 70,'
            ' "chest": 86, "hip": 92}'
        )

        response = self.client.post(
            "/generate-avatar",
            content=body,
            headers={"Content-Type": "application/json"},
        )

        self.assertEqual(response.status_code, 422)

    def test_missing_field_is_rejected(self):
        body = dict(VALID_REQUEST)
        del body["chest"]

        response = self.client.post("/generate-avatar", json=body)

        self.assertEqual(response.status_code, 422)

    def test_skin_tone(self):
        response = self.post(skin_tone="#a0705a")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            self.export_mock.call_args.kwargs["appearance"].skin_tone,
            "#a0705a",
        )
        self.assertEqual(response.json()["appearance"]["skin_tone"], "#a0705a")

        response = self.post(skin_tone="pink")

        self.assertEqual(response.status_code, 422)

    def test_eye_color_default_and_values(self):
        response = self.post()

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.export_mock.call_args.kwargs["appearance"].eye_color, "brown")
        self.assertEqual(response.json()["appearance"]["eye_color"], "brown")

        for value, expected in (("blue", "blue"), ("Green", "green"), (" HAZEL ", "hazel"), ("gray", "gray")):
            with self.subTest(value=value):
                response = self.post(eye_color=value)

                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(self.export_mock.call_args.kwargs["appearance"].eye_color, expected)

    def test_invalid_eye_color_writes_nothing(self):
        for value in ("purple", "", 5):
            with self.subTest(value=value):
                response = self.post(eye_color=value)

                self.assertEqual(response.status_code, 422)
                self.assertFalse((self.api.OUTPUT_DIR).exists() and any(self.api.OUTPUT_DIR.iterdir()))

    def test_appearance_follows_gender(self):
        female = self.post().json()["appearance"]
        male = self.post(gender="male").json()["appearance"]

        self.assertEqual(female["gender_style"], "female")
        self.assertIn("lip_tint", female["features"])
        self.assertEqual(male["gender_style"], "male")
        self.assertNotIn("lip_tint", male["features"])
        self.assertIn("hair", male["not_available"])

    def test_unknown_fields_are_ignored(self):
        response = self.post(shoulder=40, notes="x")

        self.assertEqual(response.status_code, 200)


# ============================================================
# REAL ANNY MODEL (SINGLE EVALUATIONS, NO FULL FITTING)
# ============================================================

@unittest.skipIf(
    os.environ.get("AVATAR_SKIP_MODEL_TESTS") == "1",
    "AVATAR_SKIP_MODEL_TESTS=1",
)
class RealModelTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import anny
        from measurements import BodyMeasurements

        cls.model = anny.Anny()
        cls.meas = BodyMeasurements(cls.model)

    def body(self, **params):
        values = dict(
            gender=1.0,
            age=0.5,
            height=0.5,
            weight=0.5,
            muscle=0.5,
            proportions=0.5,
        )
        values.update(params)

        output, result = ftm.evaluate(
            self.model,
            self.meas,
            values["gender"],
            values["age"],
            values["height"],
            values["weight"],
            values["muscle"],
            values["proportions"],
        )

        vertices = output["rest_vertices"][0].detach().cpu().numpy()

        return output, result, vertices

    def test_each_input_changes_the_mesh(self):
        _, _, base = self.body()

        for name, value in [
            ("gender", 0.0),
            ("age", 0.8),
            ("height", 0.8),
            ("weight", 0.8),
            ("muscle", 0.8),
            ("proportions", 0.8),
        ]:
            with self.subTest(parameter=name):
                _, _, changed = self.body(**{name: value})

                moved = np.linalg.norm(changed - base, axis=1).max()

                self.assertGreater(moved, 0.005)

    def test_gender_changes_measurements(self):
        _, female, _ = self.body(gender=1.0)
        _, male, _ = self.body(gender=0.0)

        self.assertIsNotNone(female["chest_m"])
        self.assertIsNotNone(male["chest_m"])

        differences = [
            abs(female[key] - male[key])
            for key in ["height_m", "chest_m", "hip_m", "shoulder_width_m"]
        ]

        self.assertGreater(max(differences), 0.01)

    def test_height_and_weight_parameters_are_monotonic(self):
        _, short, _ = self.body(height=0.3)
        _, tall, _ = self.body(height=0.7)

        self.assertGreater(tall["height_m"], short["height_m"])
        self.assertGreater(tall["leg_length_m"], short["leg_length_m"])

        _, light, _ = self.body(weight=0.3)
        _, heavy, _ = self.body(weight=0.7)

        self.assertGreater(heavy["mass_kg"], light["mass_kg"])
        self.assertGreater(heavy["waist_m"], light["waist_m"])

    def test_measurements_are_plausible(self):
        _, result, _ = self.body()

        expected_ranges = {
            "height_m": (1.3, 2.1),
            "mass_kg": (30.0, 120.0),
            "waist_m": (0.5, 1.3),
            "chest_m": (0.6, 1.4),
            "hip_m": (0.6, 1.4),
            "shoulder_width_m": (0.25, 0.5),
            "arm_length_m": (0.4, 0.8),
            "leg_length_m": (0.6, 1.1),
        }

        for key, (low, high) in expected_ranges.items():
            with self.subTest(key=key):
                self.assertIsNotNone(result[key])
                self.assertTrue(low < result[key] < high, result[key])

    def test_upper_arm_is_measured_around_the_arm(self):
        """Mid-upper-arm loop, including fuller bodies where the old armpit slice merged with the torso."""

        for params in [
            {},
            {"gender": 0.0},
            {"weight": 0.999, "muscle": 0.25, "age": 0.66, "height": 0.35},
            {"gender": 0.0, "weight": 0.999, "muscle": 0.72, "proportions": 0.62},
            {"gender": 0.0, "weight": 0.9, "muscle": 0.999, "proportions": 0.87, "height": 0.52},
        ]:
            with self.subTest(params=params):
                output, result, _ = self.body(**params)

                self.assertIsNotNone(result["upper_arm_m"])
                self.assertTrue(0.18 < result["upper_arm_m"] < 0.55, result["upper_arm_m"])
                self.assertLess(result["upper_arm_m"], result["chest_m"] / 2)

    def test_legs_are_measured_per_side(self):
        """
        Thigh and calf are measured on each leg with that leg's own bones.

        The rest mesh is left/right symmetric, so both sides must agree. The
        old code cut both legs with one plane from the left leg and named
        the loops by size, which gave left and right values 1-2 cm apart.
        """

        legs = {
            "thigh": ("upperleg02", "lowerleg01", (0.3, 0.9)),
            "calf": ("lowerleg01", "lowerleg02", (0.2, 0.6)),
        }

        for params in [
            {},
            {"gender": 0.0},
            {"weight": 0.999, "muscle": 0.25, "age": 0.66, "height": 0.35},
            {"gender": 0.0, "weight": 0.999, "muscle": 0.72, "proportions": 0.62},
            {"gender": 0.0, "weight": 0.9, "muscle": 0.999, "proportions": 0.87, "height": 0.52},
        ]:
            with self.subTest(params=params):
                output, result, _ = self.body(**params)

                for leg, (start, end, (low, high)) in legs.items():
                    left = result[f"{leg}_left_m"]
                    right = result[f"{leg}_right_m"]

                    self.assertIsNotNone(left, leg)
                    self.assertIsNotNone(right, leg)
                    self.assertLessEqual(abs(left - right), 1e-6, leg)

                    for side, value in ((".L", left), (".R", right)):
                        self.assertEqual(
                            value,
                            self.meas._leg_section(
                                output["rest_vertices"],
                                output["rest_bone_poses"],
                                start,
                                end,
                                side,
                            ),
                            leg + side,
                        )

                    self.assertEqual(result[f"{leg}_m"], float(np.mean([left, right])), leg)
                    self.assertTrue(low < left < high, (leg, left))

                self.assertLess(result["calf_m"], result["thigh_m"])

    def test_neck_is_measured_around_the_neck_axis(self):
        """
        The neck is measured on a plane perpendicular to the neck01 -> head
        axis at its midpoint, and the kept loop surrounds that axis. The old
        fixed-height slice cut the shoulders of taller bodies (89-141 cm).

        The loop is identified by geometry, not by matching its length: the
        mesh is left/right symmetric, so loops off the midline come in pairs
        of equal length.
        """

        from measurements import _polygon_contains_origin

        start = self.meas._bone_index["neck01"]
        end = self.meas._bone_index["head"]

        for params in [
            {},
            {"gender": 0.0},
            {"weight": 0.999, "muscle": 0.25, "age": 0.66, "height": 0.35},
            {"gender": 0.0, "weight": 0.999, "muscle": 0.72, "proportions": 0.62},
            {"gender": 0.0, "weight": 0.9, "muscle": 0.999, "proportions": 0.87, "height": 0.52},
        ]:
            with self.subTest(params=params):
                output, result, _ = self.body(**params)
                rest_vertices = output["rest_vertices"]
                rest_bone_poses = output["rest_bone_poses"]
                neck = result["neck_m"]

                self.assertIsNotNone(neck)
                self.assertTrue(0.25 < neck < 0.55, neck)

                self.assertEqual(
                    neck,
                    self.meas._neck_section(rest_vertices, rest_bone_poses),
                )

                a = self.meas._bone_origin(rest_bone_poses, start)
                b = self.meas._bone_origin(rest_bone_poses, end)
                axis = (b - a) / np.linalg.norm(b - a)
                origin = a + (b - a) * 0.5

                u = np.cross(axis, [0.0, 0.0, 1.0])

                if np.linalg.norm(u) < 1e-6:
                    u = np.cross(axis, [1.0, 0.0, 0.0])

                u /= np.linalg.norm(u)
                v = np.cross(axis, u)

                around_axis = [
                    section
                    for section in self.meas._limb_sections(
                        rest_vertices, rest_bone_poses, start, end, 0.5
                    )
                    if _polygon_contains_origin(
                        np.c_[(section["points"] - origin) @ u, (section["points"] - origin) @ v]
                    )
                ]

                self.assertEqual(len(around_axis), 1)
                self.assertEqual(around_axis[0]["circumference"], neck)

    def test_chest_matches_full_scan(self):
        """The early-stopping chest scan picks the same loop as scanning every plane."""

        from measurements import CHEST_OFFSETS, CHEST_TARGET_OFFSET

        for params in [{}, {"weight": 0.9}, {"gender": 0.0, "muscle": 0.9}]:
            with self.subTest(params=params):
                output, _, _ = self.body(**params)

                rest_vertices = output["rest_vertices"]
                context = self.meas._context(rest_vertices)

                target_z = (
                    context["waist_z"]
                    + context["height"] * CHEST_TARGET_OFFSET
                )

                candidates = []

                for offset in CHEST_OFFSETS:
                    z = context["waist_z"] + context["height"] * offset
                    loop = self.meas._chest_loop(context["mesh"], z)

                    if loop is not None:
                        loop["distance"] = abs(z - target_z)
                        candidates.append(loop)

                candidates.sort(key=lambda x: (x["distance"], -x["area"]))

                fast = self.meas._chest_section(rest_vertices)

                self.assertEqual(fast["z"], candidates[0]["z"])
                self.assertEqual(
                    fast["circumference"],
                    candidates[0]["circumference"],
                )

    def shape(self, height_m, mass_kg):
        stats = {"anny_evaluations": 0}

        _, result, _, weight = ftm.evaluate_shape(
            self.model,
            self.meas,
            1.0,
            height_m,
            mass_kg,
            0.5,
            0.5,
            0.5,
            stats=stats,
        )

        print(
            f"\n  evaluate_shape target {height_m * 100:.1f} cm / {mass_kg:.1f} kg:"
            f" got {result['height_m'] * 100:.3f} cm,"
            f" {result['mass_kg']:.3f} kg,"
            f" {stats['anny_evaluations']} Anny evaluations"
        )

        return result, weight, stats

    def test_height_and_weight_are_solved(self):
        """One shape evaluation (no optimizer) hits reachable height and weight."""

        result, _, _ = self.shape(1.70, 55.0)

        self.assertLessEqual(
            abs(result["height_m"] - 1.70),
            ftm.HEIGHT_TOLERANCE_M * 1.5,
        )
        self.assertLessEqual(
            abs(result["mass_kg"] - 55.0),
            ftm.WEIGHT_TOLERANCE_KG * 1.5,
        )

    def test_unreachable_weight_is_not_hidden(self):
        """
        With age, muscle and proportions fixed, the weight parameter alone
        cannot reach 65 kg at 170 cm (measured maximum ~59 kg). The solver
        must stop at the bound quickly and report the real mass.
        """

        result, weight, stats = self.shape(1.70, 65.0)

        self.assertGreater(weight, 0.99)
        self.assertLess(result["mass_kg"], 64.0)
        self.assertLessEqual(stats["anny_evaluations"], 12)

        report = ftm.accuracy_report({"weight": 65.0}, result)
        self.assertGreater(report["weight"]["percentage_error"], 1.0)

    def test_api_real_export_to_temporary_location(self):
        """
        Real model and real GLB export through the endpoint. The fit is
        replaced by a single body evaluation (no optimizer), and both
        output locations point to a temporary folder.
        """

        import trimesh
        from fastapi.testclient import TestClient

        import api

        glb_hash = file_hash(PROJECT_GLB)

        output, result, _ = self.body()

        def one_evaluation_fit(**kwargs):
            return {
                "output": output,
                "result": result,
                "accuracy": ftm.accuracy_report(
                    {
                        "height": kwargs["height_cm"],
                        "weight": kwargs["weight_kg"],
                        "waist": kwargs["waist_cm"],
                        "chest": kwargs["chest_cm"],
                        "hip": kwargs["hip_cm"],
                    },
                    result,
                ),
                "parameters": {
                    "gender": kwargs["gender"],
                    "height": 0.5,
                    "weight": 0.5,
                    "age": 0.5,
                    "muscle": 0.5,
                    "proportions": 0.5,
                },
            }

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)

            with mock.patch.object(
                api, "get_model", return_value=(self.model, self.meas)
            ), mock.patch.object(
                api, "fit_body", side_effect=one_evaluation_fit
            ), mock.patch.object(
                api, "AVATAR_PATH", folder / "generated_avatar.glb"
            ), mock.patch.object(
                api, "OUTPUT_DIR", folder / "generated"
            ), mock.patch.object(
                api.generated_files,
                "all_directories",
                [str(folder / "generated")],
            ):
                client = TestClient(api.app)

                response = client.post(
                    "/generate-avatar",
                    json={**VALID_REQUEST, "skin_tone": "#a0705a"},
                )

                self.assertEqual(response.status_code, 200, response.text)

                data = response.json()

                unique = folder / "generated" / f"{data['avatar_id']}.glb"
                legacy = folder / "generated_avatar.glb"

                self.assertTrue(unique.exists())
                self.assertFalse(legacy.exists())

                served = client.get(data["avatar_url"])

                self.assertEqual(served.status_code, 200)
                self.assertEqual(served.content, unique.read_bytes())

                # One mesh, two primitives (body, brow band) sharing one vertex list.
                content = unique.read_bytes()
                length = int.from_bytes(content[12:16], "little")
                gltf = json.loads(content[20:20 + length])
                primitives = gltf["meshes"][0]["primitives"]
                position = gltf["accessors"][primitives[0]["attributes"]["POSITION"]]

                self.assertEqual(len(gltf["meshes"]), 1)
                self.assertEqual(len(primitives), 2)
                self.assertEqual(primitives[0]["attributes"], primitives[1]["attributes"])
                self.assertEqual(position["count"], output["rest_vertices"].shape[1])
                self.assertEqual(data["appearance"]["skin_tone"], "#a0705a")

                scene = trimesh.load(unique)
                self.assertGreater(len(scene.geometry), 0)

        self.assertEqual(file_hash(PROJECT_GLB), glb_hash)

    def test_export_to_temporary_file(self):
        import trimesh

        from generate_avatar import export_avatar

        glb_hash = file_hash(PROJECT_GLB)

        output, _, _ = self.body()

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "avatar.glb"

            export_avatar(output, self.model, path, skin_tone="#a0705a")

            self.assertTrue(path.exists())
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), ["avatar.glb"])

            scene = trimesh.load(path)
            mesh = list(scene.geometry.values())[0]

            self.assertEqual(
                len(mesh.vertices),
                output["rest_vertices"].shape[1],
            )
            self.assertAlmostEqual(float(mesh.bounds[0][1]), 0.0, places=5)

            material = mesh.visual.material
            color = np.asarray(material.baseColorFactor)[:3]

            # sRGB #a0705a (160, 112, 90) stored as linear, rounded to 1/255 by trimesh.
            np.testing.assert_array_equal(color, [90, 41, 26])

            # The factor as written in the GLB's JSON chunk.
            content = path.read_bytes()
            length = int.from_bytes(content[12:16], "little")
            gltf = json.loads(content[20:20 + length])
            factor = gltf["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"]

            np.testing.assert_allclose(factor, np.array([90, 41, 26, 255]) / 255, atol=1e-9)

        self.assertEqual(file_hash(PROJECT_GLB), glb_hash)


if __name__ == "__main__":
    unittest.main()

from contextlib import asynccontextmanager
from pathlib import Path
import math
import os
import shutil
import threading
import time
import uuid
from typing import Optional

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from starlette.exceptions import HTTPException

from app.config import apply_environment, settings

# Before anything imports Anny: Anny reads its cache folder at import time.
apply_environment(settings)

import fit_to_measurements as ftm
from fit_to_measurements import fit_body, normalize_gender
from generate_avatar import (
    DEFAULT_SKIN_TONE,
    _replace_file,
    build,
    parse_skin_tone,
)
from appearance import (
    DEFAULT_EYE_COLOR,
    EYE_COLORS,
    FaceModel,
    appearance_for,
    export_appearance_avatar,
    normalize_eye_color,
)


API_VERSION = 1


# ===============================
# PATHS
# ===============================

BASE_DIR = Path(__file__).resolve().parent

# Legacy, shared avatar file that every generation overwrote in the
# production backend. Off by default in the module
# (settings.write_legacy_avatar); then no file is written and the
# response's legacy_avatar_url is null.
WRITE_LEGACY_AVATAR = settings.write_legacy_avatar
AVATAR_PATH = BASE_DIR / "generated_avatar.glb"
LEGACY_AVATAR_URL = "/models/generated_avatar.glb"

# One file per generation: /models/generated/<avatar_id>.glb
OUTPUT_DIR = settings.output_dir
OUTPUT_URL_PREFIX = "/models/generated"


# ===============================
# LOAD ANNY ONCE
# ===============================

model = None
measurements = None

# Anny with the cached 63-label face controls (appearance.FaceModel). Used only
# after the fit, to build the exported body with the gender face preset.
face_model = None

_model_lock = threading.Lock()
_face_model_lock = threading.Lock()

# One generation at a time: the model is shared and every request
# also updates the legacy GLB file.
_generate_lock = threading.Lock()


def get_model():
    global model, measurements

    with _model_lock:
        if model is None:
            import anny
            from measurements import BodyMeasurements

            print("Loading Anny...")

            model = anny.Anny()
            measurements = BodyMeasurements(model)

            print("Anny loaded successfully!")

    return model, measurements


def get_face_model():
    global face_model

    with _face_model_lock:
        if face_model is None:
            print("Loading Anny face model...")

            face_model = FaceModel()

            print("Anny face model loaded successfully!")

    return face_model


@asynccontextmanager
async def lifespan(app):
    get_model()
    get_face_model()
    yield


app = FastAPI(lifespan=lifespan)


# ===============================
# CORS
# ===============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)

    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    return value


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc):
    # The default handler fails with a 500 when the rejected input
    # is NaN or Infinity, because those cannot be written as JSON.
    return JSONResponse(
        status_code=422,
        content={"detail": _json_safe(jsonable_encoder(exc.errors()))},
    )


# ===============================
# REQUEST MODEL
# ===============================

# name -> (minimum, maximum, unit); used by validation and /avatar-options.
MEASUREMENT_LIMITS = {
    "height": (100.0, 230.0, "cm"),
    "weight": (25.0, 250.0, "kg"),
    "waist": (40.0, 200.0, "cm"),
    "chest": (50.0, 200.0, "cm"),
    "hip": (50.0, 200.0, "cm"),
}

# Optional, male bodies only: shoulder breadth from shoulder tip to shoulder
# tip, straight across the back (measurements.shoulder_breadth).
OPTIONAL_MEASUREMENT_LIMITS = {
    "shoulder": (30.0, 60.0, "cm"),
}


def _measurement_field(name):
    low, high, _ = MEASUREMENT_LIMITS[name]

    return Field(ge=low, le=high, allow_inf_nan=False)


def _optional_measurement_field(name):
    low, high, _ = OPTIONAL_MEASUREMENT_LIMITS[name]

    return Field(default=None, ge=low, le=high, allow_inf_nan=False)


class AvatarRequest(BaseModel):
    gender: str = "female"
    height: float = _measurement_field("height")
    weight: float = _measurement_field("weight")
    waist: float = _measurement_field("waist")
    chest: float = _measurement_field("chest")
    hip: float = _measurement_field("hip")
    shoulder: Optional[float] = _optional_measurement_field("shoulder")
    skin_tone: Optional[str] = None
    eye_color: Optional[str] = None

    @field_validator("gender")
    @classmethod
    def check_gender(cls, value):
        return normalize_gender(value)

    @field_validator("skin_tone")
    @classmethod
    def check_skin_tone(cls, value):
        if value is not None:
            parse_skin_tone(value)

        return value

    @field_validator("eye_color")
    @classmethod
    def check_eye_color(cls, value):
        if value is not None:
            return normalize_eye_color(value)

        return value


# ===============================
# STATIC FILES
# ===============================

class FilteredStaticFiles(StaticFiles):
    """
    StaticFiles limited to files with the given suffixes, directly in
    the served directory. Subfolders, hidden files (such as the
    temporary file written during export) and everything else return 404.
    """

    def __init__(self, *args, suffixes, **kwargs):
        super().__init__(*args, **kwargs)
        self.suffixes = {s.lower() for s in suffixes}

    async def get_response(self, path, scope):
        name = os.path.basename(path)

        if (
            path != name
            or name.startswith(".")
            or os.path.splitext(name)[1].lower() not in self.suffixes
        ):
            raise HTTPException(status_code=404)

        return await super().get_response(path, scope)

    async def check_config(self):
        # The generated/ folder only exists after the first generation;
        # until then requests should get 404, not a configuration error.
        if self.directory is not None and not os.path.isdir(self.directory):
            return

        await super().check_config()


# Mounted before /models so that /models/generated/... resolves here.
generated_files = FilteredStaticFiles(
    directory=OUTPUT_DIR,
    check_dir=False,
    suffixes={".glb"},
)

app.mount(
    OUTPUT_URL_PREFIX,
    generated_files,
    name="generated",
)

app.mount(
    "/models",
    FilteredStaticFiles(directory=BASE_DIR, suffixes={".glb"}),
    name="models",
)

app.mount(
    "/test",
    FilteredStaticFiles(directory=BASE_DIR, html=True, suffixes={".html"}),
    name="test",
)


# ===============================
# ROOT, HEALTH, OPTIONS
# ===============================

@app.get("/")
def root():
    return {
        "message": "Avatar API is running"
    }


@app.get("/health")
def health():
    # Does not load the model.
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.get("/avatar-options")
def avatar_options():
    """Inputs that /generate-avatar accepts and that change the avatar."""

    return {
        "api_version": API_VERSION,
        "measurements": {
            name: {
                "required": True,
                "minimum": low,
                "maximum": high,
                "unit": unit,
            }
            for name, (low, high, unit) in MEASUREMENT_LIMITS.items()
        },
        "optional_measurements": {
            name: {
                "required": False,
                "minimum": low,
                "maximum": high,
                "unit": unit,
                "genders": ["male"],
            }
            for name, (low, high, unit) in OPTIONAL_MEASUREMENT_LIMITS.items()
        },
        "gender": {
            "required": False,
            "values": sorted(ftm.GENDER_VALUES),
            "default": "female",
            "case_insensitive": True,
        },
        "skin_tone": {
            "required": False,
            "format": "#RRGGBB",
            "default": DEFAULT_SKIN_TONE,
        },
        "eye_color": {
            "required": False,
            "values": list(EYE_COLORS),
            "default": DEFAULT_EYE_COLOR,
            "case_insensitive": True,
        },
        "appearance": {
            "automatic": [
                "gender face preset (female: narrower mouth, softer cheeks; male: stronger jaw and chin)",
                "eye texture in the selected eye colour",
                "eyebrows (female and male designs)",
                "lip tint (female only)",
            ],
            "not_available": ["hair"],
        },
        "notes": [
            "Fields not listed here are ignored and do not change the avatar.",
            "shoulder: male bodies only, shoulder tip to shoulder tip straight across the back."
            " Without it the shoulder breadth follows from the other measurements."
            " Ignored (with a warning) for female bodies.",
        ],
    }


# ===============================
# HELPERS
# ===============================

def _json_number(value):
    if value is None:
        return None

    value = float(value)

    return value if math.isfinite(value) else None


def _copy_file_atomic(source, destination):
    destination = Path(destination)
    temp_path = destination.with_name(
        f".{destination.stem}.tmp{destination.suffix}"
    )

    try:
        shutil.copyfile(source, temp_path)
        _replace_file(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


# Tolerances already defined by the fitting code, in requested units.
def _tolerance(name):
    if name == "height":
        return ftm.HEIGHT_TOLERANCE_M * 100.0, "absolute"

    if name == "weight":
        return ftm.WEIGHT_TOLERANCE_KG, "absolute"

    return ftm.GOOD_ENOUGH_RELATIVE_ERROR * 100.0, "percent"


def accuracy_warnings(accuracy):
    """Human-readable notes for targets that were missed or not measured."""

    warnings = []

    for name, entry in accuracy.items():
        if entry.get("status") != "measured":
            warnings.append(
                f"{name}: could not be measured on the generated body."
            )
            continue

        if entry.get("absolute_error") is None:
            continue

        limit, kind = _tolerance(name)

        if kind == "absolute":
            missed = entry["absolute_error"] > limit
        else:
            missed = (
                entry["percentage_error"] is not None
                and entry["percentage_error"] > limit
            )

        if missed:
            unit = entry["unit"]
            pct = entry["percentage_error"]

            warnings.append(
                f"{name}: {entry['actual']:.2f} {unit} vs requested"
                f" {entry['requested']:.2f} {unit}"
                + (f" ({pct:.2f}% off)" if pct is not None else "")
                + ", outside the fitting tolerance."
            )

    return warnings


# ===============================
# GENERATE AVATAR
# ===============================

@app.post("/generate-avatar")
def generate_avatar(data: AvatarRequest):

    model, measurements = get_model()

    avatar_id = uuid.uuid4().hex
    avatar_path = OUTPUT_DIR / f"{avatar_id}.glb"

    warnings = []

    with _generate_lock:
        print()
        print("Generating personalized avatar...")

        # ---------------------------
        # 1. Fit body measurements
        # ---------------------------

        start_time = time.perf_counter()

        face = get_face_model()

        male = data.gender == "male"

        if data.shoulder is not None and not male:
            warnings.append(
                "shoulder: used for male avatars only; ignored for this female avatar."
            )

        fitted = fit_body(
            model=model,
            measurements=measurements,
            gender=data.gender,
            height_cm=data.height,
            weight_kg=data.weight,
            waist_cm=data.waist,
            chest_cm=data.chest,
            hip_cm=data.hip,
            # Male bodies: adult fit with the face model's body controls
            # (fit_to_measurements.fit_adult_male).
            shoulder_cm=data.shoulder if male else None,
            shape_model=face.controls if male else None,
        )

        fit_time = time.perf_counter() - start_time

        print(f"Fit time: {fit_time:.2f} seconds")

        params = fitted["parameters"]
        body_controls = fitted.get("body_controls") or {}

        # ---------------------------
        # 2. Build avatar with the appearance
        # ---------------------------

        # The fitted phenotype (and, for male bodies, the fitted body
        # controls) is evaluated once more with the face model and the
        # gender face preset (head vertices only).
        appearance = appearance_for(
            gender=params["gender"],
            skin_tone=data.skin_tone or DEFAULT_SKIN_TONE,
            eye_color=data.eye_color or DEFAULT_EYE_COLOR,
        )

        output = build(
            model=face.controls({**appearance.face_controls, **body_controls}),
            gender=1.0 if params["gender"] == "female" else 0.0,
            height=params["height"],
            weight=params["weight"],
            age=params["age"],
            muscle=params["muscle"],
            proportions=params["proportions"],
        )

        # Male bodies: report the measurements of the body that is exported
        # (with the face preset), not of the fitting body.
        if fitted.get("fit_method") == "adult_male":
            targets = {"height": data.height, "weight": data.weight, "waist": data.waist,
                       "chest": data.chest, "hip": data.hip}

            if data.shoulder is not None:
                targets["shoulder"] = data.shoulder

            result, accuracy, out_of_range = ftm.report_male_body(
                output, measurements, targets, fitted.get("out_of_range", [])
            )
            fitted = dict(fitted, result=result, accuracy=accuracy, out_of_range=out_of_range)

        # ---------------------------
        # 3. Export GLB
        # ---------------------------

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        export_appearance_avatar(
            output=output,
            face=face,
            path=avatar_path,
            appearance=appearance,
        )

        # Only when the legacy fixed URL is turned on.
        if WRITE_LEGACY_AVATAR:
            try:
                _copy_file_atomic(avatar_path, AVATAR_PATH)
            except OSError as exc:
                warnings.append(
                    f"legacy avatar file was not updated: {exc.__class__.__name__}"
                )

        print("Avatar generated successfully!")

    # ---------------------------
    # 4. Return response
    # ---------------------------

    result = fitted["result"]
    accuracy = fitted.get("accuracy", {})

    return {
        "message": "Avatar generated successfully",

        # Unique per generation.
        "avatar_url": f"{OUTPUT_URL_PREFIX}/{avatar_id}.glb",

        "avatar_id": avatar_id,

        # Shared file overwritten by every generation; null when turned off.
        "legacy_avatar_url": LEGACY_AVATAR_URL if WRITE_LEGACY_AVATAR else None,

        "parameters": params,

        # "adult_male" (fit_to_measurements.fit_adult_male) or "standard".
        "fit_method": fitted.get("fit_method", "standard"),

        # Anny body controls (-1..1) applied on top of the phenotype; {} for "standard".
        "body_controls": body_controls,

        # Male bodies: requested measurements beyond what Anny's controls can
        # reach (see accuracy for the values reached). They are reported, not
        # forced at the cost of other body parts.
        "out_of_range": fitted.get("out_of_range", []),

        # Appearance actually applied to this avatar.
        "appearance": appearance.summary(),

        "measurements": {
            key: float(value)
            for key, value in result.items()
            if value is not None and key != "hip_z"
        },

        # Measurements the pipeline could not produce for this body.
        "failed_measurements": sorted(
            key
            for key, value in result.items()
            if value is None and key != "hip_z"
        ),

        # Requested vs measured, for height, weight, waist, chest, hip.
        "accuracy": accuracy,

        "warnings": accuracy_warnings(accuracy) + warnings,

        "fit_time_seconds": _json_number(fit_time),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)

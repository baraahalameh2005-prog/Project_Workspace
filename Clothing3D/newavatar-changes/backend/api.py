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
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.exceptions import HTTPException

from app.config import apply_environment, settings

# Before anything imports Anny: Anny reads its cache folder at import time.
apply_environment(settings)

import fit_to_measurements as ftm
from fit_to_measurements import normalize_gender
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
from body_style import (
    ATHLETIC_MAX_WEIGHT_ERROR_KG,
    BODY_STYLES,
    DEFAULT_BODY_STYLE,
    STANDARD,
    FitError,
    fit_athletic,
    fit_body,
    normalize_body_style,
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
# Measurements on the face model's body (athletic body style); made on first use.
face_measurements = None

_model_lock = threading.Lock()
_face_model_lock = threading.Lock()

# One generation at a time: the model is shared and every request
# also updates the legacy GLB file.
_generate_lock = threading.Lock()


def get_model():
    """The model the body is fitted on: the face model's Anny (with every
    control at 0 its body equals the plain model's), so only one Anny model
    is loaded."""
    global model, measurements

    face = get_face_model()

    with _model_lock:
        if model is None:
            from measurements import BodyMeasurements

            model = face.model
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


def get_face_measurements():
    return get_model()[1]


@asynccontextmanager
async def lifespan(app):
    get_model()
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


def _measurement_field(name):
    low, high, _ = MEASUREMENT_LIMITS[name]

    return Field(ge=low, le=high, allow_inf_nan=False)


class AvatarRequest(BaseModel):
    gender: str = "female"
    height: float = _measurement_field("height")
    weight: float = _measurement_field("weight")
    waist: float = _measurement_field("waist")
    chest: float = _measurement_field("chest")
    hip: float = _measurement_field("hip")
    skin_tone: Optional[str] = None
    eye_color: Optional[str] = None
    body_style: Optional[str] = DEFAULT_BODY_STYLE

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

    @field_validator("body_style")
    @classmethod
    def check_body_style(cls, value):
        return normalize_body_style(value) if value is not None else DEFAULT_BODY_STYLE

    @model_validator(mode="after")
    def check_body_style_gender(self):
        if self.body_style == "athletic" and self.gender != "female":
            raise ValueError("body_style 'athletic' is available for gender 'female' only.")

        return self


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
        "body_style": {
            "required": False,
            "values": list(BODY_STYLES),
            "default": DEFAULT_BODY_STYLE,
            "case_insensitive": True,
            "notes": "athletic: adult, toned female body (female only); the bust is part "
                     "of the shape, so the chest can come out larger than requested.",
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

    # The fit's stopping tolerance for circumferences (body_style.STANDARD).
    return STANDARD["female"].good_enough * 100.0, "percent"


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

        body_style = data.body_style
        fitted = None

        if body_style == "athletic":
            try:
                fitted = fit_athletic(
                    model=get_face_model().model,
                    measurements=get_face_measurements(),
                    height_cm=data.height,
                    weight_kg=data.weight,
                    waist_cm=data.waist,
                    chest_cm=data.chest,
                    hip_cm=data.hip,
                )
                weight_error = fitted["accuracy"]["weight"]["absolute_error"]
                if weight_error is None or weight_error > ATHLETIC_MAX_WEIGHT_ERROR_KG:
                    raise ValueError(f"weight missed by {weight_error} kg")
            except (RuntimeError, ValueError) as exc:
                print(f"Athletic body not used: {exc}")
                warnings.append(
                    "body_style: the athletic body cannot reach these measurements "
                    "(usually the weight), so the standard body was used."
                )
                body_style = "standard"
                fitted = None

        if fitted is None:
            try:
                fitted = fit_body(
                    model=model,
                    measurements=measurements,
                    gender=data.gender,
                    height_cm=data.height,
                    weight_kg=data.weight,
                    waist_cm=data.waist,
                    chest_cm=data.chest,
                    hip_cm=data.hip,
                )
            except FitError as exc:
                return JSONResponse(
                    status_code=422,
                    content={"detail": [{"loc": ["body"], "msg": f"{exc} Check that the measurements "
                                                                "belong to one adult body.", "type": "fit_failed"}]},
                )

        # Anny's body model has a weight range per height and age; say so
        # plainly when the fitted weight is at its limit.
        if fitted["parameters"]["weight"] >= ftm.PARAM_MAX - 1e-3 and (
            (fitted["accuracy"]["weight"].get("absolute_error") or 0) > 1.0
        ):
            warnings.append(
                "weight: this weight is beyond what Anny's body model can reach at this height; "
                "the avatar is as heavy as the model allows."
            )

        fit_time = time.perf_counter() - start_time

        print(f"Fit time: {fit_time:.2f} seconds")

        params = fitted["parameters"]

        # ---------------------------
        # 2. Build avatar with the appearance
        # ---------------------------

        # The fit is unchanged; the fitted phenotype is evaluated once more
        # with the face model and the gender face preset (head vertices only).
        appearance = appearance_for(
            gender=params["gender"],
            skin_tone=data.skin_tone or DEFAULT_SKIN_TONE,
            eye_color=data.eye_color or DEFAULT_EYE_COLOR,
        )

        face = get_face_model()

        output = build(
            model=face.controls(
                appearance.face_controls,
                body_controls=fitted.get("body_controls"),
                phenotypes=fitted.get("body_phenotype"),
            ),
            gender=1.0 if params["gender"] == "female" else 0.0,
            height=params["height"],
            weight=params["weight"],
            age=params["age"],
            muscle=params["muscle"],
            proportions=params["proportions"],
        )

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

        # "standard" (production fit) or "athletic", and the body shape used.
        "body_style": body_style,
        "body_controls": fitted.get("body_controls", {}),
        "body_phenotype": fitted.get("body_phenotype", {}),

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

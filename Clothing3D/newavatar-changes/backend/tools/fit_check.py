"""
Runs the six recorded baseline inputs (tools/baseline_cases.json) through the
current fit (body_style: standard, plus athletic for the female cases;
"old" adds the earlier fit_to_measurements.fit_body) and writes
errors, fitted parameters, failed measurements and times to
var/fit_check/<label>_<timestamp>.json. Used to compare the code before and
after a change. Does not write GLBs.

Run from the backend folder:
    ..\\.venv\\Scripts\\python.exe -B tools\\fit_check.py <label>
"""

import json
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent))

import fit_to_measurements as ftm  # applies the module cache settings first
from app import config

import anny
from appearance import FACE_MODEL_LABELS
import body_style
from measurements import BodyMeasurements

CASES = json.loads((TOOLS / "baseline_cases.json").read_text())["cases"]
LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"
MODES = set(sys.argv[2:])  # "old": also run fit_to_measurements.fit_body (the earlier fit)
OUT = config.MODULE_ROOT / "var" / "fit_check" / f"{LABEL}_{time.strftime('%Y%m%d_%H%M%S')}.json"


def summarize(case, style, fitted, seconds):
    acc = fitted["accuracy"]
    return {
        "case": case, "style": style, "seconds": round(seconds, 1),
        "errors": {k: (round(v["absolute_error"], 2) if v["absolute_error"] is not None else None)
                   for k, v in acc.items()},
        "percent": {k: (round(v["percentage_error"], 2) if v["percentage_error"] is not None else None)
                    for k, v in acc.items()},
        "failed": sorted(k for k, v in fitted["result"].items() if v is None and k != "hip_z"),
        "parameters": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in fitted["parameters"].items()},
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    face = anny.Anny(local_changes=FACE_MODEL_LABELS, phenotypes="all")
    face_measurements = BodyMeasurements(face)
    plain = anny.Anny()
    plain_measurements = BodyMeasurements(plain)
    rows = []
    for case in CASES:
        inp = case["input"]
        args = dict(height_cm=inp["height_cm"], weight_kg=inp["weight_kg"], waist_cm=inp["waist_cm"],
                    chest_cm=inp["chest_cm"], hip_cm=inp["hip_cm"])
        if "old" in MODES:
            t = time.perf_counter()
            fitted = ftm.fit_body(plain, plain_measurements, inp["gender"], **args)
            rows.append(summarize(case["case"], "old", fitted, time.perf_counter() - t))
            print(json.dumps(rows[-1]), flush=True)
        t = time.perf_counter()
        fitted = body_style.fit_body(face, face_measurements, inp["gender"], **args)
        rows.append(summarize(case["case"], "standard", fitted, time.perf_counter() - t))
        print(json.dumps(rows[-1]), flush=True)
        if inp["gender"] == "female":
            t = time.perf_counter()
            fitted = body_style.fit_athletic(face, face_measurements, **args)
            rows.append(summarize(case["case"], "athletic", fitted, time.perf_counter() - t))
            print(json.dumps(rows[-1]), flush=True)
    OUT.write_text(json.dumps(rows, indent=1))
    print("OUT", OUT)


if __name__ == "__main__":
    main()

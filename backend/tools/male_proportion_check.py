"""
Male body proportions against the running API: does each measurement move the
fitted body in the requested direction, and what else moves with it?

Black-box check (standard library only). It does not load Anny itself, so it
needs no extra memory; it calls POST /generate-avatar on a running server
(each call writes one new GLB to the server's output folder, nothing is
overwritten) and reads the exported GLB's bounding box.

Run from the backend folder, with the API already running:
    python -B tools\\male_proportion_check.py [--api http://127.0.0.1:8002] [--strict]

Exit code 1 when a direction check fails. Known limitations of the current fit
are printed as KNOWN ISSUE; with --strict they fail too.
Writes var/male_proportion_check/<timestamp>.json.
"""

import argparse
import json
import struct
import sys
import time
import urllib.request
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
OUT = BACKEND.parent / "var" / "male_proportion_check"

# Anny's young-adult age anchor; below it the body blends in child proportions.
ADULT_AGE = 2.0 / 3.0

BASE = {"gender": "male", "height": 178, "weight": 78, "waist": 86, "chest": 100, "hip": 98}
# case -> (changed field, new value, measurement key that must grow)
CASES = {
    "height+10": ("height", 188, "height_m"),
    "chest+8": ("chest", 108, "chest_m"),
    "waist+8": ("waist", 94, "waist_m"),
    "hip+8": ("hip", 106, "hip_m"),
}
# A requested change should reach at least this share of the step.
MIN_REACHED_SHARE = 0.5
# Shoulder width moving more than this when no length was changed is a side effect.
SHOULDER_SIDE_EFFECT_CM = 1.5


def generate(api, body):
    request = urllib.request.Request(
        f"{api}/generate-avatar", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def glb_extent(api, avatar_url):
    """(height, width) in metres of the exported GLB, from its POSITION accessors."""
    with urllib.request.urlopen(f"{api}{avatar_url}", timeout=60) as response:
        data = response.read()
    magic, _, _ = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError("not a GLB file")
    length, _ = struct.unpack_from("<II", data, 12)  # first chunk is the JSON
    gltf = json.loads(data[20 : 20 + length])
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for mesh in gltf["meshes"]:
        for primitive in mesh["primitives"]:
            accessor = gltf["accessors"][primitive["attributes"]["POSITION"]]
            lo = [min(a, b) for a, b in zip(lo, accessor["min"])]
            hi = [max(a, b) for a, b in zip(hi, accessor["max"])]
    return hi[1] - lo[1], hi[0] - lo[0]


def summary(body, response, extent):
    m = response["measurements"]
    return {
        "request": body,
        "avatar_id": response["avatar_id"],
        "parameters": response["parameters"],
        "cm": {
            key: round(m[key] * 100, 2)
            for key in (
                "height_m", "chest_m", "waist_m", "hip_m",
                "shoulder_width_m", "arm_length_m", "leg_length_m",
            )
            if m.get(key) is not None
        },
        "mass_kg": round(m["mass_kg"], 2),
        "glb_height_cm": round(extent[0] * 100, 2),
        "glb_width_cm": round(extent[1] * 100, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8002")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    api = args.api.rstrip("/")

    rows = {}
    for name, (field, value, _) in {"base": (None, None, None), **CASES}.items():
        body = dict(BASE) if field is None else {**BASE, field: value}
        start = time.perf_counter()
        response = generate(api, body)
        rows[name] = summary(body, response, glb_extent(api, response["avatar_url"]))
        print(f"{name:<10} {time.perf_counter() - start:5.1f} s  {json.dumps(rows[name]['cm'])}"
              f"  age {rows[name]['parameters']['age']:.3f}", flush=True)

    failures, known = [], []
    base = rows["base"]
    for name, (field, value, key) in CASES.items():
        row = rows[name]
        step = (value - BASE[field])
        moved = row["cm"][key] - base["cm"][key]
        if moved <= 0:
            failures.append(f"{name}: {key} did not grow ({moved:+.2f} cm for a {step:+} cm request)")
        elif moved < MIN_REACHED_SHARE * step:
            known.append(f"{name}: {key} grew only {moved:+.2f} cm of the requested {step:+} cm")
        if field == "height":
            glb_moved = row["glb_height_cm"] - base["glb_height_cm"]
            if abs(glb_moved - step) > 1.5:
                failures.append(f"{name}: GLB height moved {glb_moved:+.2f} cm, expected about {step:+} cm")
            if row["cm"]["leg_length_m"] <= base["cm"]["leg_length_m"]:
                failures.append(f"{name}: leg length did not grow")
        else:
            shoulder = row["cm"]["shoulder_width_m"] - base["cm"]["shoulder_width_m"]
            if abs(shoulder) > SHOULDER_SIDE_EFFECT_CM:
                known.append(f"{name}: shoulder width moved {shoulder:+.2f} cm although only {field} was changed")

    for name, row in rows.items():
        if row["parameters"]["age"] < ADULT_AGE:
            share = min(1.0, (ADULT_AGE - row["parameters"]["age"]) / (ADULT_AGE - 1.0 / 3.0))
            known.append(f"{name}: fitted age {row['parameters']['age']:.3f} < {ADULT_AGE:.3f}"
                         f" ({share:.0%} child proportions)")

    print()
    for line in failures:
        print("FAIL        ", line)
    for line in known:
        print("KNOWN ISSUE ", line)
    if not failures and not known:
        print("all checks passed")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({"rows": rows, "failures": failures, "known_issues": known}, indent=1))
    print(f"\nwrote {path}")
    return 1 if failures or (args.strict and known) else 0


if __name__ == "__main__":
    sys.exit(main())

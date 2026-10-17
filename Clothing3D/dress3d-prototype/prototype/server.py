"""Local prototype server (stdlib only). Binds to 127.0.0.1 — photos never leave this machine.

  python prototype/server.py            -> http://127.0.0.1:8765

Endpoints
  GET  /                      built web UI (prototype/web/dist)
  GET  /api/schema            measurement definitions
  POST /api/generate          JSON {front_image?: dataURL, back_image?: dataURL, units, measurements, style, options}
  GET  /api/out/<id>/<file>   generated files (dress.glb, pattern.svg, mask.png, report.json)
"""
from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import sys
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dress3d.schema import MEASUREMENT_DEFS, SILHOUETTES, SLEEVES, NECKLINES, parse_spec  # noqa: E402
from dress3d.generator import DressGenerator  # noqa: E402
from dress3d.hybrid import reconstruct  # noqa: E402
from dress3d import export as ex  # noqa: E402
from dress3d.measure import measure_dress, compare  # noqa: E402

DIST = os.path.join(ROOT, "prototype", "web", "dist")
OUTDIR = os.path.join(ROOT, "prototype", "output")
EXAMPLES = os.path.join(ROOT, "experiments", "inputs", "synthetic")
os.makedirs(OUTDIR, exist_ok=True)
MAX_BODY = 40 * 2 ** 20


def decode_image(data_url):
    if not data_url:
        return None
    b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    # cap resolution for speed; aspect preserved
    img.thumbnail((900, 1200))
    return np.asarray(img)


def generate(req):
    units = req.get("units", "cm")
    meas = {k: v for k, v in (req.get("measurements") or {}).items() if v not in (None, "", {})}
    meas = {k: v for k, v in meas.items() if not (isinstance(v, dict) and v.get("value") in (None, ""))}
    style = {k: v for k, v in (req.get("style") or {}).items() if v and v != "auto"}
    options = {k: float(v) for k, v in (req.get("options") or {}).items() if v not in (None, "")}
    front = decode_image(req.get("front_image"))
    back = decode_image(req.get("back_image"))
    mask = None
    if req.get("mask_image") and front is not None:
        mimg = decode_image(req["mask_image"])
        mask = np.asarray(Image.fromarray(mimg).convert("L").resize((front.shape[1], front.shape[0]), Image.NEAREST)) > 127
        if mask.sum() < 500:
            raise ValueError("mask image has (almost) no white garment pixels")
    rid = uuid.uuid4().hex[:12]
    od = os.path.join(OUTDIR, rid)
    os.makedirs(od)
    warnings = []
    if front is not None:
        dm, tex, uv, report, pa = reconstruct(front, meas, units=units, style_override=style, back_img=back,
                                              options=options, colour_refine=bool(req.get("colour_refine")), mask=mask)
        Image.fromarray((pa.mask * 255).astype(np.uint8)).save(os.path.join(od, "mask.png"))
        m = pa.mask
        if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
            warnings.append("Garment touches the photo border - it may be cropped; hem/sleeve cues unreliable.")
        if back is None:
            warnings.append("No back photo: back appearance is ASSUMED to match the front.")
        rejected = [k for k, v in report["source"].items() if "rejected" in str(v)]
        if rejected:
            warnings.append("Implausible photo cues replaced by defaults: " + ", ".join(rejected)
                            + ". Check the segmentation; consider supplying a mask or measuring these values.")
        method = "hybrid: classical photo analysis + measurement-driven pattern construction"
        spec_m = report["measurements_cm"]
        mesh = ex.to_trimesh(dm, tex, uv=uv)
    else:
        spec = parse_spec({"units": units, "style": {"silhouette": "a-line", **style}, "measurements": meas,
                           "options": options})
        dm = DressGenerator(spec).build()
        tex = ex.make_print_texture(dm.atlas_size_cm, kind="plain", base=(150, 40, 70))
        mesh = ex.to_trimesh(dm, tex)
        report = {"style": spec.style, "measurements_cm": spec.m,
                  "source": {k: ("user" if k in spec.provided else "default") for k in spec.m}, "notes": spec.notes}
        spec_m = spec.m
        method = "procedural: measurement-driven pattern construction (no photo)"
        warnings.append("No photo: plain placeholder fabric colour.")
    glb = os.path.join(od, "dress.glb")
    size = ex.export_glb(mesh, glb)
    ex.save_landmarks(dm, os.path.join(od, "dress.landmarks.json"))
    ex.pattern_svg(dm, os.path.join(od, "pattern.svg"))
    measured, how = measure_dress(glb, os.path.join(od, "dress.landmarks.json"), hip_position=spec_m["hip_position"])
    rows = compare(measured, spec_m, [k for k in MEASUREMENT_DEFS if k in spec_m and k in measured])
    for r in rows:
        r["source"] = report["source"].get(r["measurement"], "")
        r["method"] = how.get(r["measurement"], "")
    if "hip" in meas and "hip_position" not in meas:
        warnings.append("Hip girth given without hip_position: the girth level is assumed (waist + 20 cm). "
                        "On flared skirts this can shift the hip by several cm - please measure hip_position.")
    out = {"id": rid, "method": method, "glb_url": f"/api/out/{rid}/dress.glb", "svg_url": f"/api/out/{rid}/pattern.svg",
           "mask_url": f"/api/out/{rid}/mask.png" if front is not None else None, "glb_bytes": size,
           "vertices": int(len(dm.vertices)), "faces": int(len(dm.faces)),
           "style": report["style"], "source": report["source"], "measurements_cm": spec_m,
           "verification": rows, "landmarks": dm.landmarks, "warnings": warnings,
           "notes": report.get("notes", []), "confidence": report.get("confidence", {})}
    json.dump(out, open(os.path.join(od, "report.json"), "w"), indent=2, default=float)
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, default=float).encode())

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/schema":
            return self._json(200, {"measurements": {k: {"kind": v[0], "description": v[1]} for k, v in MEASUREMENT_DEFS.items()},
                                    "silhouettes": SILHOUETTES, "sleeves": SLEEVES, "necklines": NECKLINES})
        if path == "/api/examples":
            keys = ["dress_length", "bust", "waist", "hip", "hem", "shoulder_width", "hip_position",
                    "sleeve_length", "upper_arm", "cuff"]
            out = []
            for name in sorted(os.listdir(EXAMPLES)) if os.path.isdir(EXAMPLES) else []:
                cj = os.path.join(EXAMPLES, name, "case.json")
                if os.path.isfile(cj):
                    gt = json.load(open(cj))["ground_truth"]["measurements"]
                    out.append({"name": name, "measurements": {k: gt[k] for k in keys if k in gt}})
            return self._json(200, out)
        if path.startswith("/api/examples/"):
            parts = path.split("/")
            if len(parts) != 5 or parts[4] not in ("photo_front.png", "photo_back.png") or not parts[3].replace("_", "").isalnum():
                return self._json(404, {"error": "not found"})
            fp = os.path.join(EXAMPLES, parts[3], parts[4])
        elif path.startswith("/api/out/"):
            parts = path.split("/")
            if len(parts) != 5 or not parts[3].isalnum() or parts[4] not in ("dress.glb", "pattern.svg", "mask.png", "report.json"):
                return self._json(404, {"error": "not found"})
            fp = os.path.join(OUTDIR, parts[3], parts[4])
        else:
            rel = path.lstrip("/") or "index.html"
            fp = os.path.normpath(os.path.join(DIST, rel))
            if not fp.startswith(DIST) or not os.path.isfile(fp):
                fp = os.path.join(DIST, "index.html")
        if not os.path.isfile(fp):
            return self._json(404, {"error": "not built - run npm run build in prototype/web"})
        ctype = mimetypes.guess_type(fp)[0] or "application/octet-stream"
        if fp.endswith(".glb"):
            ctype = "model/gltf-binary"
        with open(fp, "rb") as f:
            self._send(200, f.read(), ctype)

    def do_POST(self):
        if self.path != "/api/generate":
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        if n > MAX_BODY:
            return self._json(413, {"error": "request too large"})
        try:
            req = json.loads(self.rfile.read(n))
            return self._json(200, generate(req))
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        except Exception as e:  # pragma: no cover
            traceback.print_exc()
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        sys.stderr.write("[server] " + fmt % args + "\n")


if __name__ == "__main__":
    port = int(os.environ.get("DRESS3D_PORT", "8765"))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Dress3D prototype on http://127.0.0.1:{port}  (local only)")
    srv.serve_forever()

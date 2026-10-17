"""Experiment A: TripoSR (MIT) single-image-to-3D on CPU, on the same synthetic front photos.

Run with the dedicated venv:
  .venv-triposr\\Scripts\\python.exe experiments\\scripts\\run_triposr.py [--res 192]
Environment (set by this script, process-local only): HF_HOME -> <repo>/cache/hf
Input preprocessing mirrors TripoSR's run.py: RGBA foreground, resize_foreground(0.85),
composite on 0.5 grey, 512x512. The foreground mask comes from our OpenCV segmentation
(TripoSR's default rembg path is not used: its default model has a commercial-use licence).
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import json
import os
import sys
import time

os.environ["HF_HOME"] = f"{ROOT}/cache/hf"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path.insert(0, f"{ROOT}/tools/shims")
sys.path.insert(0, f"{ROOT}/tools/TripoSR")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from tsr.system import TSR  # noqa: E402
from tsr.utils import resize_foreground  # noqa: E402

RES = int(sys.argv[sys.argv.index("--res") + 1]) if "--res" in sys.argv else 256
IN = f"{ROOT}/experiments/inputs/synthetic"
OUT = f"{ROOT}/results/triposr"
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(os.cpu_count())


def mem_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:
        return float("nan")


t0 = time.time()
model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
model.renderer.set_chunk_size(8192)
model.to("cpu")
load_s = time.time() - t0
print(f"model loaded in {load_s:.1f}s, RSS {mem_mb():.0f} MB", flush=True)
log = {"load_s": load_s, "mc_resolution": RES, "torch": torch.__version__, "device": "cpu",
       "threads": torch.get_num_threads(), "cases": {}}
want = [a for a in sys.argv[1:] if a.startswith("T")]
cases = [c for c in sorted(os.listdir(IN)) if not want or c in want]
for case in cases:
    mask = np.asarray(Image.open(os.path.join(f"{ROOT}/results/hybrid_front", case, "photo_mask.png"))) > 127
    rgb = np.asarray(Image.open(os.path.join(IN, case, "photo_front.png")).convert("RGB"))
    rgba = Image.fromarray(np.dstack([rgb, (mask * 255).astype(np.uint8)]))
    image = resize_foreground(rgba, 0.85)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = Image.fromarray((image * 255.0).astype(np.uint8))
    od = os.path.join(OUT, case); os.makedirs(od, exist_ok=True)
    image.save(os.path.join(od, "input_preprocessed.png"))
    t1 = time.time()
    with torch.no_grad():
        scene_codes = model([image], device="cpu")
    t_enc = time.time() - t1
    t2 = time.time()
    meshes = model.extract_mesh(scene_codes, True, resolution=RES)
    t_mc = time.time() - t2
    mesh = meshes[0]
    mesh.export(os.path.join(od, "triposr_raw.glb"))
    mesh.export(os.path.join(od, "triposr_raw.obj"))
    info = {"encode_s": round(t_enc, 1), "extract_s": round(t_mc, 1), "rss_mb": round(mem_mb()),
            "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight), "bounds": mesh.bounds.round(4).tolist()}
    log["cases"][case] = info
    print(case, info, flush=True)
    json.dump(log, open(os.path.join(OUT, "run_log.json"), "w"), indent=2)

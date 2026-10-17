# Handover guide — continuing Dress3D

For the teammate taking over this module. It covers what works, where things live, how to verify them, what to fix
first, and what must be agreed before integrating with another clothing module. Evidence and numbers:
`FINAL_REPORT.md`.

## 1. What works today (verified)
| Capability | How to see it |
|---|---|
| Measurement schema: cm/in, full or laid-flat girths, defaults, validation | `dress3d/schema.py`; tests `SchemaTests` |
| Dress mesh built from measurements; entered girths/lengths reproduced ≤ 0.5 % on the exported GLB | `GeneratorRoundTripTests`; measurement-only mode in the UI |
| Photo cues on **synthetic** images: silhouette, sleeves, neckline, positions; front/back texture projection | `HybridSyntheticTests`; synthetic cases in the UI |
| Local web prototype: upload, form, 3D viewer, GLB/SVG export, verification table | `prototype/` |

**Not reliable:** real photos with mannequins or hangers, three-quarter views, collars, and photos whose silhouette
contradicts the measurements (`FINAL_REPORT.md` §12). **Never validated:** accuracy against a real garment measured
with a tape.

## 2. Run it (about 5 minutes)
```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v      # → "Ran 5 tests in ~14s  OK"
cd prototype\web; npm ci; npm run build; cd ..\..              # → "✓ built in …"
.venv\Scripts\python.exe prototype\server.py                   # → "Dress3D prototype on http://127.0.0.1:8765  (local only)"
```
Then, in the browser:
1. Click **T2 aline vneck** and press **Generate**.
2. Check that an A-line V-neck dress appears.
3. Check that the verification-table errors on rows marked *you* are ≤ 0.5 %.

## 3. Where things live
| Responsibility | File(s) |
|---|---|
| Frontend UI / form / results | `prototype/web/src/App.tsx`, `styles.css` |
| 3D viewer | `prototype/web/src/Viewer.tsx` (Three.js GLTFLoader + OrbitControls) |
| API client and types | `prototype/web/src/api.ts` |
| Backend HTTP API | `prototype/server.py` (`generate()` holds the whole request flow) |
| Measurement schema | `dress3d/schema.py` (`MEASUREMENT_DEFS`, `parse_spec`) |
| Geometry generation | `dress3d/generator.py` (`DressGenerator.build()` → `DressMesh`) |
| Photo analysis / segmentation | `dress3d/photo.py` (`analyse`, `segment`, `refine_garment_mask`) |
| Photo + measurements orchestration, texture | `dress3d/hybrid.py` (`reconstruct`, `project_texture`) |
| Export (GLB, landmarks, SVG) | `dress3d/export.py` |
| Verification / evaluation measurement | `dress3d/measure.py` (`measure_dress`, `compare`) |
| Experiment evaluation | `experiments/scripts/eval_lib.py`, `run_*.py`, `eval_triposr.py`, `mesh_quality.py` |
| Tests | `tests/test_core.py` |

## 4. Validated vs experimental — don't mix them up
- **Validated (tests + reported metrics):** `dress3d/schema.py`, `generator.py`, `measure.py`, `export.py`, and the
  hybrid pipeline **on synthetic inputs**.
- **Experimental / unreliable:**
  - `photo.py` on real photos.
  - `refine_garment_mask` (colour refinement; can delete fabric).
  - The neckline shading cue.
  - The sleeve-angle estimate.
  - `pattern.svg`, which is approximate and not sewable.
- **Research-only (not part of the app):** `experiments/scripts/run_triposr.py`, `eval_triposr.py`,
  `run_garmentcode.py` and `tools/shims/`. They need extra environments and third-party clones (see `README.md`).
  `debug_*.py`, `smoke_*.py` and `check_normals.py` are development checks.
- Every value in an API response carries a **source** (`user` / `photo (...)` / `default`). Only `user` values are real
  measurements. Keep that tagging if you change the pipeline.

## 5. Reproduce the important experiments
```powershell
.venv\Scripts\python.exe experiments\scripts\run_hybrid.py           # results/hybrid_front/summary.json
.venv\Scripts\python.exe experiments\scripts\run_hybrid.py --full    # results/hybrid_front_fullprotocol/
.venv\Scripts\python.exe experiments\scripts\run_real.py             # results/real_photos/ (CC0 photos, qualitative)
.venv\Scripts\python.exe experiments\scripts\mesh_quality.py         # results/measurements/mesh_quality.json
.venv\Scripts\python.exe experiments\scripts\consolidate.py          # CSV tables + screenshots
```
Expected: the numbers match `results/measurements/*.csv`. The scripts are deterministic, and the hybrid takes 3–10 s
per case. **These scripts overwrite** the files in `results/`, so commit or copy them first if you want to compare.

## 6. What to fix first
1. **Real-garment validation.** Measure 5–10 real dresses with a tape (full protocol, including `waist_position`,
   `hip_position` and `armhole_depth`) and photograph each from front, back and side on a plain background. Until then,
   make no accuracy claims for real photos.
2. **Photo vs. measurement consistency warning.** Compare the photo's hem/bust and waist/bust width ratios with the
   entered girths, and warn on disagreement. This was the cause of the "squeezed texture" in the private-photo test.
3. **Segmentation.** Add an interactive mask brush, and/or a permissively licensed garment-segmentation model (check
   the licence of both the code and the weights).
4. **Sleeve/torso interpenetration** near the armhole (`_build_sleeve` cap blend in `generator.py`).
5. **True flat patterns and drape.** Use LSCM/ARAP flattening plus a free cloth simulator. This replaces the
   decorative folds.

## 7. Before integrating with another clothing module, test
- The round-trip tests on **your** garments: girths re-measured from the GLB within your tolerance.
- **Coordinate frame and units:** the GLB is in metres, y up, the hem at y = 0, +z the front, garment centred on x/z.
  Check that your module assumes the same.
- Load the GLB in your engine. It is a double-sided PBR material with one texture and one mesh, and it is **not
  welded at seams** (UV seams). For simulation, use the welded mesh (`DressMesh.vertices/faces` in Python, or re-weld
  by position).
- Memory and latency: about 0.6 GB and 4–13 s per generation on a laptop CPU.
- Failure modes: invalid inputs give HTTP 400 `{"error": ...}`, and implausible photo cues give warnings. Check how
  your module surfaces these.

## 8. Interfaces to agree on (documentation only; nothing implemented)
| Topic | Current Dress3D behaviour | Needs agreement |
|---|---|---|
| Measurement input | JSON schema in `dress3d/schema.py` (15 fields, cm/in, flat/full, HPS-based definitions) | Shared field names and definitions with your module, and whether body or garment measurements |
| Output geometry | GLB (metres, y-up) + `*.landmarks.json` (cm) + optional `pattern.svg` | Format (GLB/OBJ/custom), welded vs UV-split, units, origin, resolution |
| Landmarks | HPS L/R, shoulder points, underarms, bust/waist/hip/hem heights, upper-arm planes, cuff centres | Which landmarks your fitting/placement needs |
| Garment metadata | `report.json`: style, per-value source, verification table, warnings | Required metadata, especially fabric properties (none exist yet) |
| Invocation | HTTP `POST /api/generate` on 127.0.0.1, or the Python call `dress3d.hybrid.reconstruct(...)` | Library call vs service, sync vs async, where outputs are stored |
| Privacy | Uploaded photos embedded in the GLB texture; stored in `prototype/output/` without cleanup | Retention, deletion, consent |
| Dependencies | Python 3.12 + NumPy/SciPy/trimesh/OpenCV/Shapely; Node for the UI | Compatible versions with your stack |

## 9. Ground rules carried over
- Use free, local, licence-checked tools only. Check licences of **both code and model weights** before adding a
  model; for example, rembg's default model and Hunyuan3D-2 have commercial or territorial restrictions.
- Don't commit private photos, `prototype/output/`, virtual environments, `node_modules`, caches, `dist` or model weights
  (`.gitignore` covers them).
- Distinguish **synthetic** results from **real-garment** results in every report.

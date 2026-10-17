# Research Log — Dress3D

All work confined to the project workspace. Package caches redirected into `cache/` (inside the workspace) per command.

## 2026-10-16 — Stage 0: Environment (VERIFIED BY EXPERIMENT — queried on this machine)

| Item | Value |
|---|---|
| OS | Windows 11 Pro 10.0.26300 |
| CPU | 12th Gen Intel Core i5-1235U (laptop, 10 cores / 12 threads) |
| RAM | 7.7 GB |
| GPU | Intel UHD Graphics (integrated). **No NVIDIA GPU, no CUDA.** |
| Free disk (D:) | 628 GB |
| Python | 3.12.6 (user install) |
| Node / npm | v24.20.0 / 11.19.0 |
| git | available |
| Blender | not installed |

**Consequence:** any method needing a CUDA GPU (most 2024–2026 image-to-3D generators, NVIDIA-Warp GPU cloth sim, nvdiffrast) cannot be executed here. Only CPU-runnable methods can be tested.

## 2026-10-16 — Stage 1: Research (summary; details in research/approach-comparison.md)
- TripoSR: MIT code+weights, ungated HF (model.ckpt 1,677,246,742 B). Only image-to-3D candidate plausibly runnable on CPU here.
- TRELLIS: MIT, needs >=16 GB NVIDIA GPU, Linux-tested -> BLOCKED (hardware).
- Hunyuan3D-2: Tencent community licence excludes EU/UK/South Korea; 6-16 GB VRAM -> BLOCKED (hardware + licence caveat).
- SF3D: gated HF (login + contact sharing required) -> EXCLUDED by rules (account).
- InstantMesh: Apache-2.0, CUDA/Linux -> BLOCKED (hardware).
- GarmentCode/pygarment: MIT; drape sim needs NvidiaWarp-GarmentCode fork = NVIDIA Source Code License (non-commercial only).
- ChatGarment: Apache-2.0 but LLaVA-scale GPU model; text scripts call GPT-4o (paid) -> BLOCKED.
- Sewformer: no licence file found; sim requires Autodesk Maya (paid) -> EXCLUDED.
- rembg: MIT, but default model now BRIA RMBG-2.0 (commercial licence required) -> use OpenCV GrabCut instead.

## 2026-10-16 — Stage 2: Core library (VERIFIED BY EXPERIMENT)
- dress3d/schema.py, generator.py, measure.py, export.py written.
- Procedural generator: A-line test, sleeveless + long sleeve. Topology: 4 boundary loops (neck, hem, 2 armholes/cuffs), consistent winding.
- Independent re-measurement of exported GLB: all errors < 0.25 % (smoke test, experiments/scripts/smoke_measure.py).
- Resume point: next = software renderer (dress3d/render.py) + synthetic test cases.

## 2026-10-16 — Stage 3: Synthetic test cases + hybrid experiment (VERIFIED BY EXPERIMENT)
- 4 synthetic GT dresses (experiments/inputs/synthetic/T1..T4): straight/sleeveless/round, A-line/V, fit-and-flare/short sleeve,
  straight/long sleeve/square. GT uses hidden construction options (depth ratio, folds, sleeve angle, positions).
- Bugs found+fixed during the experiment: inverted normals after d->y mirroring; non-conforming Delaunay at V-neck;
  boundary slivers creating seam short-cuts; side seam weld; segmentation when garment touches frame.
- Renderer now darkens interior (back-facing) surfaces (approximate self-occlusion) — neckline cue depends on it.
- Detector thresholds changed after run v1 (waist contrast >= 0.12; sleeve angle by line fit). v1 results kept in results/hybrid_front_v1.
- Results: results/hybrid_front (front photo + 6-9 user measurements), results/hybrid_frontback, results/hybrid_front_fullprotocol.
- Key finding: hip girth error up to +4.9 % when hip *position* is not provided (girth level ambiguity on flared skirts).
- Resume point: next = TripoSR CPU experiment (tools/TripoSR, .venv-triposr).

## 2026-10-16 — Stage 4: TripoSR, GarmentCode, real photos, prototype, reports (VERIFIED BY EXPERIMENT)
- TripoSR on CPU: weights 1.68 GB (MIT, ungated) into cache/hf; encode 22-34 s + extract 60-64 s per image; ~4.5 GB private RAM.
  Evaluated with best-case GT-height scaling (+ICP variant): results/triposr/.
- GarmentCode pygarment 2.0.2 installed (py3.12 OK); dress pattern generated (results/garmentcode/). Sim fork not attempted (NVSCL non-commercial + CUDA/MSVC).
- Real CC0 photos (Wikimedia Commons; Met API returned HTTP 410): R1 plaid (3/4 view), R2 ivory gown. Segmentation incl. mannequin;
  added optional GMM colour refinement + implausible-cue rejection + user mask support. Synthetic results unchanged (regression-checked).
- Prototype: prototype/server.py (127.0.0.1:8765) + prototype/web (React/TS/Three.js). Verified via API tests + headless Edge screenshots.
- Deliverables written: README.md, FINAL_REPORT.md, research/approach-comparison.md, research/sources.md, results/measurements/*.csv.
- Resume point: work complete for this phase; next steps listed in FINAL_REPORT.md §6.

## 2026-10-16 — Stage 5: Handover + publication preparation
- Audit: workspace was not a Git repo; remote https://github.com/baraahalameh2005-prog/clothes is public and empty.
- prototype/output/ contains user-generated results incl. one private real-photo run -> excluded from Git.
- Experiment scripts changed from hard-coded absolute workspace paths to repo-relative ROOT (no behaviour change).
- Added tests/test_core.py (5 unittest tests, pass), HANDOVER.md; rewrote FINAL_REPORT.md (20 sections) and README.md.

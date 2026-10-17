# Approach comparison — photo + garment measurements → independent 3D dress

Date: 2026-10-16. Machine: Intel i5-1235U laptop, 7.7 GB RAM, Intel UHD (no NVIDIA/CUDA), Windows 11.
Source IDs (S1…) refer to `research/sources.md`. Evidence labels:
**[EXP]** verified by experiment in this workspace · **[DOC]** verified from official documentation ·
**[EST]** estimated · **[NT]** not tested · **[BLK]** blocked.

## 1. Summary table

| Approach | Candidate tested | Free / licence | Runs on this PC | Uses measurements | Real 3D garment surface? | Dimensional accuracy | Status |
|---|---|---|---|---|---|---|---|
| A. Image-to-3D (generic) | **TripoSR** | MIT code+weights, ungated [DOC S1,S2] | **Yes, CPU**: 85–94 s/image (+18 s model load; first run 296 s incl. 1.68 GB download), ~4.5 GB private RAM [EXP] | No (no metric scale) | Closed solid: 0 openings, 1–6 disconnected parts [EXP] | Girths −10 … +37 % even after best-case scaling to the GT height (see §3) [EXP] | **Tested** |
| A. | TRELLIS | MIT [DOC S4] | No: needs ≥16 GB NVIDIA GPU, Linux-tested [DOC S4] | No | Mesh/GS/NeRF | — | **[BLK] hardware** |
| A. | Hunyuan3D-2 | Tencent community licence, **excludes EU/UK/South Korea**, >1M MAU needs separate licence [DOC S6] | No: 6–16 GB VRAM [DOC S5] | No | Mesh | — | **[BLK] hardware; licence caveat** |
| A. | Stable Fast 3D (SF3D) | Stability Community (free < $1M revenue) but **gated: account + contact sharing** [DOC S7] | — | No | Mesh | — | **Excluded (account required)** |
| A. | InstantMesh | Apache-2.0 repo; Zero123++ dependency licence unverified [DOC S8] | No: CUDA/Linux | No | Mesh | — | **[BLK] hardware** |
| B. Garment-specific reconstruction | ChatGarment | Apache-2.0, but LLaVA-scale model + **GPT-4o (paid)** in text pipeline [DOC S12] | No (GPU) | Indirect | Sewing pattern → sim | — | **[BLK] hardware + paid API** |
| B. | Sewformer | **No licence found**; sim via Autodesk **Maya (paid)** [DOC S13] | — | No | Pattern | — | **Excluded (licence/paid)** |
| B. | Garment3DGen, SPnet, PatternGSL | Licences/code not verifiable at access time [DOC S17,S18] | — | — | — | — | **[NT]** |
| C. Pattern-based | **GarmentCode / pygarment** | MIT [DOC S9]; drape simulator fork = **NVIDIA Source Code Licence, non-commercial only** [DOC S11] | **Pattern generation: yes** (Windows, CPU, Python 3.12) [EXP]; simulator: needs CUDA toolkit + MSVC build → not attempted | Body-relative parameters only (e.g. `shirt.width` ×1.0–1.3 of body) [EXP] | 2D sewing pattern with darts [EXP]; 3D only after simulation | Not measurable without drape | **Tested (2D only)**; 3D **[BLK]** |
| D. Procedural (own) | `dress3d.generator` | Own code (this repo) | Yes, 1–3 s/dress [EXP] | **Directly** (15-field schema) | Single-layer cloth surface with real openings, welded seams, consistent normals [EXP] | All girths/lengths reproduced ≤0.25 % from the exported GLB [EXP] | **Tested** |
| E. Hybrid (own) | `dress3d.hybrid` = photo cues + D | Own code; OpenCV/NumPy/SciPy only | Yes, 3.5–9.6 s/dress [EXP] | Directly; photo fills gaps | Same as D, photo-textured | ≤0.43 % for provided girths when their *levels* are given; up to +4.9 % hip error when hip level is not given (§3) [EXP] | **Tested — recommended** |
| F. Other | Commercial CAD (CLO3D, Marvelous Designer, Browzwear) | Paid | — | Yes (patterns) | Yes | — | **Excluded (paid)** |
| F. | Blender cloth sim (GPL, free) | GPL-3 | Likely (CPU) [EST] | Via pattern | Yes after sim | — | **[NT]** — candidate for the next phase (drape our pattern panels) |

## 2. What each approach is (and is not) good for

### A. Generic image-to-3D (TripoSR tested)
- **Strengths [EXP]:** genuinely free (MIT), no account, runs on a CPU laptop (encode 22–34 s + mesh extraction
  60–64 s at 256³ per image, model load 18 s once cached). Front silhouette resembles the photo (front IoU 0.87–0.95 with
  height-scaled bbox alignment).
- **Weaknesses [EXP]:** output is a **closed, watertight-ish blob**: neck, armholes and hem are sealed; there is no
  inside, no single-layer cloth. Depth/back are hallucinated (side IoU 0.64–0.85; flared skirts become flat slabs
  in depth). Printed stripes/dots were converted into geometric ridges/bumps (T1, T3). No metric scale — even when
  scaled to the true GT height, girths were off by −10 … +37 % (T1: +17 … +26 %). Vertex colours only; back colours
  are guesses (mean-colour ΔE 20–42 vs ≤8.4 for the hybrid). 45–57 k vertices, marching-cubes slivers (aspect p99 ≈ 70).
- **Hardware:** ~4.5 GB private memory → heavy paging on a 7.7 GB machine (free RAM dropped to ~600 MB) [EXP].
- **Verdict:** visually plausible from the photo's viewpoint only. **Not** dimensionally meaningful, **not** suitable for
  cloth simulation without full remeshing into a garment surface. Useful at most as a visual preview.

### B. Garment-specific learned reconstruction
- The relevant open projects (Sewformer, ChatGarment, NeuralTailor-style work) predict **sewing patterns** from images,
  which is the right *representation*, but every candidate failed at least one hard constraint here: GPU-scale models,
  paid components (GPT-4o, Maya), or missing licences. **Not executed.**

### C. Pattern-based construction (GarmentCode tested)
- **Strengths [EXP]:** real sewing patterns (fitted bodice with darts, circle skirt) generated on Windows CPU in seconds;
  SVG/PNG/JSON export; MIT licence; programmable garment families.
- **Weaknesses:** design parameters are **relative to body measurements**, so garment measurements must be inverted
  through a body model (not provided) [EXP]. Turning patterns into a 3D dress needs cloth simulation; the authors'
  simulator is a non-commercial NVIDIA licence and needs a CUDA/MSVC build [DOC S11] — **blocked** here.
- **Verdict:** best long-term *representation* for fitting (patterns + simulation), but on its own cannot produce a 3D
  dress on this machine today, and does not ingest garment measurements directly.

### D. Procedural construction (own)
- Garment = front/back pattern panels in (arc-length, depth) space wrapped onto measured cross-sections; optional sleeves
  lofted from the real armhole loop; seams welded. See README §"How measurements map to geometry".
- **Strengths [EXP]:** exact dimensional control; real openings; topology suited to simulation (single-layer manifold
  surface, ~4–11 k vertices, consistent outward normals, no non-manifold edges); GLB 0.5–1.5 MB.
- **Weaknesses:** limited garment vocabulary (3 silhouettes × 3 sleeve types × 4 necklines, no collars, darts, pleats,
  gathers, pockets, asymmetric designs); folds are decorative sinusoids, not draped; cross-section depth is assumed.

### E. Hybrid (own; recommended)
- Photo → classical CV cues (segmentation, width profile, sleeve presence/angle, neckline type/depth, armhole depth,
  waist position, silhouette class) + texture; measurements → geometry via D. Every value tagged user/photo/default.
- **Strengths [EXP]:** style classification 4/4 correct for silhouette, sleeves and neckline on the synthetic set;
  silhouette IoU 0.91–0.996 across front/side/back; Chamfer mean 0.36–0.95 cm; user-provided dimensions reproduced
  (see §3). Runs in seconds on a CPU, fully offline.
- **Weaknesses:** photo cues tested **only on synthetic renders** — real photos (wrinkles, hangers, mannequins, shadows,
  perspective) are untested; inferred dims are off by 0.2–3.3 cm; side depth and fold structure are not observable from
  one front photo; texture is a projection (stretches on the sides, baked lighting, back assumed = front without a back photo).

## 3. Key quantitative results (synthetic test set, 4 dresses) [EXP]

| Metric | TripoSR (best-case aligned) | Hybrid, front photo + 6–9 measurements | Hybrid, full measurement protocol |
|---|---|---|---|
| Provided girths/lengths, max abs % error | no measurements used; bust/waist/hip −10 … +37 %; length/shoulder/sleeve not measurable (closed solid) | ≤0.43 % except **hip +1.8 / +4.9 %** when hip level unknown | ≤0.43 % (girths/lengths) |
| Silhouette IoU front / side / back | 0.87–0.95 / 0.64–0.85 / 0.90–0.95 (bbox-aligned) | 0.92–0.98 / 0.92–0.996 / 0.91–0.98 | 0.93–0.99 / 0.95–1.00 / 0.92–0.99 |
| Chamfer mean / p95 (cm) | 1.5–3.3 / 4.5–11.5 | 0.36–0.95 / 0.65–2.38 | 0.35–0.86 / 0.62–2.46 |
| Openings (neck/arm/hem) | 0 | 4 (correct) | 4 (correct) |
| Runtime (CPU) | 85–94 s/image (+18 s load) | 3.5–9.6 s | 3.4–8.1 s |
| Mean-colour ΔE front / back | 8.5–13.6 / 19.8–41.6 | 5.6–7.0 / 5.4–7.4 | 6.0–7.2 / 6.2–7.4 |

Per-case numbers: `results/measurements/dimension_errors.csv`, `results/measurements/geometry_metrics.csv`, `results/measurements/mesh_quality.json`.

**Finding — measurement level ambiguity:** a girth without the height at which it was taken is ambiguous on any
shaped garment. On the A-line case the user's hip girth (taken 58 cm below HPS) was applied at the inferred level
(≈55 cm) and the resulting hip girth at 58 cm was 4.9 % too large. Providing `hip_position` fixes it (−0.08 %).
The schema therefore includes position fields and the prototype warns when they are missing.

## 4. Suitability for future fitting/simulation
1. **Hybrid/procedural (D/E):** best starting point — single-layer manifold cloth surface, known seams, panel IDs,
   flattened (approximate) patterns, physical units. Missing: rest-shape/strain consistency (panels are wrapped,
   not simulated), fabric parameters, collision layer.
2. **GarmentCode (C):** best long-term pattern representation; requires a permissively licensed cloth simulator
   (e.g. own XPBD/Blender) and garment-measurement → design-parameter inversion.
3. **TripoSR (A):** unsuitable without heavy post-processing (open the solid, remesh to a single layer, recover seams).

## 5. Real-photo robustness (2 CC0 museum photos, qualitative only — no ground truth) [EXP]
- Both photos show dresses on **mannequins with heads/arms and stands** — the commonest real-world setting.
- Plain GrabCut segmentation included mannequin and stand → wrong HPS, mannequin skin textured onto the bodice.
- Optional colour-model refinement (GMM on central skirt colours) removed the white mannequin from the plaid dress
  (R1, good mask) but also removed the cream bodice + puff sleeves of the ivory gown (R2, fabric ≈ mannequin colour).
- R1 is a 3/4 view: frontal assumptions break (sleeve texture samples background; inferred armhole/waist/neck cues
  rejected as implausible and replaced by defaults).
- **Conclusion:** on real photos the photo-analysis front-end is the weak link; a user-supplied mask (supported in the
  prototype) or a permissively licensed garment-segmentation model is required before real-photo accuracy can be claimed.

## 6. Mesh quality / simulation readiness [EXP] (`results/measurements/mesh_quality.json`)
| | Hybrid (8 meshes) | TripoSR (4 meshes) |
|---|---|---|
| Components / openings | 1 / 4 (neck, hem, 2 armholes or cuffs) | 1–6 / 0 |
| Non-manifold edges | 0 | 0 |
| Triangle aspect ratio p99 | 5.9–8.2 | 66–73 (+ degenerate faces) |
| Edge-length CV | 0.29–0.45 | 0.37–0.39 |
| Pattern→3D strain median / p95 | 0.1–3.7 % / 10–41 % (approximate flattening; worst on flared skirts and the shoulder rounding) | n/a |
| Sleeve vertices inside torso | T3 106/1408 (7.5 %), T4 26–46/≈6000 (≤0.7 %) — near the armhole | n/a |

## 7. Later real-photo test by the user (private photo, not published) [EXP, qualitative]
A dark long-sleeve dress with a white collar and cuffs and a wide flared skirt was processed with all 15 measurements
entered by hand. Segmentation was clean. The entered hem girth (118 cm) describes a much narrower skirt than the photo
shows. The geometry follows the measurements, so the photo texture was squeezed. There were smeared bands along the
sides and the hem, and the collar and cuff pixels landed as patches on the shoulders and sleeves. The verification
table still read ≤ 0.15 % because it checks consistency with the inputs, not resemblance to the photo.
**Gap identified:** there is no photo-vs-measurement consistency check (see FINAL_REPORT.md §12, §18).

# Sources

All accessed **2026-10-16**. "Claim supported" lists only what the page itself stated when fetched.
Where a page did not state something (e.g. VRAM, Windows support), that is noted rather than assumed.

| # | Source (URL) | Claim(s) supported |
|---|---|---|
| S1 | https://github.com/VAST-AI-Research/TripoSR | MIT licence covering code and pretrained models; default settings ~6 GB VRAM for one image; single-image input; `torchmcubes` dependency (CUDA build issues described in troubleshooting); `--bake-texture` option. CPU inference and Windows support not documented. |
| S2 | https://huggingface.co/api/models/stabilityai/TripoSR (and /tree/main) | `"gated": false`; licence `mit`; `model.ckpt` = 1,677,246,742 bytes; `config.yaml` 987 bytes. |
| S3 | https://github.com/VAST-AI-Research/TripoSR (cloned, commit `107cefdc244c39106fa830359024f6a2f1c78871`, 2026-06-04) | LICENSE file text (MIT, © 2024 Tripo AI & Stability AI); `requirements.txt` (transformers 4.35.0, torchmcubes from git, rembg, xatlas, moderngl); code paths: `tsr/models/isosurface.py` (marching cubes), `tsr/models/tokenizers/image.py` (downloads `facebook/dino-vitb16` config.json only; weights come from TripoSR ckpt). |
| S4 | https://github.com/microsoft/TRELLIS | MIT for models and most code (some submodules differently licensed); "An NVIDIA GPU with at least 16GB of memory is necessary"; "currently tested only on Linux"; outputs GLB / Gaussians / radiance fields. |
| S5 | https://github.com/Tencent-Hunyuan/Hunyuan3D-2 | 6 GB VRAM for shape, 16 GB for shape+texture; claims macOS/Windows/Linux support; GLB export via trimesh. |
| S6 | https://raw.githubusercontent.com/Tencent-Hunyuan/Hunyuan3D-2/main/LICENSE | "TENCENT HUNYUAN 3D 2.0 COMMUNITY LICENSE AGREEMENT"; territory excludes the European Union, United Kingdom and South Korea; >1M MAU requires a separate licence from Tencent. |
| S7 | https://huggingface.co/stabilityai/stable-fast-3d | Stability AI Community License (free < US$1M annual revenue, Enterprise licence above); **gated** — login + agreeing to share contact information required. |
| S8 | https://github.com/TencentARC/InstantMesh | Apache-2.0 repository licence; Linux/CUDA-style instructions; relies on a customised Zero123++ UNet (its licence not stated on that page). VRAM not stated. |
| S9 | https://github.com/maria-korosteleva/GarmentCode | MIT licence (repository); "modular programming framework for designing parametric sewing patterns"; body measurement presets; data-generation via a Warp-based simulator. |
| S10 | https://raw.githubusercontent.com/maria-korosteleva/GarmentCode/main/docs/Installation.md | Python 3.9; `pip install pygarment` for the core library; custom NVIDIA Warp fork must be built manually for simulation; cairoSVG quirks on Windows; dependency list (CGAL, libigl, pyrender, NiceGUI…). |
| S11 | https://github.com/maria-korosteleva/NvidiaWarp-GarmentCode | Warp fork under the **NVIDIA Source Code License — "currently allows only non-commercial use"**; CUDA Toolkit ≥ 11.5 listed (CPU-only build possible per build notes); Visual Studio 2019+ for Windows builds. |
| S12 | https://github.com/biansy000/ChatGarment | Apache-2.0; built on LLaVA and LISA; depends on GarmentCodeRC + ContourCraft-CG for simulation; text pipeline uses GPT-4o (a paid API). |
| S13 | https://github.com/sail-sg/sewformer | Sewing pattern prediction from a single image; no licence stated on the page; simulation of predictions via Autodesk Maya (`mayapy.exe`). |
| S14 | https://arxiv.org/abs/2311.04218 | Sewformer paper: SewFactory dataset (~1M images with sewing patterns), Transformer predicting patterns from photos. |
| S15 | https://github.com/danielgatis/rembg | rembg itself MIT; default model is now BRIA RMBG-2.0 which needs a paid agreement for commercial use; models download to `~/.rembg` (or `REMBG_HOME`/`U2NET_HOME`). |
| S16 | https://github.com/xuebinqin/U-2-Net | U-2-Net code Apache-2.0; weight licence not stated on the page (Google Drive / Baidu downloads). |
| S17 | https://arxiv.org/abs/2403.18816 | Garment3DGen paper (3DV 2025) — code availability claimed; licence **not verified** (repository URL 404 at access time). |
| S18 | https://arxiv.org/abs/2312.16264 | SPnet — single image → T-pose garment → sewing-pattern parameters; no code/licence found. |
| S19 | https://download.pytorch.org/whl/cpu | Source of the CPU-only PyTorch wheel installed in `.venv-triposr` (BSD-style PyTorch licence). |
| S20 | PyPI (numpy, scipy, trimesh, shapely, opencv-python-headless, pillow, matplotlib, networkx, scikit-image, omegaconf, einops, transformers, huggingface-hub) | Installed into project-local venvs only; all permissive licences (BSD/MIT/Apache-2.0; OpenCV ≥4.5 Apache-2.0). |
| S21 | https://pypi.org/pypi/pygarment/json | pygarment 2.0.2 (2025-04-18), licence MIT, requires numpy<2, CGAL, libigl, pyrender, nicegui, CairoSVG. |
| S22 | https://github.com/maria-korosteleva/GarmentCode/blob/main/assets/bodies/Readme.md (cloned, commit `d449629`) | SMPL-based average bodies redistributed under CC BY 4.0 (per authors); other bodies from the authors' own model; full SMPL body set not shareable. |
| S23 | https://commons.wikimedia.org/wiki/File:Claire_McCardell_day_dress,_yellow_and_black_plaid_wool,_c_1950.jpg | CC0 (Commons licence metadata via API); used as real test photo R1 (600 px thumbnail). |
| S24 | https://commons.wikimedia.org/wiki/File:Japon_van_lichtblauwe_zijde_met_strepen_van_paren_uitwaaierende_bloemen_versiering_met_gewatteerde_randen,_langs_de_zoom_in_schakels_en_op_het_lijf_in_vijf_strepen,_BK-NM-8471.jpg | CC0 (Rijksmuseum via Commons); used as real test photo R2 (960 px thumbnail). |
| S25 | https://collectionapi.metmuseum.org/public/collection/v1/search | Returned **HTTP 410 Gone** on 2026-10-16 — Met Open Access API not usable; not relied on. |

## Facts verified by running code in this workspace (not from documentation)
- Hardware: Intel i5-1235U, 7.7 GB RAM, Intel UHD integrated GPU, no CUDA (see research/research-log.md).
- All results under `results/` — see FINAL_REPORT.md.

# Male avatar system work — progress log (started 2026-10-17 11:12)

Workspace: D:\GraduationProject_Workspace\MyAvatar_GitHub (var/ is git-ignored).
Backups: var/backups/20261017_before_male_work (before any male work),
         var/backups/20261017_before_male_system_work (state at the start of this task:
         decoupled fit, no pregnancy control). Restore: tar -xf files.tar from the repo root.
Running services: avatar API 8002 = this clone's backend/api.py, started with
  D:\BarahaAvatarModule\.venv\Scripts\python.exe -B api.py (cwd backend/, interpreter only),
  env AVATAR_OUTPUT_DIR=D:\GraduationProject_Workspace\var\generated, TMP/TEMP=var\tmp.
Protected (never written): D:\BarahaAvatarModule, D:\BarahaAvatar, D:\AvatarGitHub\avatar.

## Log
- 11:12 backups made; audit starting.
- Audit (baseline_decoupled): deterministic; reported == GLB mesh (<=0.03); 3 silent failures (chest None).
- ROOT CAUSE 1 (measurement): measure().chest_m slices at armpit level (22-28 cm above waist, nipples at 17-21)
  where the rest-pose torso is fused with the raised arms; reads 6-13 cm over a tape, None on bulky torsos.
  -> measurements.tape_measurements(): chest at nipple height (vertices 8526/1870), arms excluded by skinning,
     convex outline; hip at greatest buttock prominence (ISO 8559-1). Male only; measure() unchanged.
- ROOT CAUSE 2 (solver): weight re-solved after every control step -> mass removed by controls refilled as global
  fat -> spiral to fat 0.999 + all controls compressing. -> staged fit: A phenotype, B controls with phenotype
  frozen, C damped weight correction (fat, or muscle) only while no control is pushed to its limit.
- Muscle capped 0.7 (0.87 gave 44 cm upper arms). Weight tolerance compared in real kg. API reports the exported body.
- Rejected: muscle as waist/hip lever (moves them only 1-2 cm at fixed weight).
- Matrix runs: var/male_qa/{baseline_decoupled,tape_v1,staged_v1..v4}; staged_v4 = current: no silent misses.
- Hip re-anchored at greatest buttock prominence (max girth slid onto thigh roots in the rest pose, +4 cm).
- Weight correction falls back to muscle when a fat step is rejected (tall_heavy +3.5 -> +1.2 kg).
- Final matrix: var/male_qa/staged_v4 (no silent misses). Visual sheets: var/male_qa/compare/.
- Tests: 112 fast (31 skipped) OK; 27 male incl. 9 real-model OK; frontend tsc/build/8 unit OK; browser e2e 26/26.
- Female: F1 GLB byte-identical to the pre-task F1 GLB; fit parameters identical to tools/baseline_cases.json.
- API running on final code (restart command at the top). Nothing committed.
- Removed my stray backend/landmarks.json (written by a scratch prototype).

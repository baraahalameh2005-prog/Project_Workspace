# Stage 1 handoff: avatar creation page (state of 2026-10-17)

For anyone integrating with the avatar module (e.g. the clothing-size recommendation and
fit-assessment module). Facts only; details and evidence are linked.

## What changed in Stage 1

| Area | What it does now | Main files |
|---|---|---|
| Avatar persistence | The last generated avatar's reference (`avatar_id`, `avatar_url`, request and response, ~3 KB) is kept in the browser's `localStorage` (key `baraha-avatar:last`) and restored after navigation or refresh. No GLB data is stored in the browser. | `frontend/src/savedAvatar.ts`, `frontend/src/App.tsx` |
| Transfer to the fitting room | Button «إرسال الأفاتار إلى غرفة القياس» opens `http://127.0.0.1:8010/viewer/?avatar=<avatar_id>` for the avatar on screen, after checking that it has loaded. | `frontend/src/App.tsx` |
| Adult male fit | Separate male path `fit_to_measurements.fit_adult_male` (female path unchanged): adult age, one Anny control per measurement, no pregnancy shape, staged so that body fat is not inflated, muscle capped. Optional male `shoulder` input. Measurements that the model cannot reach are reported in `out_of_range`, not forced. | `backend/fit_to_measurements.py`, `backend/measurements.py`, `backend/api.py` |

API details: `docs/API.md` (section "Male bodies").

## Integration points

- `POST /generate-avatar` (port 8002) returns `avatar_id`, `avatar_url` (`/models/generated/<avatar_id>.glb`),
  `measurements`, `accuracy`, `out_of_range`, `fit_method`. For male avatars these values are measured on the
  exported body; chest and hip are tape-style girths (`chest_tape_m`, `hip_tape_m`).
- GLB: one static mesh, metres, Y up, feet at y = 0, arms in Anny's raised rest pose (A-pose), no skeleton.
- The GLBs are written to the folder set by `AVATAR_OUTPUT_DIR` (on the development machine
  `D:\GraduationProject_Workspace\var\generated`), which the fitting room (Clothing3D, a separate repository:
  https://github.com/Saja-Khalil/-clothing-3d) reads by `avatar_id`.

## Verification (what was actually run)

- Backend fast tests: 112 run, 31 skipped (they need the Anny model), 0 failures.
- Real-model male tests (9) and the female baseline check: passed in the development session; female output
  was byte-identical to the pre-change output for the recorded baseline case.
- Frontend: `tsc --noEmit` clean, 8/8 unit tests (`node --test frontend/tests/savedAvatar.test.mjs`).
- Browser workflow in headless Edge (generate, navigate away, refresh, transfer, error states): 26/26 checks.
- 27-case male test matrix measured on the exported GLBs, with front/side/back renders:
  `docs/reports/male_avatar_qa_2026-10-17/`. All cases are synthetic test data, except the inputs
  reconstructed from one reported avatar.

## Not verified / known limitations

- No real person's measurements were available: resemblance to a real body has not been validated.
- Shoulder width (straight distance between the shoulder tips) is limited by Anny: about 44-45 cm at 170 cm.
  At the shoulder control's maximum the upper body looks squared (`images/upper_body_study_closeup.png`).
  An experiment on lowering that maximum was started and interrupted; nothing from it has been applied.
- Very heavy bodies (BMI above about 32) cannot be represented; weight is an estimate (volume x 1040 kg/m3).
- Manual testing by the user in a normal browser is still pending.

## Running

Backend: see `docs/API.md` (port 8002; needs the Anny model environment, `backend/requirements.txt`).
Frontend: `cd frontend`, `npm ci`, `npm run dev` (port 5173; backend URL via `VITE_API_URL`,
fitting room URL via `VITE_CLOTHING_URL`).

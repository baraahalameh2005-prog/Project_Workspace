import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import AvatarViewer, { ViewMode, ViewerStatus } from "./AvatarViewer";
import {
  API_URL,
  ApiError,
  AvatarOptions,
  AvatarRequest,
  AvatarResponse,
  GENERATE_TIMEOUT_MS,
  MEASUREMENT_NAMES,
  MeasurementLimit,
  MeasurementName,
  generateAvatar,
  getAvatarOptions,
  getHealth,
  resolveApiUrl,
} from "./api";
import { clearSavedAvatar, fittingRoomUrl, loadSavedAvatar, storeSavedAvatar } from "./savedAvatar";

// The clothing module's server (Clothing3D backend/server.py); its fitting room
// loads the avatar by id from var/generated/.
const CLOTHING_URL = ((import.meta.env.VITE_CLOTHING_URL as string | undefined) ?? "http://127.0.0.1:8010").replace(
  /\/+$/,
  "",
);
const CLOTHING_CHECK_TIMEOUT_MS = 3_000;

// The size-recommendation module (FitEngine/, Nagham). It receives the measurements
// the user entered for this avatar and recommends a garment size; it does not use
// the generated mesh.
const FIT_URL = ((import.meta.env.VITE_FIT_URL as string | undefined) ?? "http://127.0.0.1:8020").replace(/\/+$/, "");
const FIT_KEYS = ["height", "chest", "waist", "hip", "shoulder"] as const;

// Used only until /avatar-options has loaded; the backend validates either way.
const FALLBACK_LIMITS: Record<MeasurementName, MeasurementLimit> = {
  height: { required: true, minimum: 100, maximum: 230, unit: "cm" },
  weight: { required: true, minimum: 25, maximum: 250, unit: "kg" },
  waist: { required: true, minimum: 40, maximum: 200, unit: "cm" },
  chest: { required: true, minimum: 50, maximum: 200, unit: "cm" },
  hip: { required: true, minimum: 50, maximum: 200, unit: "cm" },
};
const FALLBACK_GENDERS = ["female", "male"];
// Arabic names for the response's out_of_range entries.
const OUT_OF_RANGE_LABELS: Record<string, string> = {
  chest: "محيط الصدر",
  waist: "محيط الخصر",
  hip: "محيط الورك",
  shoulder: "عرض الكتفين",
  weight: "الوزن",
};
// Used only until /avatar-options has loaded (optional_measurements.shoulder).
const FALLBACK_SHOULDER_LIMIT = { minimum: 30, maximum: 60, unit: "cm" };
const FALLBACK_SKIN_TONE = "#c8997e"; // backend default (/avatar-options)
const FALLBACK_EYE_COLOR = "brown";
const SKIN_TONE_PRESETS = [
  { label: "Light", value: "#e2bfa3" },
  { label: "Medium (default)", value: "#c8997e" },
  { label: "Tan", value: "#a8754f" },
  { label: "Brown", value: "#8a5a3c" },
  { label: "Dark", value: "#6b4430" },
];

const DEFAULT_VALUES: Record<MeasurementName, string> = {
  height: "165",
  weight: "60",
  waist: "72",
  chest: "90",
  hip: "98",
};

const LABELS: Record<MeasurementName, string> = {
  height: "Height",
  weight: "Weight",
  waist: "Waist",
  chest: "Chest",
  hip: "Hip",
};

type BackendState = "checking" | "unreachable" | "loading-model" | "ready";

interface ShownResult {
  response: AvatarResponse;
  glbUrl: string;
  receivedAt: Date;
  requestNumber: number;
  request: AvatarRequest;
  // Restored from browser storage, not generated in this page load.
  restored: boolean;
}

interface RequestError {
  requestNumber: number;
  error: ApiError;
}

// An existing avatar opened with ?avatar=/models/generated/<id>.glb (no generation request).
interface LinkedAvatar {
  avatarId: string;
  path: string;
  glbUrl: string;
}

const LINKED_AVATAR_PATTERN = /^\/models\/generated\/([0-9a-f]{32})\.glb$/;

// Temporary face preview: existing face-sweep GLBs copied into frontend/public, served by Vite itself.
const FACE_PREVIEW_PATTERN = /^\/face_preview_20261011_235711\/([a-z0-9-]+_(?:0|plus1|minus1))\.glb$/;

function readLinkedAvatar(): { linked: LinkedAvatar | null; error: string | null } {
  const raw = new URLSearchParams(window.location.search).get("avatar");
  if (raw === null) return { linked: null, error: null };

  const preview = FACE_PREVIEW_PATTERN.exec(raw);
  if (preview) {
    return { linked: { avatarId: preview[1], path: raw, glbUrl: new URL(raw, window.location.origin).toString() }, error: null };
  }

  const match = LINKED_AVATAR_PATTERN.exec(raw);
  if (!match) {
    return {
      linked: null,
      error: `Ignored avatar link "${raw}": expected /models/generated/<32 hex characters>.glb.`,
    };
  }
  return { linked: { avatarId: match[1], path: raw, glbUrl: resolveApiUrl(raw) }, error: null };
}

function fmt(value: number | null | undefined, digits = 2) {
  return value === null || value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

/** Resolves when the fitting room's server answers (the response itself is opaque: no CORS). */
async function fittingRoomReachable(): Promise<boolean> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), CLOTHING_CHECK_TIMEOUT_MS);
  try {
    await fetch(`${CLOTHING_URL}/viewer/`, { mode: "no-cors", cache: "no-store", signal: controller.signal });
    return true;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

/** Resolves when the size-recommendation server answers. */
async function fitEngineReachable(): Promise<boolean> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), CLOTHING_CHECK_TIMEOUT_MS);
  try {
    await fetch(`${FIT_URL}/health`, { mode: "no-cors", cache: "no-store", signal: controller.signal });
    return true;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

/** FitEngine page with the entered measurements in the query string (cm). */
function fitEngineUrl(request: AvatarRequest, avatarId: string): string {
  const params = new URLSearchParams({ avatar_id: avatarId });
  const entered = request as unknown as Record<string, unknown>;
  for (const key of FIT_KEYS) {
    const value = entered[key];
    if (typeof value === "number" && Number.isFinite(value)) params.set(key, String(value));
  }
  return `${FIT_URL}/?${params.toString()}`;
}

export default function App() {
  // An explicit ?avatar= link wins; otherwise the last generated avatar is restored.
  const [link] = useState(readLinkedAvatar);
  const [saved] = useState(() => (link.linked ? null : loadSavedAvatar(API_URL)));

  const [backend, setBackend] = useState<BackendState>("checking");
  const [options, setOptions] = useState<AvatarOptions | null>(null);
  const [optionsError, setOptionsError] = useState<string | null>(null);

  const [values, setValues] = useState<Record<MeasurementName, string>>(() =>
    saved
      ? (Object.fromEntries(MEASUREMENT_NAMES.map((n) => [n, String(saved.request[n])])) as Record<MeasurementName, string>)
      : DEFAULT_VALUES,
  );
  const [gender, setGender] = useState(saved?.request.gender ?? "female");
  // Male only and optional; "" means not given (the backend then derives the shoulders).
  const [shoulder, setShoulder] = useState(
    saved?.request.shoulder !== undefined ? String(saved.request.shoulder) : "",
  );
  const [shoulderError, setShoulderError] = useState<string | null>(null);
  const [skinTone, setSkinTone] = useState(saved?.request.skin_tone ?? FALLBACK_SKIN_TONE);
  const [skinToneTouched, setSkinToneTouched] = useState(saved?.request.skin_tone !== undefined);
  const [eyeColor, setEyeColor] = useState(saved?.request.eye_color ?? FALLBACK_EYE_COLOR);
  const [fieldErrors, setFieldErrors] = useState<Partial<Record<MeasurementName, string>>>({});

  const [pending, setPending] = useState<{ requestNumber: number; startedAt: number } | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<RequestError | null>(null);
  const [result, setResult] = useState<ShownResult | null>(() =>
    saved
      ? {
          response: saved.response,
          glbUrl: resolveApiUrl(saved.response.avatar_url),
          receivedAt: new Date(saved.savedAt),
          requestNumber: 0,
          request: saved.request,
          restored: true,
        }
      : null,
  );
  const [storageFailed, setStorageFailed] = useState(false);
  const [transferMessage, setTransferMessage] = useState<string | null>(null);
  const [fitMessage, setFitMessage] = useState<string | null>(null);
  const [fitChecking, setFitChecking] = useState(false);
  const [transferring, setTransferring] = useState(false);
  const [viewer, setViewer] = useState<ViewerStatus>({ state: "empty" });
  // "?view=face" opens the face framing directly (handy for screenshots).
  const [view, setView] = useState<ViewMode>(() =>
    new URLSearchParams(window.location.search).get("view") === "face" ? "face" : "body",
  );

  const requestCounter = useRef(0);
  const latestRequest = useRef(0);
  const cancelRef = useRef<AbortController | null>(null);

  const limits = options?.measurements ?? FALLBACK_LIMITS;
  const genders = options?.gender.values ?? FALLBACK_GENDERS;
  // Only offered when the backend advertises eye colours (/avatar-options).
  const eyeColors = options?.eye_color?.values ?? null;
  // Offered for male avatars when the backend accepts it (or before /avatar-options has loaded).
  const shoulderLimit = options
    ? (options.optional_measurements?.shoulder ?? null)
    : { ...FALLBACK_SHOULDER_LIMIT, required: false, genders: ["male"] };
  const showShoulder = shoulderLimit !== null && shoulderLimit.genders.includes(gender);

  // Backend readiness: poll /health; load /avatar-options once it answers.
  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;

    const poll = async () => {
      try {
        const health = await getHealth();
        if (stopped) return;
        setBackend(health.model_loaded ? "ready" : "loading-model");
      } catch {
        if (stopped) return;
        setBackend("unreachable");
      }
      timer = setTimeout(poll, 5000);
    };
    poll();

    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    if (options || backend === "checking" || backend === "unreachable") return;
    let stopped = false;
    getAvatarOptions()
      .then((opts) => {
        if (stopped) return;
        setOptions(opts);
        setOptionsError(null);
        if (!skinToneTouched) setSkinTone(opts.skin_tone.default);
        if (opts.eye_color && !opts.eye_color.values.includes(eyeColor)) setEyeColor(opts.eye_color.default);
        if (!opts.gender.values.includes(gender)) setGender(opts.gender.default);
      })
      .catch((err: unknown) => {
        if (!stopped) setOptionsError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      stopped = true;
    };
    // Load once when the backend first becomes reachable.
  }, [backend, options]);

  // Elapsed-time counter while waiting.
  useEffect(() => {
    if (!pending) return;
    setElapsed(0);
    const id = setInterval(() => setElapsed((Date.now() - pending.startedAt) / 1000), 500);
    return () => clearInterval(id);
  }, [pending]);

  function validate(): AvatarRequest | null {
    const errors: Partial<Record<MeasurementName, string>> = {};
    const parsed = {} as Record<MeasurementName, number>;

    for (const name of MEASUREMENT_NAMES) {
      const raw = values[name].trim();
      const value = Number(raw);
      const { minimum, maximum, unit } = limits[name];
      if (raw === "" || !Number.isFinite(value)) {
        errors[name] = "Enter a number.";
      } else if (value < minimum || value > maximum) {
        errors[name] = `Must be between ${minimum} and ${maximum} ${unit}.`;
      } else {
        parsed[name] = value;
      }
    }

    let shoulderValue: number | undefined;
    let shoulderMessage: string | null = null;
    if (showShoulder && shoulderLimit && shoulder.trim() !== "") {
      const value = Number(shoulder.trim());
      if (!Number.isFinite(value)) {
        shoulderMessage = "أدخل رقماً صحيحاً لعرض الكتفين، أو اترك الحقل فارغاً.";
      } else if (value < shoulderLimit.minimum || value > shoulderLimit.maximum) {
        shoulderMessage = `عرض الكتفين يجب أن يكون بين ${shoulderLimit.minimum} و${shoulderLimit.maximum} سم.`;
      } else {
        shoulderValue = value;
      }
    }

    setFieldErrors(errors);
    setShoulderError(shoulderMessage);
    if (Object.keys(errors).length > 0 || shoulderMessage) return null;

    return {
      gender,
      ...parsed,
      ...(shoulderValue !== undefined ? { shoulder: shoulderValue } : {}),
      skin_tone: skinTone,
      ...(eyeColors ? { eye_color: eyeColor } : {}),
    };
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (pending) return;

    const body = validate();
    if (!body) return;

    const requestNumber = ++requestCounter.current;
    latestRequest.current = requestNumber;
    const cancel = new AbortController();
    cancelRef.current = cancel;
    setError(null);
    setTransferMessage(null);
    setPending({ requestNumber, startedAt: Date.now() });

    try {
      const response = await generateAvatar(body, cancel.signal);
      // Ignore responses from requests that are no longer the latest.
      if (requestNumber !== latestRequest.current) return;
      setStorageFailed(!storeSavedAvatar(API_URL, body, response));
      setResult({
        response,
        glbUrl: resolveApiUrl(response.avatar_url),
        receivedAt: new Date(),
        requestNumber,
        request: body,
        restored: false,
      });
    } catch (err) {
      if (requestNumber !== latestRequest.current) return;
      setError({
        requestNumber,
        error: err instanceof ApiError ? err : new ApiError("network", String(err)),
      });
    } finally {
      if (requestNumber === latestRequest.current) {
        setPending(null);
        cancelRef.current = null;
      }
    }
  }

  function onCancel() {
    cancelRef.current?.abort();
  }

  const onViewerStatus = useCallback((status: ViewerStatus) => setViewer(status), []);

  // A generated result always takes precedence over the linked avatar.
  const linked = result ? null : link.linked;
  const viewerUrl = result?.glbUrl ?? linked?.glbUrl ?? null;
  const viewerMatchesResult = result && "url" in viewer && viewer.url === result.glbUrl;
  const viewerMatchesLinked = linked && "url" in viewer && viewer.url === linked.glbUrl;
  const errorIsNewerThanResult = error && (!result || error.requestNumber > result.requestNumber);
  const restoreFailed = result?.restored && viewerMatchesResult && viewer.state === "failed";

  // The avatar on screen is the one sent; null id means it is not a generated avatar (face preview).
  const selected = result
    ? { avatarId: result.response.avatar_id, glbUrl: result.glbUrl }
    : linked
      ? { avatarId: /^[0-9a-f]{32}$/.test(linked.avatarId) ? linked.avatarId : null, glbUrl: linked.glbUrl }
      : null;

  function transferBlocker(): string | null {
    if (pending) return "الأفاتار قيد الإنشاء الآن. انتظر حتى يكتمل الإنشاء ثم أرسله.";
    if (!selected) return "لا يوجد أفاتار بعد. أدخل القياسات واضغط «Generate avatar» أولاً.";
    if (!selected.avatarId) return "هذا الأفاتار للمعاينة فقط وليس أفاتاراً مُنشأً، فلا يمكن إرساله إلى غرفة القياس.";
    if (!("url" in viewer) || viewer.url !== selected.glbUrl || viewer.state === "loading")
      return "الأفاتار ما زال قيد التحميل. انتظر حتى يظهر ثم أعد المحاولة.";
    if (viewer.state === "failed") return "تعذّر تحميل ملف هذا الأفاتار من الخادم، لذلك لا يمكن إرساله. أنشئ أفاتاراً جديداً.";
    return null;
  }

  async function onTransfer() {
    const blocker = transferBlocker();
    if (blocker || !selected?.avatarId) {
      setTransferMessage(blocker);
      return;
    }
    setTransferMessage(null);
    setTransferring(true);
    const reachable = await fittingRoomReachable();
    setTransferring(false);
    if (!reachable) {
      setTransferMessage(`غرفة القياس غير متاحة على ${CLOTHING_URL}. تأكد من تشغيل خادم الملابس ثم أعد المحاولة.`);
      return;
    }
    window.location.assign(fittingRoomUrl(CLOTHING_URL, selected.avatarId));
  }

  // Size recommendation needs the measurements the user entered, which only a
  // generated (or restored) result carries; a ?avatar= link has the GLB only.
  function fitBlocker(): string | null {
    if (pending) return "الأفاتار قيد الإنشاء الآن. انتظر حتى يكتمل الإنشاء.";
    if (!result) return "لا توجد قياسات بعد. أدخل القياسات واضغط «Generate avatar» أولاً.";
    return null;
  }

  async function onRecommendSize() {
    const blocker = fitBlocker();
    if (blocker || !result) {
      setFitMessage(blocker);
      return;
    }
    setFitMessage(null);
    setFitChecking(true);
    const reachable = await fitEngineReachable();
    setFitChecking(false);
    if (!reachable) {
      setFitMessage(`خدمة اقتراح المقاس غير متاحة على ${FIT_URL}. شغّلها من مجلد FitEngine: python -m fit_engine serve --port 8020`);
      return;
    }
    window.location.assign(fitEngineUrl(result.request, result.response.avatar_id));
  }

  function onForgetSaved() {
    clearSavedAvatar();
    setResult(null);
  }

  return (
    <div className="app">
      <header>
        <h1>Avatar Module</h1>
        <span className={`backend backend-${backend}`}>
          {backend === "checking" && "Checking backend…"}
          {backend === "unreachable" && `Backend not reachable (${API_URL})`}
          {backend === "loading-model" && "Backend up, model still loading"}
          {backend === "ready" && "Backend ready"}
        </span>
      </header>

      <main>
        <section className="panel">
          <form onSubmit={onSubmit} noValidate>
            <fieldset disabled={pending !== null}>
              <legend>Body measurements</legend>

              <label className="field">
                <span>Gender</span>
                <select value={gender} onChange={(e) => setGender(e.target.value)}>
                  {genders.map((g) => (
                    <option key={g} value={g}>
                      {g}
                    </option>
                  ))}
                </select>
              </label>

              {MEASUREMENT_NAMES.map((name) => {
                const { minimum, maximum, unit } = limits[name];
                return (
                  <label className="field" key={name}>
                    <span>
                      {LABELS[name]} <small>({unit}, {minimum}–{maximum})</small>
                    </span>
                    <input
                      type="number"
                      inputMode="decimal"
                      step="any"
                      min={minimum}
                      max={maximum}
                      value={values[name]}
                      aria-invalid={fieldErrors[name] ? true : undefined}
                      onChange={(e) => setValues((v) => ({ ...v, [name]: e.target.value }))}
                    />
                    {fieldErrors[name] && <em className="field-error">{fieldErrors[name]}</em>}
                  </label>
                );
              })}

              {gender === "male" && (
                <p className="hint" dir="rtl" lang="ar">
                  <strong>طريقة القياس:</strong> قِف مستقيماً والذراعان منسدلتان، وشدّ الشريط أفقياً على الجلد دون ضغط.
                  الصدر: حول الصدر عند مستوى الحلمتين. الخصر: حول البطن عند مستوى السرة تقريباً. الورك: حول الأرداف عند
                  أبرز نقطة لها من الجانب. الطول بلا حذاء.
                </p>
              )}

              {showShoulder && shoulderLimit && (
                <label className="field" dir="rtl" lang="ar">
                  <span>
                    عرض الكتفين{" "}
                    <small>
                      (سم، {shoulderLimit.minimum}–{shoulderLimit.maximum}، اختياري)
                    </small>
                  </span>
                  <input
                    type="number"
                    inputMode="decimal"
                    step="any"
                    min={shoulderLimit.minimum}
                    max={shoulderLimit.maximum}
                    dir="ltr"
                    value={shoulder}
                    placeholder="مثال: 42"
                    aria-invalid={shoulderError ? true : undefined}
                    onChange={(e) => setShoulder(e.target.value)}
                  />
                  <small className="hint">
                    المسافة <strong>المستقيمة</strong> بين طرفي الكتفين: العظمة البارزة في نهاية كل كتف من الأعلى. قِسها
                    من الخلف بمسطرة أو بشريط مشدود بخط مستقيم، لا يلتف حول الكتفين. القيمة المعتادة للرجل البالغ بين 36
                    و46 سم. لا تقِس أعرض نقطة فوق عضلات الكتفين ولا عرض كتف القميص، فهما أكبر من هذا القياس. إذا تُرك
                    الحقل فارغاً يُحسب عرض الكتفين من القياسات الأخرى.
                  </small>
                  {shoulderError && <em className="field-error">{shoulderError}</em>}
                </label>
              )}

            </fieldset>

            <fieldset disabled={pending !== null} className="appearance">
              <legend>Appearance</legend>

              <div className="field">
                <span>Skin tone</span>
                <div className="swatches">
                  {SKIN_TONE_PRESETS.map((p) => (
                    <button
                      type="button"
                      key={p.value}
                      title={`${p.label} ${p.value}`}
                      className={`swatch ${skinTone.toLowerCase() === p.value ? "selected" : ""}`}
                      style={{ background: p.value }}
                      onClick={() => {
                        setSkinTone(p.value);
                        setSkinToneTouched(true);
                      }}
                    />
                  ))}
                  <label className="custom-color" title="Custom colour">
                    <input
                      type="color"
                      value={skinTone}
                      onChange={(e) => {
                        setSkinTone(e.target.value);
                        setSkinToneTouched(true);
                      }}
                    />
                  </label>
                  <code>{skinTone}</code>
                </div>
              </div>

              {eyeColors ? (
                <label className="field">
                  <span>Eye colour</span>
                  <select value={eyeColor} onChange={(e) => setEyeColor(e.target.value)}>
                    {eyeColors.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </label>
              ) : (
                <p className="hint">Eye colour: not offered by this backend version.</p>
              )}

              <p className="hint">
                Applied automatically from gender: face shape, eyebrows, lip tint (female only).
                <br />
                Not available yet: hair.
              </p>
            </fieldset>

            <div className="actions">
              <button type="submit" disabled={pending !== null || backend === "unreachable"}>
                {pending ? "Generating…" : "Generate avatar"}
              </button>
              {pending && (
                <button type="button" className="secondary" onClick={onCancel}>
                  Stop waiting
                </button>
              )}
            </div>
            {optionsError && <p className="hint">Could not load limits from the backend; using built-in limits.</p>}
          </form>

          <div className="transfer" dir="rtl" lang="ar">
            <button
              type="button"
              className={`transfer-button ${transferBlocker() ? "blocked" : ""}`}
              aria-disabled={transferBlocker() !== null || transferring}
              onClick={transferring ? undefined : onTransfer}
            >
              {transferring ? "جارٍ التحقق من غرفة القياس…" : "إرسال الأفاتار إلى غرفة القياس"}
            </button>
            {selected?.avatarId && (
              <p className="hint">
                الأفاتار الذي سيُرسل: <code dir="ltr">{selected.avatarId}</code>
              </p>
            )}
            {transferMessage && (
              <div className="notice error" role="alert">
                {transferMessage}
              </div>
            )}
          </div>

          <div className="transfer" dir="rtl" lang="ar">
            <button
              type="button"
              className={`transfer-button ${fitBlocker() ? "blocked" : ""}`}
              aria-disabled={fitBlocker() !== null || fitChecking}
              onClick={fitChecking ? undefined : onRecommendSize}
            >
              {fitChecking ? "جارٍ التحقق من خدمة المقاسات…" : "اقترح لي مقاسًا"}
            </button>
            <p className="hint">تُرسل القياسات التي أدخلتها (الطول، الصدر، الخصر، الورك، والكتفين إن وُجدت)، وليس قياسات الموديل.</p>
            {fitMessage && (
              <div className="notice error" role="alert">
                {fitMessage}
              </div>
            )}
          </div>

          {storageFailed && (
            <div className="notice warning" dir="rtl" lang="ar">
              تعذّر حفظ الأفاتار في هذا المتصفح (التخزين محظور). سيبقى معروضاً الآن، لكنه لن يُسترجع بعد مغادرة الصفحة أو
              تحديثها.
            </div>
          )}

          {restoreFailed && (
            <div className="notice error" dir="rtl" lang="ar" role="alert">
              <strong>تعذّر استرجاع الأفاتار السابق.</strong>
              <p>
                لم يُحمَّل ملف الأفاتار <code dir="ltr">{result.response.avatar_id}</code> من الخادم ({API_URL}). قد
                يكون الخادم متوقفاً أو أن الملف حُذف. إذا كان الخادم يعمل، أنشئ أفاتاراً جديداً.
              </p>
              <button type="button" className="secondary" onClick={onForgetSaved}>
                نسيان الأفاتار المحفوظ
              </button>
            </div>
          )}

          {pending && (
            <div className="notice info">
              Request #{pending.requestNumber} running: {elapsed.toFixed(0)} s (stops waiting after{" "}
              {GENERATE_TIMEOUT_MS / 1000} s). Fitting usually takes tens of seconds.
            </div>
          )}

          {error && errorIsNewerThanResult && (
            <div className="notice error">
              <strong>
                Request #{error.requestNumber}: {error.error.message}
              </strong>
              {error.error.details.length > 0 && (
                <ul>
                  {error.error.details.map((d, i) => (
                    <li key={i}>{d}</li>
                  ))}
                </ul>
              )}
              {result && result.restored && (
                <p dir="rtl" lang="ar">
                  الأفاتار المعروض ما زال الأفاتار المحفوظ السابق.
                </p>
              )}
              {result && !result.restored && <p>The avatar shown is still from request #{result.requestNumber}.</p>}
            </div>
          )}

          {link.error && !result && <div className="notice error">{link.error}</div>}

          {linked && (
            <div className="result">
              <h2>Loaded from link, not from a generation in this page</h2>
              <dl>
                <dt>Avatar ID</dt>
                <dd>
                  <code>{linked.avatarId}</code>
                </dd>
                <dt>Viewer</dt>
                <dd>
                  {!viewerMatchesLinked && "waiting"}
                  {viewerMatchesLinked && viewer.state === "loading" && "loading GLB…"}
                  {viewerMatchesLinked && viewer.state === "displayed" && "showing this avatar"}
                  {viewerMatchesLinked && viewer.state === "failed" && "GLB failed to load"}
                </dd>
                <dt>GLB</dt>
                <dd>
                  <a href={linked.glbUrl} target="_blank" rel="noreferrer">
                    {linked.path}
                  </a>
                </dd>
              </dl>
              <p className="hint">No measurements, accuracy or warnings: this page did not receive the generation response.</p>
            </div>
          )}

          {result && (
            <div className="result">
              {result.restored ? (
                <h2 dir="rtl" lang="ar">
                  آخر أفاتار أنشأته (مُسترجَع، أُنشئ في {result.receivedAt.toLocaleString("ar")})
                </h2>
              ) : (
                <h2>Result of request #{result.requestNumber}</h2>
              )}
              <dl>
                <dt>Avatar ID</dt>
                <dd>
                  <code>{result.response.avatar_id}</code>
                </dd>
                <dt>Received</dt>
                <dd>{result.restored ? result.receivedAt.toLocaleString() : result.receivedAt.toLocaleTimeString()}</dd>
                <dt>Fit time</dt>
                <dd>{fmt(result.response.fit_time_seconds, 1)} s</dd>
                <dt>Viewer</dt>
                <dd>
                  {!viewerMatchesResult && "waiting"}
                  {viewerMatchesResult && viewer.state === "loading" && "loading GLB…"}
                  {viewerMatchesResult && viewer.state === "displayed" && "showing this avatar"}
                  {viewerMatchesResult && viewer.state === "failed" && "GLB failed to load"}
                </dd>
                <dt>GLB</dt>
                <dd>
                  <a href={result.glbUrl} target="_blank" rel="noreferrer">
                    {result.response.avatar_url}
                  </a>
                </dd>
              </dl>

              {(result.response.out_of_range?.length ?? 0) > 0 && (
                <div className="notice warning" dir="rtl" lang="ar" role="status">
                  <strong>بعض القياسات خارج ما يستطيع النموذج تمثيله، فلم تُفرض على الجسم:</strong>
                  <ul>
                    {result.response.out_of_range!.map((name) => {
                      const entry = result.response.accuracy[name];
                      return (
                        <li key={name}>
                          {OUT_OF_RANGE_LABELS[name] ?? name}: المطلوب {fmt(entry?.requested, 1)}، والممكن في هذا الأفاتار{" "}
                          {fmt(entry?.actual, 1)} {name === "weight" ? "كغ" : "سم"}.
                        </li>
                      );
                    })}
                  </ul>
                  {result.response.out_of_range!.includes("weight") && (
                    <p>
                      الوزن في الأفاتار تقدير من حجم الجسم. لم يُفرض الوزن المطلوب كاملاً لأن ذلك كان سيتطلب تضخيم
                      الذراعين والساقين أو إخراج قياس آخر عن حدود النموذج؛ القياسات المحيطية لها الأولوية.
                    </p>
                  )}
                  {result.response.out_of_range!.includes("shoulder") && (
                    <p>
                      عرض الكتفين المطلوب أكبر من أقصى عرض يمكن تمثيله لهذا الطول، فاستُخدم أقرب عرض ممكن دون تشويه الصدر
                      أو البطن. تأكّد أنك قِست المسافة المستقيمة بين طرفي عظمتي الكتفين، وليس أعرض نقطة فوق العضلات.
                    </p>
                  )}
                </div>
              )}

              {result.response.warnings.length > 0 ? (
                <div className="notice warning">
                  <strong>Fitting warnings</strong>
                  <ul>
                    {result.response.warnings.map((w, i) => (
                      <li key={i}>{w}</li>
                    ))}
                  </ul>
                </div>
              ) : (
                <p className="hint">No fitting warnings.</p>
              )}

              {result.response.failed_measurements.length > 0 && (
                <div className="notice warning">
                  Not measured: {result.response.failed_measurements.join(", ")}
                </div>
              )}

              <table>
                <thead>
                  <tr>
                    <th>Target</th>
                    <th>Requested</th>
                    <th>Actual</th>
                    <th>Error</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.response.accuracy).map(([name, a]) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td>
                        {fmt(a.requested)} {a.unit}
                      </td>
                      <td>
                        {fmt(a.actual)} {a.unit}
                      </td>
                      <td>
                        {fmt(a.absolute_error)} {a.unit} ({fmt(a.percentage_error)} %)
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {result.response.appearance ? (
                <div className="applied">
                  <h2>Appearance applied</h2>
                  <dl>
                    <dt>Face style</dt>
                    <dd>{result.response.appearance.gender_style}</dd>
                    <dt>Skin tone</dt>
                    <dd>
                      <span className="swatch inline" style={{ background: result.response.appearance.skin_tone }} />{" "}
                      <code>{result.response.appearance.skin_tone}</code>
                    </dd>
                    <dt>Eye colour</dt>
                    <dd>{result.response.appearance.eye_color}</dd>
                    <dt>Features</dt>
                    <dd>{result.response.appearance.features.join(", ")}</dd>
                    <dt>Not available</dt>
                    <dd>{result.response.appearance.not_available.join(", ") || "—"}</dd>
                  </dl>
                </div>
              ) : (
                <p className="hint">
                  Requested skin tone: {result.request.skin_tone ?? "backend default"} (this backend did not report the
                  applied appearance).
                </p>
              )}
            </div>
          )}
        </section>

        <section className="stage">
          <AvatarViewer url={viewerUrl} view={view} onStatus={onViewerStatus} />
          <div className="view-buttons">
            <button type="button" className={view === "body" ? "active" : ""} onClick={() => setView("body")}>
              Full body
            </button>
            <button type="button" className={view === "face" ? "active" : ""} onClick={() => setView("face")}>
              Face
            </button>
          </div>
          {viewer.state === "empty" && (
            <div className="overlay" dir="rtl" lang="ar">
              لا يوجد أفاتار بعد. أنشئ أفاتاراً.
            </div>
          )}
          {viewer.state === "loading" && (
            <div className="overlay" dir="rtl" lang="ar">
              {result?.restored ? "جارٍ استرجاع آخر أفاتار…" : "جارٍ تحميل الأفاتار…"}
            </div>
          )}
          {viewer.state === "failed" && (
            <div className="overlay error" dir="rtl" lang="ar">
              تعذّر تحميل ملف الأفاتار: <span dir="ltr">{viewer.error}</span>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

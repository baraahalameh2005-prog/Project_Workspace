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

// Used only until /avatar-options has loaded; the backend validates either way.
const FALLBACK_LIMITS: Record<MeasurementName, MeasurementLimit> = {
  height: { required: true, minimum: 100, maximum: 230, unit: "cm" },
  weight: { required: true, minimum: 25, maximum: 250, unit: "kg" },
  waist: { required: true, minimum: 40, maximum: 200, unit: "cm" },
  chest: { required: true, minimum: 50, maximum: 200, unit: "cm" },
  hip: { required: true, minimum: 50, maximum: 200, unit: "cm" },
};
const FALLBACK_GENDERS = ["female", "male"];
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

// The clothing module's server (NewAvatar/clothing/backend/server.py); its
// try-on page loads the avatar by id from var/generated/.
const CLOTHING_URL = import.meta.env.VITE_CLOTHING_URL ?? "http://127.0.0.1:8010";
// Baraah's Dress3D prototype (the clothes repo): a dress from its photo and
// measurements, on its own in 3D.
const DRESS3D_URL = import.meta.env.VITE_DRESS3D_URL ?? "http://127.0.0.1:8765";
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

// Arabic-Indic (٠-٩) and Persian (۰-۹) digits and the Arabic decimal mark, as typed on
// an Arabic keyboard, read as ordinary numbers.
function toLatinDigits(text: string) {
  return text
    .replace(/[٠-٩]/g, (d) => String(d.charCodeAt(0) - 0x0660))
    .replace(/[۰-۹]/g, (d) => String(d.charCodeAt(0) - 0x06f0))
    .replace(/٫/g, ".");
}

/**
 * Notes about measurements that rarely belong to one adult body (usually a typo or
 * two swapped fields). They do not block generation: the fit would match them and
 * give an odd body, so the user should look first.
 */
function plausibilityNotes(values: Record<MeasurementName, string>): string[] {
  const n = (name: MeasurementName) => Number(toLatinDigits(values[name]).trim());
  const height = n("height"), weight = n("weight"), waist = n("waist"), chest = n("chest"), hip = n("hip");
  if (![height, weight, waist, chest, hip].every((v) => Number.isFinite(v) && v > 0)) return [];

  const notes: string[] = [];
  const bmi = weight / (height / 100) ** 2;
  if (waist > chest && waist > hip) {
    notes.push("Waist is larger than both chest and hip. Check that waist and chest are not swapped.");
  }
  if (chest < height * 0.45) {
    notes.push(`Chest ${chest} cm is very small for a height of ${height} cm (usually at least ${Math.round(height * 0.47)} cm).`);
  }
  if (hip < height * 0.48) {
    notes.push(`Hip ${hip} cm is very small for a height of ${height} cm (usually at least ${Math.round(height * 0.5)} cm).`);
  }
  if (bmi < 20 && waist > height * 0.5) {
    notes.push(`Waist ${waist} cm is large for ${weight} kg at ${height} cm; the avatar will get a belly.`);
  }
  if (bmi < 16 || bmi > 40) {
    notes.push(`Weight ${weight} kg is unusual for ${height} cm (BMI ${bmi.toFixed(1)}); check both values.`);
  }
  return notes;
}

function fmt(value: number | null | undefined, digits = 2) {
  return value === null || value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

export default function App() {
  const [backend, setBackend] = useState<BackendState>("checking");
  const [options, setOptions] = useState<AvatarOptions | null>(null);
  const [optionsError, setOptionsError] = useState<string | null>(null);

  const [values, setValues] = useState(DEFAULT_VALUES);
  const [gender, setGender] = useState("female");
  const [skinTone, setSkinTone] = useState(FALLBACK_SKIN_TONE);
  const [skinToneTouched, setSkinToneTouched] = useState(false);
  const [eyeColor, setEyeColor] = useState(FALLBACK_EYE_COLOR);
  // Athletic is the default for women; it does not apply to men.
  const [bodyStyle, setBodyStyle] = useState("athletic");
  const [fieldErrors, setFieldErrors] = useState<Partial<Record<MeasurementName, string>>>({});

  const [pending, setPending] = useState<{ requestNumber: number; startedAt: number } | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<RequestError | null>(null);
  const [result, setResult] = useState<ShownResult | null>(null);
  const [link] = useState(readLinkedAvatar);
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
  // Only offered when the backend advertises body styles, and only for women.
  const bodyStyles = options?.body_style?.values ?? null;
  const offerBodyStyle = bodyStyles !== null && gender === "female";
  const measurementNotes = plausibilityNotes(values);

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
      const raw = toLatinDigits(values[name]).trim();
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

    setFieldErrors(errors);
    if (Object.keys(errors).length > 0) return null;

    return {
      gender,
      ...parsed,
      skin_tone: skinTone,
      ...(eyeColors ? { eye_color: eyeColor } : {}),
      ...(offerBodyStyle ? { body_style: bodyStyle } : {}),
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
    setPending({ requestNumber, startedAt: Date.now() });

    try {
      const response = await generateAvatar(body, cancel.signal);
      // Ignore responses from requests that are no longer the latest.
      if (requestNumber !== latestRequest.current) return;
      setResult({
        response,
        glbUrl: resolveApiUrl(response.avatar_url),
        receivedAt: new Date(),
        requestNumber,
        request: body,
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

              {offerBodyStyle && (
                <label className="field">
                  <span>Body style</span>
                  <select value={bodyStyle} onChange={(e) => setBodyStyle(e.target.value)}>
                    {bodyStyles!.map((s) => (
                      <option key={s} value={s}>
                        {s === "athletic" ? "athletic (adult, toned)" : s}
                      </option>
                    ))}
                  </select>
                  <small className="hint">
                    {bodyStyle === "athletic"
                      ? "Athletic: an adult, toned body for display (slimmer hips and legs, natural bust). Less exact for sizing."
                      : "Standard: the body follows your measurements as they are. Best for sizing clothes."}
                  </small>
                </label>
              )}

              {MEASUREMENT_NAMES.map((name) => {
                const { minimum, maximum, unit } = limits[name];
                return (
                  <label className="field" key={name}>
                    <span>
                      {LABELS[name]} <small>({unit}, {minimum}–{maximum})</small>
                    </span>
                    <input
                      // Text, not number: a number input drops Arabic-Indic digits.
                      type="text"
                      inputMode="decimal"
                      dir="ltr"
                      value={values[name]}
                      aria-invalid={fieldErrors[name] ? true : undefined}
                      onChange={(e) => setValues((v) => ({ ...v, [name]: e.target.value }))}
                    />
                    {fieldErrors[name] && <em className="field-error">{fieldErrors[name]}</em>}
                  </label>
                );
              })}

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

            {measurementNotes.length > 0 && (
              <div className="notice warning">
                <strong>Please check these measurements</strong>
                <ul>
                  {measurementNotes.map((note) => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
                You can still generate; the avatar will follow the numbers as entered.
              </div>
            )}

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
              {result && <p>The avatar shown is still from request #{result.requestNumber}.</p>}
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
              <h2>Result of request #{result.requestNumber}</h2>
              <dl>
                <dt>Avatar ID</dt>
                <dd>
                  <code>{result.response.avatar_id}</code>
                </dd>
                <dt>Received</dt>
                <dd>{result.receivedAt.toLocaleTimeString()}</dd>
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
              <p>
                <a
                  className="try-on"
                  href={`${CLOTHING_URL}/viewer/?avatar=${encodeURIComponent(result.response.avatar_id)}`}
                >
                  👗 جرّبي الملابس
                </a>{" "}
                <a className="try-on alt" href={DRESS3D_URL} target="_blank" rel="noreferrer">
                  🧵 شوفي الفستان 3D
                </a>
              </p>

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
                    {result.response.body_style && (
                      <>
                        <dt>Body style</dt>
                        <dd>{result.response.body_style}</dd>
                      </>
                    )}
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
          {viewer.state === "empty" && <div className="overlay">No avatar yet. Generate one.</div>}
          {viewer.state === "loading" && <div className="overlay">Loading avatar…</div>}
          {viewer.state === "failed" && (
            <div className="overlay error">
              Could not load the GLB: {viewer.error}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

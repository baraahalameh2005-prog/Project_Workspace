import { useEffect, useMemo, useState } from "react";
import Viewer from "./Viewer";
import {
  fetchSchema,
  fileToDataURL,
  generate,
  urlToDataURL,
  type GenerateResult,
  type MeasurementInput,
  type Schema,
} from "./api";

const GROUPS: { title: string; keys: string[]; note?: string }[] = [
  { title: "Core", keys: ["dress_length", "bust", "waist", "hip", "hem"], note: "dress length and bust are required" },
  {
    title: "Where the girths were measured",
    keys: ["armhole_depth", "waist_position", "hip_position"],
    note: "vertical distance from the high point shoulder (HPS); strongly recommended",
  },
  { title: "Shoulders & neckline", keys: ["shoulder_width", "neck_width", "front_neck_drop", "back_neck_drop"] },
  { title: "Sleeves", keys: ["sleeve_length", "upper_arm", "cuff"] },
];

const LABEL: Record<string, string> = {
  dress_length: "Dress length",
  bust: "Bust / chest",
  waist: "Waist",
  hip: "Hip",
  hem: "Hem (sweep)",
  armhole_depth: "Armhole depth",
  waist_position: "Waist position",
  hip_position: "Hip position",
  shoulder_width: "Shoulder width",
  neck_width: "Neck width",
  front_neck_drop: "Front neck drop",
  back_neck_drop: "Back neck drop",
  sleeve_length: "Sleeve length",
  upper_arm: "Upper arm",
  cuff: "Cuff",
};

interface Example {
  name: string;
  measurements: Record<string, number>;
}

type Inputs = Record<string, MeasurementInput>;

function sourceTag(s: string | undefined) {
  if (!s) return <span className="tag">—</span>;
  if (s === "user") return <span className="tag tag-user">you</span>;
  if (s.startsWith("photo")) return <span className="tag tag-photo" title={s}>photo</span>;
  return <span className="tag tag-default" title="assumed default">default</span>;
}

export default function App() {
  const [schema, setSchema] = useState<Schema | null>(null);
  const [examples, setExamples] = useState<Example[]>([]);
  const [units, setUnits] = useState<"cm" | "in">("cm");
  const [inputs, setInputs] = useState<Inputs>({});
  const [front, setFront] = useState<string | null>(null);
  const [back, setBack] = useState<string | null>(null);
  const [mask, setMask] = useState<string | null>(null);
  const [colourRefine, setColourRefine] = useState(false);
  const [style, setStyle] = useState({ silhouette: "auto", sleeve: "auto", neckline: "auto" });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [res, setRes] = useState<GenerateResult | null>(null);

  useEffect(() => {
    fetchSchema().then(setSchema).catch((e) => setErr(String(e)));
    fetch("/api/examples")
      .then((r) => r.json())
      .then(setExamples)
      .catch(() => undefined);
    // reopen a previous local result: ?id=<result id>
    const id = new URLSearchParams(window.location.search).get("id");
    if (id && /^[0-9a-f]{12}$/.test(id)) {
      fetch(`/api/out/${id}/report.json`)
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`result ${id} not found`))))
        .then(setRes)
        .catch((e) => setErr(String(e)));
    }
  }, []);

  const setVal = (k: string, patch: Partial<MeasurementInput>) =>
    setInputs((p) => ({ ...p, [k]: { ...(p[k] ?? { value: "", method: "circumference" }), ...patch } }));

  async function loadExample(ex: Example) {
    setFront(await urlToDataURL(`/api/examples/${ex.name}/photo_front.png`));
    setBack(null);
    setUnits("cm");
    const next: Inputs = {};
    Object.entries(ex.measurements).forEach(([k, v]) => (next[k] = { value: String(v), method: "circumference" }));
    setInputs(next);
    setRes(null);
  }

  async function onGenerate() {
    setBusy(true);
    setErr(null);
    try {
      const measurements: Record<string, { value: number; method: string }> = {};
      for (const [k, v] of Object.entries(inputs)) {
        if (v.value.trim() === "") continue;
        const n = Number(v.value);
        if (!Number.isFinite(n) || n <= 0) throw new Error(`${LABEL[k] ?? k}: enter a positive number`);
        measurements[k] = { value: n, method: v.method };
      }
      setRes(
        await generate({ front_image: front, back_image: back, mask_image: mask, colour_refine: colourRefine, units, measurements, style }),
      );
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const levels = useMemo(() => {
    if (!res) return [];
    const L = res.landmarks as Record<string, number>;
    return (["y_bust", "y_waist", "y_hip"] as const)
      .filter((k) => typeof L[k] === "number")
      .map((k) => ({ name: k.slice(2), y_cm: L[k] }));
  }, [res]);

  return (
    <div className="app">
      <header>
        <h1>Dress3D prototype</h1>
        <p className="sub">
          Photo + garment measurements → measured 3D dress. Runs entirely on this computer (local server at 127.0.0.1).
        </p>
      </header>
      <main>
        <section className="panel inputs">
          <h2>1 · Photo</h2>
          <div className="photos">
            <PhotoPick label="Front photo (recommended)" value={front} onChange={setFront} />
            <PhotoPick label="Back photo (optional)" value={back} onChange={setBack} />
          </div>
          {examples.length > 0 && (
            <div className="examples">
              <span>Synthetic test cases:</span>
              {examples.map((ex) => (
                <button key={ex.name} className="link" onClick={() => loadExample(ex)}>
                  {ex.name.replace(/_/g, " ")}
                </button>
              ))}
            </div>
          )}
          <p className="small">
            Best results: dress upright on a plain contrasting background, whole garment in frame, photographed straight on.
          </p>
          <details className="seg">
            <summary>Segmentation options (mannequins, hangers)</summary>
            <label className="row">
              <input type="checkbox" checked={colourRefine} onChange={(e) => setColourRefine(e.target.checked)} />
              Remove non-fabric colours (mannequin/stand). Can also remove fabric that looks similar.
            </label>
            <PhotoPick label="Garment mask (optional): white = dress, black = everything else" value={mask} onChange={setMask} />
          </details>

          <h2>2 · Measurements</h2>
          <div className="row">
            <label>
              Units{" "}
              <select value={units} onChange={(e) => setUnits(e.target.value as "cm" | "in")}>
                <option value="cm">cm</option>
                <option value="in">inches (× 2.54)</option>
              </select>
            </label>
          </div>
          {GROUPS.map((g) => (
            <fieldset key={g.title}>
              <legend>{g.title}</legend>
              {g.note && <p className="small">{g.note}</p>}
              {g.keys.map((k) => {
                const def = schema?.measurements[k];
                const v = inputs[k] ?? { value: "", method: "circumference" };
                return (
                  <div className="mrow" key={k}>
                    <label htmlFor={k} title={def?.description}>
                      {LABEL[k] ?? k}
                    </label>
                    <input
                      id={k}
                      inputMode="decimal"
                      placeholder="auto"
                      value={v.value}
                      onChange={(e) => setVal(k, { value: e.target.value })}
                    />
                    {def?.kind === "girth" ? (
                      <select
                        aria-label={`${k} method`}
                        value={v.method}
                        onChange={(e) => setVal(k, { method: e.target.value as MeasurementInput["method"] })}
                      >
                        <option value="circumference">full girth</option>
                        <option value="flat">laid flat (×2)</option>
                      </select>
                    ) : (
                      <span className="unit">{units}</span>
                    )}
                  </div>
                );
              })}
            </fieldset>
          ))}

          <h2>3 · Style (optional overrides)</h2>
          <div className="row wrap">
            {(["silhouette", "sleeve", "neckline"] as const).map((k) => (
              <label key={k}>
                {k}{" "}
                <select value={style[k]} onChange={(e) => setStyle((s) => ({ ...s, [k]: e.target.value }))}>
                  <option value="auto">auto (from photo)</option>
                  {(schema?.[k === "silhouette" ? "silhouettes" : k === "sleeve" ? "sleeves" : "necklines"] ?? []).map((o) => (
                    <option key={o} value={o}>
                      {o}
                    </option>
                  ))}
                </select>
              </label>
            ))}
          </div>

          <button className="primary" onClick={onGenerate} disabled={busy}>
            {busy ? "Generating…" : "Generate 3D dress"}
          </button>
          {err && <p className="error">{err}</p>}
        </section>

        <section className="panel view">
          <Viewer glbUrl={res?.glb_url ?? null} levels={levels} />
          {res && (
            <div className="actions">
              <a className="btn" href={res.glb_url} download={`dress-${res.id}.glb`}>
                Download GLB ({(res.glb_bytes / 1024).toFixed(0)} KB)
              </a>
              <a className="btn" href={res.svg_url} download={`pattern-${res.id}.svg`}>
                Download flat pattern (SVG, approx.)
              </a>
            </div>
          )}
        </section>

        <section className="panel results">
          <h2>Result</h2>
          {!res && <p className="small">The verification table appears after generation.</p>}
          {res && (
            <>
              <p className="method">
                <strong>Method:</strong> {res.method}. Not a neural reconstruction — the geometry is constructed from your
                measurements; the photo provides style cues and the fabric texture.
              </p>
              <p className="small">
                Style: {res.style.silhouette} · sleeves {res.style.sleeve} · {res.style.neckline} neckline ·{" "}
                {res.vertices.toLocaleString("en-US")} vertices / {res.faces.toLocaleString("en-US")} triangles
              </p>
              {res.warnings.length > 0 && (
                <ul className="warn">
                  {res.warnings.map((w) => (
                    <li key={w}>{w}</li>
                  ))}
                </ul>
              )}
              <h3>Dimensions measured back from the exported mesh</h3>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Measurement</th>
                      <th>Used (cm)</th>
                      <th>Mesh</th>
                      <th>Error</th>
                      <th>From</th>
                    </tr>
                  </thead>
                  <tbody>
                    {res.verification.map((r) => (
                      <tr key={r.measurement}>
                        <td title={r.method}>{LABEL[r.measurement] ?? r.measurement}</td>
                        <td className="num">{r.target_cm.toFixed(1)}</td>
                        <td className="num">{r.measured_cm.toFixed(1)}</td>
                        <td className={`num ${Math.abs(r.pct_err) > 1 ? "bad" : ""}`}>
                          {r.abs_err_cm >= 0 ? "+" : ""}
                          {r.abs_err_cm.toFixed(2)} ({r.pct_err.toFixed(2)}%)
                        </td>
                        <td>{sourceTag(res.source[r.measurement])}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="small">
                "photo" and "default" values were not measured by you — they are estimates. Mesh values are re-measured
                from the GLB by plane sections and boundary loops (see README for the method).
              </p>
              {res.mask_url && (
                <details>
                  <summary>Photo segmentation used</summary>
                  <img className="mask" src={res.mask_url} alt="garment mask" />
                </details>
              )}
            </>
          )}
        </section>
      </main>
    </div>
  );
}

function PhotoPick({ label, value, onChange }: { label: string; value: string | null; onChange: (v: string | null) => void }) {
  return (
    <label className="photo">
      <span>{label}</span>
      <div className="thumb">{value ? <img src={value} alt={label} /> : <span>no photo</span>}</div>
      <input
        type="file"
        accept="image/*"
        onChange={async (e) => {
          const f = e.target.files?.[0];
          onChange(f ? await fileToDataURL(f) : null);
        }}
      />
      {value && (
        <button type="button" className="link" onClick={() => onChange(null)}>
          remove
        </button>
      )}
    </label>
  );
}

export type Kind = "girth" | "length";

export interface Schema {
  measurements: Record<string, { kind: Kind; description: string }>;
  silhouettes: string[];
  sleeves: string[];
  necklines: string[];
}

export interface VerificationRow {
  measurement: string;
  target_cm: number;
  measured_cm: number;
  abs_err_cm: number;
  pct_err: number;
  source: string;
  method: string;
}

export interface GenerateResult {
  id: string;
  method: string;
  glb_url: string;
  svg_url: string;
  mask_url: string | null;
  glb_bytes: number;
  vertices: number;
  faces: number;
  style: Record<string, string>;
  source: Record<string, string>;
  measurements_cm: Record<string, number>;
  verification: VerificationRow[];
  landmarks: Record<string, unknown>;
  warnings: string[];
  notes: string[];
  confidence: Record<string, string>;
}

export interface MeasurementInput {
  value: string;
  method: "circumference" | "flat";
}

export interface GenerateRequest {
  front_image: string | null;
  back_image: string | null;
  mask_image: string | null;
  colour_refine: boolean;
  units: "cm" | "in";
  measurements: Record<string, { value: number; method: string } | number>;
  style: Record<string, string>;
}

export async function fetchSchema(): Promise<Schema> {
  const r = await fetch("/api/schema");
  if (!r.ok) throw new Error(`schema: HTTP ${r.status}`);
  return r.json();
}

export async function generate(req: GenerateRequest): Promise<GenerateResult> {
  const r = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error ?? `HTTP ${r.status}`);
  return j;
}

export function fileToDataURL(f: File): Promise<string> {
  return new Promise((res, rej) => {
    const rd = new FileReader();
    rd.onload = () => res(rd.result as string);
    rd.onerror = () => rej(rd.error);
    rd.readAsDataURL(f);
  });
}

export async function urlToDataURL(url: string): Promise<string> {
  const b = await (await fetch(url)).blob();
  return new Promise((res, rej) => {
    const rd = new FileReader();
    rd.onload = () => res(rd.result as string);
    rd.onerror = () => rej(rd.error);
    rd.readAsDataURL(b);
  });
}

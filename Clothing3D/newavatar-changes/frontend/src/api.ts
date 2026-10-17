// Client for the avatar backend (backend/api.py).

export const API_URL: string = (
  (import.meta.env.VITE_API_URL as string | undefined) ?? "http://127.0.0.1:8002"
).replace(/\/+$/, "");

// Matches the agreed synchronous request limit (decision D7).
export const GENERATE_TIMEOUT_MS = 120_000;
const SHORT_TIMEOUT_MS = 5_000;

export const MEASUREMENT_NAMES = ["height", "weight", "waist", "chest", "hip"] as const;
export type MeasurementName = (typeof MEASUREMENT_NAMES)[number];

export interface MeasurementLimit {
  required: boolean;
  minimum: number;
  maximum: number;
  unit: string;
}

export interface AvatarOptions {
  api_version: number;
  measurements: Record<MeasurementName, MeasurementLimit>;
  gender: { required: boolean; values: string[]; default: string; case_insensitive: boolean };
  skin_tone: { required: boolean; format: string; default: string };
  // Present from the appearance version of the backend on.
  eye_color?: { required: boolean; values: string[]; default: string; case_insensitive: boolean };
  appearance?: { automatic: string[]; not_available: string[] };
  // Present from the body-style version of the backend on.
  body_style?: { required: boolean; values: string[]; default: string; case_insensitive: boolean; notes: string };
  notes: string[];
}

export interface AppliedAppearance {
  gender_style: string;
  skin_tone: string;
  eye_color: string;
  face_controls: Record<string, number>;
  features: string[];
  not_available: string[];
}

export interface Health {
  status: string;
  model_loaded: boolean;
}

export interface AvatarRequest {
  gender: string;
  height: number;
  weight: number;
  waist: number;
  chest: number;
  hip: number;
  skin_tone?: string;
  eye_color?: string;
  body_style?: string;
}

export interface AccuracyEntry {
  status?: string;
  requested?: number | null;
  actual?: number | null;
  absolute_error?: number | null;
  percentage_error?: number | null;
  unit?: string;
}

export interface AvatarResponse {
  message: string;
  avatar_url: string;
  avatar_id: string;
  legacy_avatar_url: string | null;
  parameters: Record<string, unknown>;
  appearance?: AppliedAppearance;
  body_style?: string;
  measurements: Record<string, number>;
  failed_measurements: string[];
  accuracy: Record<string, AccuracyEntry>;
  warnings: string[];
  fit_time_seconds: number | null;
}

export type ApiErrorKind = "network" | "timeout" | "cancelled" | "validation" | "http" | "invalid-response";

export class ApiError extends Error {
  constructor(
    public kind: ApiErrorKind,
    message: string,
    public details: string[] = [],
    public status?: number,
  ) {
    super(message);
  }
}

/** Absolute URL for a path the backend returned (avatar_url is relative). */
export function resolveApiUrl(path: string): string {
  return new URL(path, `${API_URL}/`).toString();
}

function formatValidationDetail(detail: unknown): string[] {
  if (typeof detail === "string") return [detail];
  if (!Array.isArray(detail)) return [];

  return detail.map((item) => {
    if (item && typeof item === "object") {
      const { loc, msg } = item as { loc?: unknown[]; msg?: string };
      // loc is like ["body", "height"]; drop the "body" prefix.
      const field = Array.isArray(loc) ? loc.filter((p) => p !== "body").join(".") : "";
      return field ? `${field}: ${msg ?? "invalid"}` : (msg ?? "invalid input");
    }
    return String(item);
  });
}

async function request<T>(
  path: string,
  init: RequestInit,
  timeoutMs: number,
  cancel?: AbortSignal,
): Promise<T> {
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const onCancel = () => controller.abort();
  cancel?.addEventListener("abort", onCancel);

  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, { ...init, signal: controller.signal });
  } catch (err) {
    if (timedOut) {
      throw new ApiError(
        "timeout",
        `No response from the backend within ${Math.round(timeoutMs / 1000)} s. ` +
          "This does not mean generation failed: the backend may still be working " +
          "and may finish writing a GLB, but this page stopped waiting and will not show it.",
      );
    }
    if (cancel?.aborted) {
      throw new ApiError(
        "cancelled",
        "Stopped waiting. The backend may still finish this generation; its result will not be shown.",
      );
    }
    throw new ApiError("network", `Backend not reachable at ${API_URL}. Is the server running on port 8002?`, [
      String(err),
    ]);
  } finally {
    clearTimeout(timer);
    cancel?.removeEventListener("abort", onCancel);
  }

  const text = await response.text();
  let body: unknown = undefined;
  try {
    body = text ? JSON.parse(text) : undefined;
  } catch {
    // Non-JSON body (for example a plain "Internal Server Error").
  }

  if (!response.ok) {
    const detail = body && typeof body === "object" ? (body as { detail?: unknown }).detail : undefined;
    if (response.status === 422) {
      throw new ApiError("validation", "The backend rejected the input.", formatValidationDetail(detail), 422);
    }
    const details = detail !== undefined ? formatValidationDetail(detail) : text ? [text.slice(0, 500)] : [];
    throw new ApiError("http", `Backend error ${response.status} ${response.statusText}`.trim(), details, response.status);
  }

  if (body === undefined) {
    throw new ApiError("invalid-response", "The backend returned a response that is not JSON.", [text.slice(0, 500)]);
  }
  return body as T;
}

export function getHealth(): Promise<Health> {
  return request<Health>("/health", { method: "GET" }, SHORT_TIMEOUT_MS);
}

export function getAvatarOptions(): Promise<AvatarOptions> {
  return request<AvatarOptions>("/avatar-options", { method: "GET" }, SHORT_TIMEOUT_MS);
}

export async function generateAvatar(body: AvatarRequest, cancel?: AbortSignal): Promise<AvatarResponse> {
  const result = await request<AvatarResponse>(
    "/generate-avatar",
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) },
    GENERATE_TIMEOUT_MS,
    cancel,
  );
  if (typeof result.avatar_url !== "string" || !result.avatar_url) {
    throw new ApiError("invalid-response", "The response has no avatar_url.");
  }
  return result;
}

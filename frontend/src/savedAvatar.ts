// The last successfully generated avatar, kept so it survives navigation and reloads.
// Only the reference (avatar_id, avatar_url) and the small generation response are
// stored; the GLB itself stays on the backend in var/generated/.

import type { AvatarRequest, AvatarResponse } from "./api";

export const SAVED_AVATAR_KEY = "baraha-avatar:last";
const SAVED_AVATAR_VERSION = 1;

const AVATAR_ID = /^[0-9a-f]{32}$/;

export interface SavedAvatar {
  version: number;
  // Backend the GLB belongs to; a record from another backend is not restored.
  apiUrl: string;
  savedAt: string;
  request: AvatarRequest;
  response: AvatarResponse;
}

/** A record is usable only if its id and GLB path agree and it belongs to this backend. */
export function parseSavedAvatar(raw: string | null, apiUrl: string): SavedAvatar | null {
  if (!raw) return null;
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!data || typeof data !== "object") return null;
  const saved = data as Partial<SavedAvatar>;
  const response = saved.response;
  if (
    saved.version !== SAVED_AVATAR_VERSION ||
    saved.apiUrl !== apiUrl ||
    typeof saved.savedAt !== "string" ||
    !saved.request ||
    typeof saved.request !== "object" ||
    !response ||
    typeof response !== "object" ||
    typeof response.avatar_id !== "string" ||
    !AVATAR_ID.test(response.avatar_id) ||
    response.avatar_url !== `/models/generated/${response.avatar_id}.glb`
  ) {
    return null;
  }
  return saved as SavedAvatar;
}

export function loadSavedAvatar(apiUrl: string): SavedAvatar | null {
  try {
    return parseSavedAvatar(window.localStorage.getItem(SAVED_AVATAR_KEY), apiUrl);
  } catch {
    // Storage blocked (private window, site data disabled).
    return null;
  }
}

/** Returns false when the browser refused to store it. */
export function storeSavedAvatar(apiUrl: string, request: AvatarRequest, response: AvatarResponse): boolean {
  const record: SavedAvatar = {
    version: SAVED_AVATAR_VERSION,
    apiUrl,
    savedAt: new Date().toISOString(),
    request,
    response,
  };
  try {
    window.localStorage.setItem(SAVED_AVATAR_KEY, JSON.stringify(record));
    return true;
  } catch {
    return false;
  }
}

export function clearSavedAvatar() {
  try {
    window.localStorage.removeItem(SAVED_AVATAR_KEY);
  } catch {
    // Nothing stored or storage blocked.
  }
}

/** The fitting room's address for this avatar (Clothing3D backend/server.py reads it from var/generated/). */
export function fittingRoomUrl(clothingUrl: string, avatarId: string): string {
  if (!AVATAR_ID.test(avatarId)) throw new Error(`Not a generated avatar id: ${avatarId}`);
  return `${clothingUrl.replace(/\/+$/, "")}/viewer/?avatar=${avatarId}`;
}

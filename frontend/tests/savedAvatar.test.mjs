// Tests for src/savedAvatar.ts. Run from frontend/ (Node 22.18+ strips the TypeScript types):
//   node --test tests/savedAvatar.test.mjs
import assert from "node:assert/strict";
import { beforeEach, test } from "node:test";

// Minimal localStorage stand-in; installed before the module is used.
const store = new Map();
globalThis.window = {
  localStorage: {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: (k) => store.delete(k),
  },
};

const { SAVED_AVATAR_KEY, clearSavedAvatar, fittingRoomUrl, loadSavedAvatar, parseSavedAvatar, storeSavedAvatar } =
  await import("../src/savedAvatar.ts");

const API = "http://127.0.0.1:8002";
const ID = "5ed1a4bb4feb465c87990fe957fdde50";
const REQUEST = { gender: "male", height: 178, weight: 78, waist: 86, chest: 100, hip: 98, skin_tone: "#c8997e" };
const RESPONSE = { avatar_id: ID, avatar_url: `/models/generated/${ID}.glb`, warnings: [], accuracy: {} };

beforeEach(() => store.clear());

test("stores only the reference and response, and restores the same avatar", () => {
  assert.equal(storeSavedAvatar(API, REQUEST, RESPONSE), true);
  const raw = store.get(SAVED_AVATAR_KEY);
  assert.ok(raw.length < 10_000, "no model data in storage");
  const saved = loadSavedAvatar(API);
  assert.equal(saved.response.avatar_id, ID);
  assert.equal(saved.response.avatar_url, `/models/generated/${ID}.glb`);
  assert.deepEqual(saved.request, REQUEST);
});

test("the male shoulder width is kept with the saved request", () => {
  storeSavedAvatar(API, { ...REQUEST, shoulder: 42 }, { ...RESPONSE, fit_method: "adult_male" });
  const saved = loadSavedAvatar(API);
  assert.equal(saved.request.shoulder, 42);
  assert.equal(saved.response.fit_method, "adult_male");
});

test("a newer generation replaces the older one", () => {
  storeSavedAvatar(API, REQUEST, RESPONSE);
  const other = "0123456789abcdef0123456789abcdef";
  storeSavedAvatar(API, REQUEST, { ...RESPONSE, avatar_id: other, avatar_url: `/models/generated/${other}.glb` });
  assert.equal(loadSavedAvatar(API).response.avatar_id, other);
});

test("rejects a record whose id and GLB path disagree (stale or unrelated model)", () => {
  storeSavedAvatar(API, REQUEST, { ...RESPONSE, avatar_url: "/models/generated/0123456789abcdef0123456789abcdef.glb" });
  assert.equal(loadSavedAvatar(API), null);
});

test("rejects records from another backend, other versions, bad ids and broken JSON", () => {
  storeSavedAvatar("http://127.0.0.1:9999", REQUEST, RESPONSE);
  assert.equal(loadSavedAvatar(API), null);

  const good = JSON.parse(JSON.stringify({ version: 1, apiUrl: API, savedAt: "x", request: REQUEST, response: RESPONSE }));
  assert.ok(parseSavedAvatar(JSON.stringify(good), API));
  assert.equal(parseSavedAvatar(JSON.stringify({ ...good, version: 2 }), API), null);
  assert.equal(
    parseSavedAvatar(JSON.stringify({ ...good, response: { ...RESPONSE, avatar_id: "../x", avatar_url: "/models/generated/../x.glb" } }), API),
    null,
  );
  assert.equal(parseSavedAvatar("{not json", API), null);
  assert.equal(parseSavedAvatar(null, API), null);
});

test("clear removes it", () => {
  storeSavedAvatar(API, REQUEST, RESPONSE);
  clearSavedAvatar();
  assert.equal(loadSavedAvatar(API), null);
});

test("blocked storage is reported, not thrown", () => {
  const original = globalThis.window.localStorage;
  globalThis.window.localStorage = {
    getItem() {
      throw new Error("SecurityError");
    },
    setItem() {
      throw new Error("QuotaExceededError");
    },
    removeItem() {
      throw new Error("SecurityError");
    },
  };
  try {
    assert.equal(storeSavedAvatar(API, REQUEST, RESPONSE), false);
    assert.equal(loadSavedAvatar(API), null);
    clearSavedAvatar();
  } finally {
    globalThis.window.localStorage = original;
  }
});

test("fitting room URL carries exactly this avatar id", () => {
  assert.equal(fittingRoomUrl("http://127.0.0.1:8010/", ID), `http://127.0.0.1:8010/viewer/?avatar=${ID}`);
  assert.throws(() => fittingRoomUrl("http://127.0.0.1:8010", "female_plus1"));
});

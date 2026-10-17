"""Minimal local HTTP JSON API using only the standard library.

Endpoints:
    GET  /            -> the web UI (fit_engine/web/index.html), which calls POST /recommend
    GET  /health      -> {"status": "ok", "engine_version": ...}
    GET  /schema      -> supported categories, fit types, stretch levels, measurement keys
    POST /recommend   -> body: {"body": {...}, "garment": {...}}  -> recommendation result
    POST /avatar-to-body -> body: {"avatar": <avatar API request or response>}
                         -> {"body": <fit_engine body document>, "notes": [...]} (adapters.py)

Intended for local development and future integration testing only.
It binds to 127.0.0.1 by default and has no authentication.
"""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import rules, schema
from .adapters import avatar_to_body
from .engine import ENGINE_VERSION, recommend_size

MAX_BODY_BYTES = 1_000_000
UI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "index.html")


def schema_info():
    return {
        "engine_version": ENGINE_VERSION,
        "unit": "cm",
        "accepted_units": ["cm", "mm", "m", "in"],
        "categories": list(rules.SUPPORTED_CATEGORIES),
        "fit_types": list(rules.SUPPORTED_FIT_TYPES),
        "stretch_levels": list(rules.SUPPORTED_STRETCH),
        "body_measurements": list(schema.BODY_RANGES),
        "garment_measurements": list(schema.GARMENT_RANGES),
        "category_rules": rules.CATEGORY_RULES,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "fit_engine/" + ENGINE_VERSION

    def _send(self, code, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):  # CORS preflight for a local web front-end
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _send_ui(self):
        try:
            with open(UI_PATH, "rb") as f:
                data = f.read()
        except OSError:
            self._send(500, {"error": "ui_not_found"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send_ui()
        elif path == "/health":
            self._send(200, {"status": "ok", "engine_version": ENGINE_VERSION})
        elif path == "/schema":
            self._send(200, schema_info())
        else:
            self._send(404, {"error": "not_found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path not in ("/recommend", "/avatar-to-body"):
            self._send(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send(400, {"error": "invalid_request_size"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send(400, {"error": "invalid_json"})
            return
        if path == "/avatar-to-body":
            if not isinstance(payload, dict) or not isinstance(payload.get("avatar"), dict):
                self._send(400, {"error": "expected {\"avatar\": {...}}"})
                return
            body, notes = avatar_to_body(payload["avatar"])
            self._send(200, {"body": body, "notes": notes})
            return
        if not isinstance(payload, dict) or "body" not in payload or "garment" not in payload:
            self._send(400, {"error": "expected {\"body\": {...}, \"garment\": {...}}"})
            return
        res = recommend_size(payload["body"], payload["garment"])
        self._send(422 if res["status"] == "invalid_input" else 200, res)

    def log_message(self, fmt, *args):  # keep test output quiet
        pass


def make_server(host="127.0.0.1", port=8765):
    return ThreadingHTTPServer((host, port), Handler)


def serve(host="127.0.0.1", port=8765):
    httpd = make_server(host, port)
    print(f"fit_engine: UI on http://{host}:{port}/  API: POST /recommend, GET /health, GET /schema")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

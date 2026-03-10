"""Settings server — FastAPI/uvicorn for local config management.

Serves a web UI for managing recognition settings, camera preview
(MJPEG stream), and system status. Runs on localhost only.
"""

import json
import os
import sys
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn


class SettingsUpdate(BaseModel):
    confidence_threshold: float | None = None
    command_confidence_threshold: float | None = None
    fingerspell_hold_ms: int | None = None
    cooldown_ms: int | None = None
    camera_index: int | None = None
    inactivity_timeout_seconds: int | None = None


class GestureBinding(BaseModel):
    gesture_name: str
    command_type: str
    command_value: str


# ── Shared camera reference for MJPEG ────────────────────────────────

_camera_ref = None  # Set by run_settings_server


def create_settings_app(config_path: str = None) -> FastAPI:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if config_path is None:
        config_path = os.path.join(base_dir, "config.json")

    app = FastAPI(title="SignType Settings", version="1.0.0")

    def _load_config() -> dict:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)
        return {"settings": {}, "gestures": {}}

    def _save_config(config: dict):
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    # ── Web UI ─────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def settings_page():
        return _SETTINGS_HTML

    # ── MJPEG camera stream ──────────────────────────────────────

    @app.get("/api/camera/stream")
    async def camera_stream():
        """MJPEG stream of the camera feed."""
        if _camera_ref is None:
            raise HTTPException(503, "Camera not available")

        def generate():
            import cv2
            while True:
                frame = _camera_ref.read(timeout=1.0)
                if frame is None:
                    continue
                # Resize for web preview
                h, w = frame.shape[:2]
                scale = 320 / max(w, 1)
                small = cv2.resize(frame, (320, int(h * scale)))
                _, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    jpeg.tobytes() + b"\r\n"
                )
                time.sleep(0.066)  # ~15 FPS for web

        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    # ── API endpoints ────────────────────────────────────────────

    @app.get("/api/settings")
    async def get_settings():
        return _load_config()

    @app.put("/api/settings")
    async def update_settings(update: SettingsUpdate):
        config = _load_config()
        if "settings" not in config:
            config["settings"] = {}
        for field, value in update.model_dump(exclude_none=True).items():
            config["settings"][field] = value
        _save_config(config)
        return {"status": "ok", "settings": config["settings"]}

    @app.get("/api/gestures")
    async def get_gestures():
        config = _load_config()
        return {"gestures": config.get("gestures", {})}

    @app.post("/api/gestures")
    async def add_gesture(binding: GestureBinding):
        config = _load_config()
        if "gestures" not in config:
            config["gestures"] = {}
        config["gestures"][binding.gesture_name] = {
            "type": binding.command_type,
            "value": binding.command_value,
        }
        _save_config(config)
        return {"status": "ok", "gesture": binding.gesture_name}

    @app.delete("/api/gestures/{gesture_name}")
    async def delete_gesture(gesture_name: str):
        config = _load_config()
        gestures = config.get("gestures", {})
        if gesture_name not in gestures:
            raise HTTPException(404, f"Gesture '{gesture_name}' not found")
        del gestures[gesture_name]
        config["gestures"] = gestures
        _save_config(config)
        return {"status": "ok"}

    @app.get("/api/status")
    async def get_status():
        return {
            "running": True,
            "camera": _camera_ref is not None,
            "models": {
                "fingerspell": os.path.exists(
                    os.path.join(base_dir, "data", "model_fingerspell.pkl")
                ),
                "dynamic": os.path.exists(
                    os.path.join(base_dir, "data", "model_dynamic.pt")
                ),
                "static_gesture": os.path.exists(
                    os.path.join(base_dir, "data", "model_static.pkl")
                ),
            },
        }

    @app.post("/api/test-typing")
    async def test_typing():
        """Type 'hello' into the focused app as a test."""
        import subprocess
        try:
            subprocess.run(["ydotool", "type", "--", "hello"], check=True, timeout=3)
            return {"status": "ok", "typed": "hello"}
        except Exception as e:
            raise HTTPException(500, str(e))

    return app


def run_settings_server(config_path: str = None, port: int = 7842,
                        camera=None):
    """Run the settings server in a background thread."""
    global _camera_ref
    _camera_ref = camera

    app = create_settings_app(config_path)

    def _run():
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


# ── Settings HTML ────────────────────────────────────────────────────

_SETTINGS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SignType Settings</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0d0d1a;
            --surface: rgba(22, 28, 48, 0.85);
            --surface-hover: rgba(30, 38, 62, 0.9);
            --primary: #6c5ce7;
            --primary-dim: rgba(108, 92, 231, 0.15);
            --accent: #fd79a8;
            --success: #00b894;
            --warning: #fdcb6e;
            --danger: #d63031;
            --text: #f0f0f5;
            --text-dim: #8a8a9a;
            --border: rgba(255, 255, 255, 0.06);
            --radius: 14px;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            padding: 2rem;
            background-image:
                radial-gradient(ellipse at 20% 50%, rgba(108, 92, 231, 0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(253, 121, 168, 0.04) 0%, transparent 50%);
        }

        .header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 2rem;
        }
        .header h1 {
            font-size: 1.6rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header .subtitle {
            color: var(--text-dim);
            font-size: 0.85rem;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            max-width: 900px;
        }
        @media (max-width: 700px) {
            .grid { grid-template-columns: 1fr; }
        }

        .card {
            background: var(--surface);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: var(--radius);
            padding: 1.5rem;
            border: 1px solid var(--border);
            transition: border-color 0.3s;
        }
        .card:hover { border-color: rgba(255,255,255,0.1); }
        .card.full-width { grid-column: 1 / -1; }

        .card h2 {
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 1.2rem;
        }

        .field {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.65rem 0;
            border-bottom: 1px solid var(--border);
        }
        .field:last-of-type { border-bottom: none; }
        .field label {
            color: var(--text);
            font-size: 0.88rem;
            font-weight: 500;
        }
        .field .hint {
            font-size: 0.75rem;
            color: var(--text-dim);
            margin-top: 2px;
        }

        input[type="number"], input[type="text"], select {
            background: rgba(255,255,255,0.04);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.5rem 0.75rem;
            width: 100px;
            text-align: right;
            font-family: 'Inter', sans-serif;
            font-size: 0.88rem;
            transition: border-color 0.2s;
        }
        input:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px var(--primary-dim);
        }

        .btn {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.55rem 1.2rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            font-weight: 600;
            transition: all 0.2s;
        }
        .btn-primary {
            background: linear-gradient(135deg, var(--primary), #8b5cf6);
            color: white;
        }
        .btn-primary:hover { opacity: 0.9; transform: translateY(-1px); }
        .btn-secondary {
            background: rgba(255,255,255,0.06);
            color: var(--text);
        }
        .btn-secondary:hover { background: rgba(255,255,255,0.1); }
        .btn-success {
            background: var(--success);
            color: white;
        }

        .actions {
            margin-top: 1.2rem;
            display: flex;
            gap: 0.6rem;
            justify-content: flex-end;
        }

        /* Status indicators */
        .status-row {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.5rem 0;
        }
        .status-dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .status-dot.ok { background: var(--success); box-shadow: 0 0 6px var(--success); }
        .status-dot.missing { background: var(--danger); box-shadow: 0 0 6px var(--danger); }

        /* Camera preview */
        .camera-preview {
            border-radius: 10px;
            overflow: hidden;
            background: #000;
            text-align: center;
            margin-bottom: 1rem;
        }
        .camera-preview img {
            width: 100%;
            max-width: 320px;
            border-radius: 10px;
        }

        /* Toast */
        .toast {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background: var(--surface);
            backdrop-filter: blur(20px);
            color: var(--text);
            padding: 0.8rem 1.4rem;
            border-radius: 10px;
            border: 1px solid var(--border);
            display: none;
            z-index: 1000;
            font-size: 0.88rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        }
        .toast.success { border-left: 3px solid var(--success); }
        .toast.error { border-left: 3px solid var(--danger); }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>🤟 SignType Settings</h1>
            <p class="subtitle">Configure ASL recognition — all feedback is visual</p>
        </div>
    </div>

    <div class="grid">
        <!-- Camera Preview -->
        <div class="card">
            <h2>📷 Camera Preview</h2>
            <div class="camera-preview">
                <img src="/api/camera/stream" alt="Camera feed"
                     onerror="this.alt='Camera not available'; this.style.padding='3rem';">
            </div>
            <div class="actions">
                <button class="btn btn-success" onclick="testTyping()">⌨ Test Typing</button>
            </div>
        </div>

        <!-- System Status -->
        <div class="card">
            <h2>📊 System Status</h2>
            <div id="status-list"></div>
        </div>

        <!-- Recognition Settings -->
        <div class="card">
            <h2>⚙ Recognition</h2>
            <div class="field">
                <div>
                    <label>Confidence threshold</label>
                    <div class="hint">Min certainty to accept a letter</div>
                </div>
                <input type="number" id="confidence_threshold" step="0.05" min="0.5" max="1.0">
            </div>
            <div class="field">
                <div>
                    <label>Hold time (ms)</label>
                    <div class="hint">How long to hold a sign</div>
                </div>
                <input type="number" id="fingerspell_hold_ms" step="50" min="100" max="1000">
            </div>
            <div class="field">
                <div>
                    <label>Cooldown (ms)</label>
                    <div class="hint">Delay between same letter</div>
                </div>
                <input type="number" id="cooldown_ms" step="50" min="200" max="2000">
            </div>
            <div class="field">
                <div>
                    <label>Camera index</label>
                    <div class="hint">0 = default webcam</div>
                </div>
                <input type="number" id="camera_index" min="0" max="10">
            </div>
            <div class="field">
                <div>
                    <label>Inactivity timeout (s)</label>
                    <div class="hint">Auto-idle after no hands</div>
                </div>
                <input type="number" id="inactivity_timeout_seconds" min="10" max="600">
            </div>
            <div class="actions">
                <button class="btn btn-primary" onclick="saveSettings()">Save</button>
            </div>
        </div>

        <!-- Gesture Bindings -->
        <div class="card">
            <h2>🤏 Gesture Bindings</h2>
            <div id="gesture-list"></div>
            <div class="actions">
                <button class="btn btn-secondary" onclick="refreshGestures()">Refresh</button>
            </div>
        </div>
    </div>

    <div class="toast" id="toast"></div>

    <script>
        async function loadSettings() {
            const resp = await fetch('/api/settings');
            const data = await resp.json();
            const s = data.settings || {};
            for (const [key, val] of Object.entries(s)) {
                const el = document.getElementById(key);
                if (el) el.value = val;
            }
        }

        async function saveSettings() {
            const fields = ['confidence_threshold', 'fingerspell_hold_ms',
                          'cooldown_ms', 'camera_index', 'inactivity_timeout_seconds'];
            const payload = {};
            for (const f of fields) {
                const el = document.getElementById(f);
                if (el && el.value) payload[f] = Number(el.value);
            }
            const resp = await fetch('/api/settings', {
                method: 'PUT',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload),
            });
            showToast(resp.ok ? 'Settings saved ✓' : 'Save failed', resp.ok);
        }

        async function loadStatus() {
            try {
                const resp = await fetch('/api/status');
                const data = await resp.json();
                const list = document.getElementById('status-list');
                const models = data.models || {};
                let html = '';

                html += statusRow('Camera', data.camera);
                html += statusRow('Fingerspell model', models.fingerspell);
                html += statusRow('Dynamic model', models.dynamic);
                html += statusRow('Static gesture model', models.static_gesture);

                list.innerHTML = html;
            } catch(e) {
                console.error(e);
            }
        }

        function statusRow(label, ok) {
            const cls = ok ? 'ok' : 'missing';
            const text = ok ? 'Ready' : 'Not found';
            return `<div class="status-row">
                <span class="status-dot ${cls}"></span>
                <span style="flex:1">${label}</span>
                <span style="color:var(--text-dim);font-size:0.8rem">${text}</span>
            </div>`;
        }

        async function refreshGestures() {
            const resp = await fetch('/api/gestures');
            const data = await resp.json();
            const list = document.getElementById('gesture-list');
            const gestures = data.gestures || {};
            if (!Object.keys(gestures).length) {
                list.innerHTML = '<p style="color:var(--text-dim);font-size:0.85rem">No gesture bindings configured</p>';
                return;
            }
            let html = '';
            for (const [name, binding] of Object.entries(gestures)) {
                html += `<div class="status-row">
                    <span style="color:var(--success);font-weight:600">${name}</span>
                    <span style="color:var(--text-dim);font-size:0.8rem">${binding.type}: ${binding.value}</span>
                </div>`;
            }
            list.innerHTML = html;
        }

        async function testTyping() {
            showToast('Click a text field within 3 seconds...', true);
            setTimeout(async () => {
                const resp = await fetch('/api/test-typing', { method: 'POST' });
                const data = await resp.json();
                showToast(resp.ok ? `Typed "${data.typed}" ✓` : 'Test failed', resp.ok);
            }, 3000);
        }

        function showToast(msg, success = true) {
            const t = document.getElementById('toast');
            t.textContent = msg;
            t.className = 'toast ' + (success ? 'success' : 'error');
            t.style.display = 'block';
            setTimeout(() => t.style.display = 'none', 3000);
        }

        loadSettings();
        loadStatus();
        refreshGestures();
        setInterval(loadStatus, 10000);
    </script>
</body>
</html>"""


if __name__ == "__main__":
    app = create_settings_app()
    print("Starting settings server on http://127.0.0.1:7842")
    uvicorn.run(app, host="127.0.0.1", port=7842)

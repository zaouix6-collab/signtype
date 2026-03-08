"""Settings server — FastAPI/uvicorn for local config management.

Serves a web UI for managing gesture bindings, recording new gestures,
and configuring system settings. Runs on localhost only.
"""

import json
import os
import sys
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn


class SettingsUpdate(BaseModel):
    """Settings update payload."""
    confidence_threshold: float | None = None
    command_confidence_threshold: float | None = None
    fingerspell_hold_ms: int | None = None
    camera_index: int | None = None
    inactivity_timeout_seconds: int | None = None
    settings_port: int | None = None


class GestureBinding(BaseModel):
    """A gesture-to-command binding."""
    gesture_name: str
    command_type: str  # "shell", "hotkey", "launch", "url", "file"
    command_value: str


def create_settings_app(config_path: str = None) -> FastAPI:
    """Create the FastAPI settings application.

    Args:
        config_path: Path to config.json.

    Returns:
        FastAPI application instance.
    """
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

    @app.get("/", response_class=HTMLResponse)
    async def settings_page():
        """Serve the settings web UI."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SignType Settings</title>
    <style>
        :root {
            --bg: #1a1a2e;
            --surface: #16213e;
            --primary: #0f3460;
            --accent: #e94560;
            --text: #eee;
            --text-dim: #aaa;
            --success: #2ecc71;
            --border: #333;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            padding: 2rem;
        }
        h1 {
            color: var(--accent);
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            color: var(--text-dim);
            font-size: 0.9rem;
            margin-bottom: 2rem;
        }
        .card {
            background: var(--surface);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid var(--border);
        }
        .card h2 {
            color: var(--accent);
            font-size: 1.2rem;
            margin-bottom: 1rem;
        }
        .field {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .field:last-child { border-bottom: none; }
        .field label {
            color: var(--text-dim);
            font-size: 0.9rem;
        }
        .field input, .field select {
            background: var(--primary);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.4rem 0.8rem;
            width: 120px;
            text-align: right;
        }
        .field input:focus, .field select:focus {
            outline: none;
            border-color: var(--accent);
        }
        button {
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            cursor: pointer;
            font-size: 0.9rem;
            transition: opacity 0.2s;
        }
        button:hover { opacity: 0.85; }
        .btn-secondary {
            background: var(--primary);
            margin-right: 0.5rem;
        }
        .actions { margin-top: 1rem; text-align: right; }
        .gesture-list { list-style: none; }
        .gesture-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .gesture-name { font-weight: bold; color: var(--success); }
        .gesture-cmd { color: var(--text-dim); font-size: 0.85rem; }
        .toast {
            position: fixed; bottom: 2rem; right: 2rem;
            background: var(--success); color: white;
            padding: 0.8rem 1.5rem; border-radius: 8px;
            display: none; z-index: 1000;
        }
        .toast.error { background: var(--accent); }
    </style>
</head>
<body>
    <h1>🤟 SignType Settings</h1>
    <p class="subtitle">Configure your ASL recognition system — all feedback is visual</p>

    <div class="card" id="settings-card">
        <h2>Recognition Settings</h2>
        <div class="field">
            <label>Confidence threshold</label>
            <input type="number" id="confidence_threshold" step="0.05" min="0.5" max="1.0">
        </div>
        <div class="field">
            <label>Command confidence threshold</label>
            <input type="number" id="command_confidence_threshold" step="0.05" min="0.5" max="1.0">
        </div>
        <div class="field">
            <label>Hold time (ms)</label>
            <input type="number" id="fingerspell_hold_ms" step="50" min="100" max="1000">
        </div>
        <div class="field">
            <label>Camera index</label>
            <input type="number" id="camera_index" min="0" max="10">
        </div>
        <div class="field">
            <label>Inactivity timeout (s)</label>
            <input type="number" id="inactivity_timeout_seconds" min="10" max="600">
        </div>
        <div class="actions">
            <button onclick="saveSettings()">Save Settings</button>
        </div>
    </div>

    <div class="card">
        <h2>Gesture Bindings</h2>
        <ul class="gesture-list" id="gesture-list"></ul>
        <div class="actions">
            <button class="btn-secondary" onclick="refreshGestures()">Refresh</button>
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
            const fields = ['confidence_threshold', 'command_confidence_threshold',
                          'fingerspell_hold_ms', 'camera_index', 'inactivity_timeout_seconds'];
            const payload = {};
            for (const f of fields) {
                const el = document.getElementById(f);
                if (el && el.value) payload[f] = Number(el.value);
            }
            const resp = await fetch('/api/settings', {
                method: 'PUT', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload),
            });
            if (resp.ok) showToast('Settings saved');
            else showToast('Save failed', true);
        }

        async function refreshGestures() {
            const resp = await fetch('/api/gestures');
            const data = await resp.json();
            const list = document.getElementById('gesture-list');
            list.innerHTML = '';
            for (const [name, binding] of Object.entries(data.gestures || {})) {
                const li = document.createElement('li');
                li.className = 'gesture-item';
                li.innerHTML = `<span class="gesture-name">${name}</span>
                               <span class="gesture-cmd">${binding.type}: ${binding.value}</span>`;
                list.appendChild(li);
            }
            if (!Object.keys(data.gestures || {}).length) {
                list.innerHTML = '<li style="color:var(--text-dim)">No gesture bindings configured</li>';
            }
        }

        function showToast(msg, isError = false) {
            const t = document.getElementById('toast');
            t.textContent = msg;
            t.className = 'toast' + (isError ? ' error' : '');
            t.style.display = 'block';
            setTimeout(() => t.style.display = 'none', 2500);
        }

        loadSettings();
        refreshGestures();
    </script>
</body>
</html>"""

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
        """Return system status — useful for overlay integration."""
        return {
            "running": True,
            "mode": "idle",
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

    return app


def run_settings_server(config_path: str = None, port: int = 7842):
    """Run the settings server in a background thread."""
    app = create_settings_app(config_path)

    def _run():
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    print(f"[SignType] Settings server at http://127.0.0.1:{port}")
    return thread


if __name__ == "__main__":
    app = create_settings_app()
    print("Starting settings server on http://127.0.0.1:7842")
    uvicorn.run(app, host="127.0.0.1", port=7842)

"""SignType — Main entry point.

Orchestrates camera capture, hand landmark extraction, ASL classification,
and direct keystroke injection into any focused application.

All feedback is visual. No audio output — designed for deaf/HoH users.
"""

# ── Suppress noisy warnings before any imports ──────────────────────────
import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # TensorFlow: errors only
os.environ["GLOG_minloglevel"] = "3"               # MediaPipe glog
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"           # CPU only
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Layer-shell must be preloaded before GTK/Wayland libs
_LAYER_SHELL_PATH = "/usr/lib/libgtk4-layer-shell.so"
if os.path.exists(_LAYER_SHELL_PATH):
    existing = os.environ.get("LD_PRELOAD", "")
    if _LAYER_SHELL_PATH not in existing:
        os.environ["LD_PRELOAD"] = (
            f"{_LAYER_SHELL_PATH}:{existing}" if existing else _LAYER_SHELL_PATH
        )

import queue
import threading
import time
import json
import signal

# Redirect stderr briefly to suppress MediaPipe/absl init spam
import io
_orig_stderr = sys.stderr
sys.stderr = io.StringIO()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.camera import Camera
from core.landmark_extractor import LandmarkExtractor
from core.state_machine import StateMachine, State, GestureEvent
from core.fingerspell_classifier import FingerspellClassifier
from core.gesture_classifier import GestureClassifier
from core.dynamic_classifier import DynamicClassifier
from core.text_injector import TextInjector
from core.command_dispatcher import CommandDispatcher
from feedback.buffer_overlay import BufferOverlay
from feedback.audio import VisualFeedback
from feedback.tray import TrayIcon
from settings.server import run_settings_server

# Restore stderr after imports
sys.stderr = _orig_stderr


# ── Dependency validation ─────────────────────────────────────────────

def _check_dependencies() -> list[str]:
    """Check for missing system dependencies. Returns list of issues."""
    issues = []

    # Check ydotool (needed for typing into apps)
    import subprocess
    try:
        subprocess.run(["which", "ydotool"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        issues.append("ydotool not found — install with: sudo pacman -S ydotool")

    # Check if camera is accessible
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            issues.append("No camera found — check that your webcam is connected")
        cap.release()
    except Exception:
        issues.append("OpenCV could not access the camera")

    return issues


class SignTypeApp:
    """Main SignType application — wires all components together."""

    def __init__(self, config_path: str = ""):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = config_path or os.path.join(base_dir, "config.json")
        self.config = self._load_config()
        self.data_dir = os.path.join(base_dir, "data")

        # Shared event queue for the state machine
        self.event_queue = queue.Queue()

        # --- Core components ---
        settings = self.config.get("settings", {})
        cam_index = settings.get("camera_index", 0)

        # Camera with graceful error handling
        try:
            self.camera = Camera(camera_index=cam_index)
        except Exception as e:
            print(f"  ⚠ Camera error (index {cam_index}): {e}")
            print("    Trying index 0...")
            self.camera = Camera(camera_index=0)

        self.extractor = LandmarkExtractor()
        self.state_machine = StateMachine(
            self.event_queue,
            inactivity_timeout=settings.get("inactivity_timeout_seconds", 60),
        )

        # --- Classifiers ---
        self.fingerspell = FingerspellClassifier(
            os.path.join(self.data_dir, "model_fingerspell.pkl")
        )
        self.gesture_clf = GestureClassifier(
            os.path.join(self.data_dir, "model_static.pkl")
        )
        dyn_path = os.path.join(self.data_dir, "model_dynamic.pt")
        self.dynamic_clf = DynamicClassifier(model_path=dyn_path)

        # --- Feedback (all visual) ---
        self.visual_feedback = VisualFeedback()
        self.overlay = BufferOverlay()
        self.tray = TrayIcon()

        # --- Action handlers ---
        hold_ms = settings.get("fingerspell_hold_ms", 400)
        cooldown_ms = settings.get("cooldown_ms", 600)
        self.text_injector = TextInjector(
            fingerspell_hold_ms=hold_ms, cooldown_ms=cooldown_ms
        )
        self.command_dispatcher = CommandDispatcher(self.config_path)

        # --- Wiring ---
        self.state_machine.on_state_change = self._on_state_change

        self.visual_feedback.on_notification = lambda n: self.overlay.show_notification(
            n.message, n.level, n.duration_ms
        )

        # Tray callbacks
        self.tray.on_quit = self.stop
        self.tray.on_toggle = self._toggle_system
        self.tray.on_open_settings = lambda: self.visual_feedback.notify(
            f"Settings: http://127.0.0.1:{settings.get('settings_port', 7842)}",
            level="info", duration_ms=5000,
        )

        # --- Confidence settings ---
        self._conf_threshold = settings.get("confidence_threshold", 0.70)
        self._cmd_conf_threshold = settings.get("command_confidence_threshold", 0.85)

        self._running = False
        self._paused = False
        self._hand_visible = False

    def _load_config(self) -> dict:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                return json.load(f)
        return {"settings": {}, "gestures": {}}

    def _reload_config(self):
        """Hot-reload config.json."""
        try:
            self.config = self._load_config()
            settings = self.config.get("settings", {})
            self._conf_threshold = settings.get("confidence_threshold", 0.70)
            self._cmd_conf_threshold = settings.get("command_confidence_threshold", 0.85)
            self.visual_feedback.notify("Config reloaded", level="success", duration_ms=2000)
        except Exception as e:
            self.visual_feedback.notify(f"Config reload failed: {e}", level="error")

    # --- State machine callbacks ---

    def _on_state_change(self, old_state, new_state):
        """Propagate state changes to all visual components."""
        self.overlay.update_mode(new_state.name)
        self.tray.update_mode(new_state.name)
        self.visual_feedback.announce_mode_switch(new_state.name)

        if new_state == State.IDLE:
            self.overlay.hide()
        else:
            self.overlay.show()

        if old_state == State.TYPING and new_state != State.TYPING:
            self.text_injector.clear()
            self.overlay.update_buffer("")

    def _toggle_system(self):
        """Pause/resume from tray."""
        self._paused = not self._paused
        status = "paused" if self._paused else "resumed"
        self.visual_feedback.notify(f"System {status}", level="info", duration_ms=2000)

    # --- Main inference loop ---

    def _run_inference_loop(self):
        """Landmark extraction → classification → action execution."""
        while self._running:
            if self._paused:
                time.sleep(0.1)
                continue

            frame = self.camera.read(timeout=0.5)
            if frame is None:
                continue

            landmarks = self.extractor.extract_single(frame)
            current_state = self.state_machine.state

            if landmarks is not None:
                # Hand appeared
                if not self._hand_visible:
                    self._hand_visible = True

                # Feed frame to dynamic gesture classifier
                self.dynamic_clf.add_frame(landmarks)

                if current_state == State.TYPING:
                    self._handle_typing(landmarks)
                elif current_state == State.COMMAND:
                    self._handle_command(landmarks)

                # Check dynamic gestures in any state
                if self.dynamic_clf.is_loaded and self.dynamic_clf.buffer_length >= 30:
                    self._handle_dynamic_gesture()

                # Reset inactivity timer
                self.state_machine._last_activity = time.time()
            else:
                if self._hand_visible:
                    self._hand_visible = False
                    # Update overlay to show no hand
                    self.overlay.update_confidence(0.0, "")

            time.sleep(0.016)  # ~60 FPS target

    def _handle_typing(self, landmarks):
        """Classify landmarks and type confirmed letters."""
        letter, confidence = self.fingerspell.predict(landmarks)

        # Update overlay confidence bar and detected letter
        self.overlay.update_confidence(confidence, letter if confidence > 0.5 else "")

        # Process through hold-to-confirm gating
        self.text_injector.process_classification(letter, confidence, self._conf_threshold)
        self.overlay.update_buffer(self.text_injector.buffer_text)

    def _handle_command(self, landmarks):
        """Process a landmark frame in command mode."""
        gesture, confidence = self.gesture_clf.predict(landmarks)

        if confidence < self._cmd_conf_threshold:
            return

        binding = self.config.get("gestures", {}).get(gesture)
        if binding:
            fired = self.command_dispatcher.dispatch(
                binding["type"], binding["value"], gesture
            )
            if fired:
                self.visual_feedback.announce_command_fired(gesture, binding["value"])

    def _handle_dynamic_gesture(self):
        """Check the internal buffer for dynamic gestures like mode switch."""
        gesture, confidence = self.dynamic_clf.predict()

        if confidence > self._cmd_conf_threshold and gesture not in ("none", ""):
            if gesture == "mode_switch":
                self.event_queue.put(GestureEvent(
                    "dynamic", "mode_switch", confidence, time.time()
                ))
                self.dynamic_clf.clear_buffer()
            elif gesture == "enter_idle":
                self.event_queue.put(GestureEvent(
                    "dynamic", "enter_idle", confidence, time.time()
                ))
                self.dynamic_clf.clear_buffer()

    # --- Lifecycle ---

    def start(self):
        """Launch all subsystems and enter main loop."""
        self._running = True

        # Startup banner
        print()
        print("  🤟 SignType v1.0 — ASL-to-text for deaf/HoH users")
        print()

        # Check dependencies
        issues = _check_dependencies()
        if issues:
            for issue in issues:
                print(f"  ⚠ {issue}")
            print()

        # Check model
        if not self._check_models():
            print("  ⚠ Finger spelling model not found.")
            print("    Run: python training/preprocess_dataset.py")
            print("    Then: python training/train_fingerspell.py")
            print()

        # Start all subsystems
        self.visual_feedback.start()
        self.visual_feedback.announce_startup()

        self.camera.start()
        self.overlay.start()
        self.tray.start()
        self.state_machine.start()

        port = self.config.get("settings", {}).get("settings_port", 7842)
        run_settings_server(self.config_path, port, camera=self.camera)

        threading.Thread(
            target=self._run_inference_loop, daemon=True, name="inference"
        ).start()

        threading.Thread(
            target=self._watch_config, daemon=True, name="config-watcher"
        ).start()

        self.visual_feedback.announce_ready()

        # Start in TYPING mode directly
        self.state_machine.wake()

        print(f"  ✓ Ready — typing into focused app")
        print(f"  ✓ Settings: http://127.0.0.1:{port}")
        print(f"  ✓ Press Ctrl+C to stop")
        print()

        signal.signal(signal.SIGINT, lambda s, f: self.stop())
        signal.signal(signal.SIGTERM, lambda s, f: self.stop())

        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def _check_models(self) -> bool:
        return os.path.exists(os.path.join(self.data_dir, "model_fingerspell.pkl"))

    def _watch_config(self):
        """Poll config.json for changes every 5 seconds."""
        last_mtime = 0.0
        while self._running:
            try:
                mtime = os.path.getmtime(self.config_path)
                if mtime > last_mtime and last_mtime > 0:
                    self._reload_config()
                last_mtime = mtime
            except OSError:
                pass
            time.sleep(5)

    def stop(self):
        """Graceful shutdown."""
        if not self._running:
            return
        self._running = False

        print("\n  Shutting down...")
        self.camera.stop()
        self.state_machine.stop()
        self.overlay.quit()
        self.tray.stop()
        self.visual_feedback.stop()
        self.extractor.close()
        print("  ✓ Stopped.\n")


if __name__ == "__main__":
    app = SignTypeApp()
    app.start()

"""SignType — Main entry point.

Orchestrates 7 threads:
1. Camera capture (threaded frame grabber)
2. Inference loop (landmarks → classifiers → actions)
3. State machine (mode transitions, inactivity timeout)
4. Overlay (GTK4 + gtk4-layer-shell floating buffer)
5. Settings server (FastAPI on localhost:7842)
6. Visual notification processor (toast messages)
7. System tray (pystray with mode-colored icon)

All feedback is visual. No audio output — designed for deaf/HoH users.
"""

import os
import sys
import queue
import threading
import time
import json
import signal

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
        self.camera = Camera(camera_index=settings.get("camera_index", 0))
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
        self.text_injector = TextInjector()
        self.command_dispatcher = CommandDispatcher(self.config_path)

        # --- Wiring ---
        # State machine → overlay + tray + visual notification
        self.state_machine.on_state_change = self._on_state_change

        # Visual feedback → overlay toasts
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
        self._hold_ms = settings.get("fingerspell_hold_ms", 300)

        # --- Typing state (hold-to-confirm lives in TextInjector) ---

        self._running = False
        self._paused = False

    def _load_config(self) -> dict:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                return json.load(f)
        return {"settings": {}, "gestures": {}}

    def _reload_config(self):
        """Hot-reload config.json."""
        try:
            old_config = self.config
            self.config = self._load_config()
            settings = self.config.get("settings", {})
            self._conf_threshold = settings.get("confidence_threshold", 0.70)
            self._cmd_conf_threshold = settings.get("command_confidence_threshold", 0.85)
            self._hold_ms = settings.get("fingerspell_hold_ms", 300)
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

        # Clear typing buffer on mode switch
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
                # Feed frame to dynamic gesture classifier's internal buffer
                self.dynamic_clf.add_frame(landmarks)

                if current_state == State.TYPING:
                    self._handle_typing(landmarks)

                elif current_state == State.COMMAND:
                    self._handle_command(landmarks)

                # Check for dynamic gestures (mode switch, etc.) in any state
                if self.dynamic_clf.is_loaded and self.dynamic_clf.buffer_length >= 30:
                    self._handle_dynamic_gesture()

            # Reset inactivity timer on hand detection
            if landmarks is not None:
                self.state_machine._last_activity = time.time()

            # ~30 FPS target
            time.sleep(0.016)

    def _handle_typing(self, landmarks):
        """Process a landmark frame in typing mode with hold-to-confirm.
        
        Uses TextInjector.process_classification which already implements
        the 300ms consistent classification gating.
        """
        letter, confidence = self.fingerspell.predict(landmarks)
        if confidence > 0.5:
            print(f"  [Detected] {letter} ({confidence:.0%})", end="\r")
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
        """Launch all 7 threads and enter main loop."""
        self._running = True

        # First run check
        if not self._check_models():
            self._first_run_experience()

        # Start all subsystems
        self.visual_feedback.start()
        self.visual_feedback.announce_startup()

        self.camera.start()
        self.overlay.start()
        self.tray.start()
        self.state_machine.start()

        # Settings server
        port = self.config.get("settings", {}).get("settings_port", 7842)
        run_settings_server(self.config_path, port)

        # Inference thread
        inference_thread = threading.Thread(
            target=self._run_inference_loop, daemon=True, name="inference"
        )
        inference_thread.start()

        # Config file watcher (poll every 5 seconds)
        config_thread = threading.Thread(
            target=self._watch_config, daemon=True, name="config-watcher"
        )
        config_thread.start()

        self.visual_feedback.announce_ready()

        # Start in TYPING mode directly (dynamic gesture model for mode
        # switching may not be trained yet)
        self.state_machine.wake()

        print("[SignType] All systems running.")
        print(f"[SignType] Settings: http://127.0.0.1:{port}")
        print("[SignType] Press Ctrl+C to stop.")

        # Graceful shutdown on SIGINT/SIGTERM
        signal.signal(signal.SIGINT, lambda s, f: self.stop())
        signal.signal(signal.SIGTERM, lambda s, f: self.stop())

        # Keep main thread alive
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def _check_models(self) -> bool:
        """Check if required models exist."""
        main_model = os.path.join(self.data_dir, "model_fingerspell.pkl")
        return os.path.exists(main_model)

    def _first_run_experience(self):
        """Visual-only first run guidance."""
        print()
        print("╔══════════════════════════════════════════════╗")
        print("║     SignType — First Run Setup Needed        ║")
        print("╠══════════════════════════════════════════════╣")
        print("║                                              ║")
        print("║  The finger spelling model is missing.       ║")
        print("║  To set up:                                  ║")
        print("║                                              ║")
        print("║  1. python training/preprocess_dataset.py    ║")
        print("║  2. python training/train_fingerspell.py     ║")
        print("║                                              ║")
        print("║  Or use: make train                          ║")
        print("║                                              ║")
        print("║  The system will start in limited mode.      ║")
        print("║  Settings server will be available at:       ║")
        print("║  http://127.0.0.1:7842                       ║")
        print("║                                              ║")
        print("╚══════════════════════════════════════════════╝")
        print()

        self.visual_feedback.announce_first_run()

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
        """Graceful shutdown of all subsystems."""
        if not self._running:
            return
        self._running = False

        print("[SignType] Shutting down...")
        self.camera.stop()
        self.state_machine.stop()
        self.overlay.quit()
        self.tray.stop()
        self.visual_feedback.stop()
        self.extractor.close()
        print("[SignType] Shutdown complete.")


if __name__ == "__main__":
    app = SignTypeApp()
    app.start()

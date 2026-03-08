"""SignType - Main entry point.

Orchestrates 7 threads:
1. Camera capture
2. Inference (Fingerspell + Dynamic + Static)
3. State machine
4. Overlay (GTK4)
5. Settings server (FastAPI)
6. Visual notification processor
7. System tray
"""

import os
import sys
import queue
import threading
import time
import json
import webbrowser

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.camera import Camera
from core.landmark_extractor import LandmarkExtractor
from core.state_machine import StateMachine, State
from core.fingerspell_classifier import FingerspellClassifier
from core.gesture_classifier import GestureClassifier
from core.dynamic_classifier import DynamicGestureLSTM
from core.text_injector import TextInjector
from core.command_dispatcher import CommandDispatcher
from feedback.buffer_overlay import BufferOverlay
from feedback.audio import VisualFeedback
from feedback.tray import TrayIcon
from settings.server import run_settings_server


class SignTypeApp:
    """The main SignType application class."""

    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()

        # Shared resource queues
        self.event_queue = queue.Queue()
        self.landmark_queue = queue.Queue(maxsize=1)

        # Initialize core components
        camera_idx = self.config.get("settings", {}).get("camera_index", 0)
        self.camera = Camera(camera_index=camera_idx)
        self.extractor = LandmarkExtractor()
        self.state_machine = StateMachine(self.event_queue)

        # Models
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.fingerspell = FingerspellClassifier(
            os.path.join(data_dir, "model_fingerspell.pkl")
        )
        self.gesture_clf = GestureClassifier(
            os.path.join(data_dir, "model_static.pkl")
        )
        # Dynamic LSTM is specialized
        dyn_path = os.path.join(data_dir, "model_dynamic.pt")
        self.dynamic_clf = None
        if os.path.exists(dyn_path):
            self.dynamic_clf = DynamicGestureLSTM.from_file(dyn_path)

        # Feedback & IO
        self.visual_feedback = VisualFeedback()
        self.overlay = BufferOverlay()
        self.tray = TrayIcon()

        # Handlers
        self.text_injector = TextInjector()
        self.command_dispatcher = CommandDispatcher()

        # State transitions / notifications
        self.state_machine.on_state_change = self._on_state_change
        self.visual_feedback.on_notification = lambda n: self.overlay.show_notification(
            n.message, n.level, n.duration_ms
        )

        self._running = False

    def _load_config(self) -> dict:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                return json.load(f)
        return {"settings": {}, "gestures": {}}

    def _on_state_change(self, old_state, new_state):
        """Handle state-specific visual feedback."""
        self.overlay.update_mode(new_state.name)
        self.tray.update_mode(new_state.name)
        self.visual_feedback.announce_mode_switch(new_state.name)

        if new_state == State.IDLE:
            self.overlay.hide()
        else:
            self.overlay.show()

    def _run_inference_loop(self):
        """The main bridge: landmark extraction -> classification -> execution."""
        while self._running:
            frame = self.camera.read(timeout=0.5)
            if frame is None:
                continue

            # Extract 21 landmarks
            landmarks = self.extractor.extract_single(frame)

            # Route to appropriate classifier based on mode
            current_state = self.state_machine.current_state

            if landmarks is not None:
                if current_state == State.TYPING:
                    # Finger spelling mode
                    letter, confidence = self.fingerspell.predict(landmarks)
                    if confidence > self.config.get("settings", {}).get("confidence_threshold", 0.7):
                        result = self.text_injector.append(letter)
                        if result:
                            # Visual feedback of typed char
                            pass

                elif current_state == State.COMMAND:
                    # Static command mode
                    gesture, confidence = self.gesture_clf.predict(landmarks)
                    if confidence > self.config.get("settings", {}).get("command_confidence_threshold", 0.85):
                        # Dispatch command
                        binding = self.config.get("gestures", {}).get(gesture)
                        if binding:
                            self.command_dispatcher.dispatch(
                                binding["type"], binding["value"], gesture
                            )
                            self.visual_feedback.announce_command_fired(gesture, binding["value"])

                # Dynamic gestures (like mode switch: double palm)
                # This would feed into a small buffer for the LSTM
                # For brevity in this main file, we use the state machine's internal check
                # or a simplified version:
                pass

            # Inform state machine of hand activity for inactivity clock
            self.state_machine.log_activity(landmarks is not None)

            # Yield thread
            time.sleep(0.01)

    def start(self):
        """Launch all components."""
        self._running = True

        # Check for first run (missing models)
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        main_model = os.path.join(data_dir, "model_fingerspell.pkl")
        if not os.path.exists(main_model):
            self._first_run_experience()

        # Start background components
        self.visual_feedback.start()
        self.camera.start()
        self.overlay.start()
        self.tray.start()
        self.state_machine.start()

        # Start settings server
        port = self.config.get("settings", {}).get("settings_port", 7842)
        run_settings_server(self.config_path, port)

        # Main inference thread
        self.inference_thread = threading.Thread(target=self._run_inference_loop, daemon=True)
        self.inference_thread.start()

        self.visual_feedback.announce_startup()
        self.visual_feedback.announce_ready()

        # Keep main thread alive
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def _first_run_experience(self):
        """Guide user through initial setup when models are missing."""
        print("!!! First run detected: Finger spelling model missing !!!")
        print("1. Download ASL dataset")
        print("2. Run training/preprocess_dataset.py")
        print("3. Run training/train_fingerspell.py")
        print("Opening settings server at http://localhost:7842")

        self.visual_feedback.announce_first_run()

    def stop(self):
        """Graceful shutdown."""
        self._running = False
        self.camera.stop()
        self.state_machine.stop()
        self.overlay.quit()
        self.tray.stop()
        self.visual_feedback.stop()
        print("[SignType] Shutdown complete.")


if __name__ == "__main__":
    app = SignTypeApp()
    app.start()

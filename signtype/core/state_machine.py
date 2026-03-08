"""State machine — manages IDLE / TYPING / COMMAND / RECORDING states."""

import enum
import threading
import queue
import time
from dataclasses import dataclass


class State(enum.Enum):
    IDLE = "idle"
    TYPING = "typing"
    COMMAND = "command"
    RECORDING = "recording"


@dataclass
class GestureEvent:
    """An event from the inference pipeline."""
    gesture_type: str  # "fingerspell", "static_command", "dynamic", "mode_switch"
    label: str
    confidence: float
    timestamp: float
    hand_count: int = 1


class StateMachine:
    """Thread-safe state machine for SignType modes.

    Transitions:
        IDLE → TYPING (via mode-switch gesture or startup)
        IDLE → RECORDING (via settings trigger)
        TYPING ↔ COMMAND (via mode-switch gesture: both open palms)
        TYPING → RECORDING (via settings trigger)
        COMMAND → RECORDING (via settings trigger)
        RECORDING → (previous state) on completion
        ANY → IDLE (via double open palm / inactivity timeout)
    """

    def __init__(
        self,
        event_queue: queue.Queue,
        inactivity_timeout: float = 60.0,
    ):
        self._state = State.IDLE
        self._previous_state = State.IDLE
        self._lock = threading.Lock()
        self._event_queue = event_queue
        self._inactivity_timeout = inactivity_timeout
        self._last_activity = time.time()
        self._running = False
        self._thread = None

        # Callbacks — set by the application
        self.on_state_change = None  # (old_state, new_state) -> None
        self.on_fingerspell = None   # (label, confidence) -> None
        self.on_command = None       # (label, confidence) -> None
        self.on_dynamic = None       # (label, confidence) -> None

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    def _set_state(self, new_state: State):
        with self._lock:
            old_state = self._state
            if old_state == new_state:
                return
            self._previous_state = old_state
            self._state = new_state

        if self.on_state_change:
            self.on_state_change(old_state, new_state)

    def start(self):
        """Start the state machine event loop."""
        self._running = True
        self._thread = threading.Thread(target=self._event_loop, daemon=True)
        self._thread.start()

    def _event_loop(self):
        """Main loop: consume gesture events and route them."""
        while self._running:
            try:
                event = self._event_queue.get(timeout=1.0)
            except queue.Empty:
                self._check_inactivity()
                continue

            self._last_activity = time.time()
            self._process_event(event)

    def _process_event(self, event: GestureEvent):
        """Route gesture event based on current state."""
        current = self.state

        # IDLE: no gesture processing
        if current == State.IDLE:
            # Only allow mode-switch to wake up
            if event.gesture_type == "dynamic" and event.label == "mode_switch":
                self._set_state(State.TYPING)
            return

        # RECORDING: ignore all gestures
        if current == State.RECORDING:
            return

        # Mode switch: both open palms (dynamic gesture)
        if event.gesture_type == "dynamic" and event.label == "mode_switch":
            if current == State.TYPING:
                self._set_state(State.COMMAND)
            elif current == State.COMMAND:
                self._set_state(State.TYPING)
            return

        # Enter IDLE: double open palm
        if event.gesture_type == "dynamic" and event.label == "enter_idle":
            self._set_state(State.IDLE)
            return

        # TYPING mode events
        if current == State.TYPING:
            if event.gesture_type == "fingerspell":
                if self.on_fingerspell:
                    self.on_fingerspell(event.label, event.confidence)
            elif event.gesture_type == "dynamic" and event.label == "delete":
                if self.on_fingerspell:
                    self.on_fingerspell("__DELETE__", event.confidence)
            elif event.gesture_type == "dynamic" and event.label == "confirm":
                if self.on_fingerspell:
                    self.on_fingerspell("__CONFIRM__", event.confidence)

        # COMMAND mode events
        elif current == State.COMMAND:
            if event.gesture_type in ("static_command", "dynamic"):
                if self.on_command:
                    self.on_command(event.label, event.confidence)

    def _check_inactivity(self):
        """Enter IDLE if no activity for timeout period."""
        if self.state not in (State.IDLE, State.RECORDING):
            elapsed = time.time() - self._last_activity
            if elapsed > self._inactivity_timeout:
                self._set_state(State.IDLE)

    def enter_recording(self):
        """Enter RECORDING state (called from settings/API)."""
        self._set_state(State.RECORDING)

    def exit_recording(self):
        """Exit RECORDING state, return to previous state."""
        with self._lock:
            prev = self._previous_state
        self._set_state(prev if prev != State.RECORDING else State.IDLE)

    def wake(self):
        """Wake from IDLE to TYPING mode."""
        if self.state == State.IDLE:
            self._set_state(State.TYPING)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)


# Quick self-test
if __name__ == "__main__":
    eq = queue.Queue()
    sm = StateMachine(eq)

    def on_change(old, new):
        print(f"  State: {old.value} → {new.value}")

    sm.on_state_change = on_change

    print("Testing state machine transitions:")
    sm.start()

    # IDLE → TYPING
    eq.put(GestureEvent("dynamic", "mode_switch", 0.95, time.time()))
    time.sleep(0.5)

    # TYPING → COMMAND
    eq.put(GestureEvent("dynamic", "mode_switch", 0.90, time.time()))
    time.sleep(0.5)

    # COMMAND → TYPING
    eq.put(GestureEvent("dynamic", "mode_switch", 0.90, time.time()))
    time.sleep(0.5)

    # TYPING → IDLE
    eq.put(GestureEvent("dynamic", "enter_idle", 0.90, time.time()))
    time.sleep(0.5)

    sm.stop()
    print(f"Final state: {sm.state.value}")
    print("✓ State machine test complete")

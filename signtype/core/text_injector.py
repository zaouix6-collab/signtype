"""Text injector — types directly into the focused application.

UX model: Sign → Hold → Confirm → Release → Next Sign

State machine per letter:
  IDLE        → no sign detected, waiting
  DETECTING   → consistent sign detected, accumulating hold time
  CONFIRMED   → letter typed! waiting for hand to CHANGE before accepting again
  
Key design decisions:
  - A letter types ONCE when held for hold_ms
  - After typing, the sign must CHANGE (different letter or rest) before
    the same letter can be typed again
  - This prevents repeats naturally without fragile cooldown timers
  - To type double letters (e.g. "ll"), sign L → release/rest → sign L again
  - "nothing" / low confidence resets state (hand went to rest)
"""

import os
import subprocess
import threading
import time
from typing import Callable


def _detect_session_type() -> str:
    return os.environ.get("XDG_SESSION_TYPE", "x11")


def _has_command(name: str) -> bool:
    try:
        subprocess.run(["which", name], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# States for the typing state machine
IDLE = "idle"
DETECTING = "detecting"
CONFIRMED = "confirmed"


class TextInjector:
    """Types ASL letters into the focused application.

    Uses a sign-release-sign model: each sign produces exactly ONE
    keystroke, and the hand must move away before the same letter
    can be typed again.
    """

    def __init__(self, fingerspell_hold_ms: int = 400, cooldown_ms: int = 300):
        self._lock = threading.Lock()
        self._hold_ms = fingerspell_hold_ms
        self._cooldown_ms = cooldown_ms  # minimum gap between ANY two keystrokes
        self._session_type = _detect_session_type()

        # Typing backend
        self._use_ydotool = _has_command("ydotool")
        self._use_wtype = (not self._use_ydotool and
                           self._session_type == "wayland" and
                           _has_command("wtype"))

        if self._use_ydotool:
            print("[TextInjector] Using ydotool")
        elif self._use_wtype:
            print("[TextInjector] Using wtype (Wayland)")
        else:
            print("[TextInjector] WARNING: No typing tool found!")
            print("  Install ydotool: sudo pacman -S ydotool")

        # --- State machine ---
        self._state: str = IDLE
        self._current_char: str | None = None      # char being tracked
        self._hold_start: float | None = None       # when tracking started
        self._confirmed_char: str | None = None     # last typed char (waiting for release)
        self._last_type_time: float = 0.0           # when last keystroke was sent
        self._hold_progress: float = 0.0            # 0.0 to 1.0 for overlay

        # Visual buffer for overlay display
        self._buffer: list[str] = []

        # Callbacks
        self.on_buffer_change: Callable[[str], None] | None = None
        self.on_inject: Callable[[str], None] | None = None
        self.on_hold_progress: Callable[[float, str], None] | None = None

    @property
    def buffer_text(self) -> str:
        with self._lock:
            return "".join(self._buffer[-20:])

    @property
    def hold_progress(self) -> float:
        return self._hold_progress

    def process_classification(self, letter: str, confidence: float,
                                min_confidence: float = 0.70):
        """Process one frame's classification result.

        This implements the Sign → Hold → Confirm → Release cycle.
        """
        now = time.time()

        # --- Special: delete gesture ---
        if letter == "del":
            if now - self._last_type_time > self._cooldown_ms / 1000:
                self._type_backspace()
                self._last_type_time = now
            self._go_idle()
            return

        # --- Rest position / low confidence → reset ---
        if letter in ("nothing", "NOTHING", "") or confidence < min_confidence:
            # Hand went to rest or model is uncertain.
            # This is the "release" that allows the same letter again.
            if self._state == CONFIRMED:
                # Great — user released after typing. Ready for next letter.
                pass
            self._go_idle()
            return

        # --- Convert label to char ---
        if letter in ("space", "SPACE"):
            char = " "
        elif len(letter) == 1 and letter.isalpha():
            char = letter.lower()
        else:
            self._go_idle()
            return

        # --- State machine ---

        if self._state == IDLE:
            # New sign detected — start tracking
            self._state = DETECTING
            self._current_char = char
            self._hold_start = now
            self._hold_progress = 0.0
            self._notify_progress(0.0, char)

        elif self._state == DETECTING:
            if char == self._current_char:
                # Same letter — accumulate hold time
                elapsed_ms = (now - self._hold_start) * 1000
                self._hold_progress = min(elapsed_ms / self._hold_ms, 1.0)
                self._notify_progress(self._hold_progress, char)

                if elapsed_ms >= self._hold_ms:
                    # CONFIRMED! Type the letter.
                    # But enforce a minimum gap between keystrokes.
                    if now - self._last_type_time > self._cooldown_ms / 1000:
                        self._type_char(char)
                        self._last_type_time = now

                        with self._lock:
                            self._buffer.append(char)
                        if self.on_buffer_change:
                            self.on_buffer_change(self.buffer_text)
                        if self.on_inject:
                            self.on_inject(char)

                    # Move to CONFIRMED — waiting for hand to change
                    self._state = CONFIRMED
                    self._confirmed_char = char
                    self._hold_progress = 1.0
                    self._notify_progress(1.0, char)
            else:
                # Different letter — restart tracking with new letter
                self._state = DETECTING
                self._current_char = char
                self._hold_start = now
                self._hold_progress = 0.0
                self._notify_progress(0.0, char)

        elif self._state == CONFIRMED:
            if char == self._confirmed_char:
                # Still showing the same sign — ignore.
                # User must release or change sign first.
                pass
            else:
                # Hand changed to a DIFFERENT letter — start tracking it
                self._state = DETECTING
                self._current_char = char
                self._confirmed_char = None
                self._hold_start = now
                self._hold_progress = 0.0
                self._notify_progress(0.0, char)

    def _go_idle(self):
        """Reset to idle state."""
        self._state = IDLE
        self._current_char = None
        self._hold_start = None
        self._confirmed_char = None
        self._hold_progress = 0.0

    def _notify_progress(self, progress: float, char: str):
        """Notify overlay of hold progress."""
        if self.on_hold_progress:
            self.on_hold_progress(progress, char)

    # --- Typing backends ---

    def _type_char(self, char: str):
        if self._use_ydotool:
            self._type_ydotool(char)
        elif self._use_wtype:
            self._type_wtype(char)
        else:
            print(f"  [WOULD TYPE] {char}")

    def _type_wtype(self, char: str):
        try:
            subprocess.run(["wtype", char], check=True, timeout=2.0,
                          capture_output=True)
        except FileNotFoundError:
            self._use_wtype = False
            self._use_ydotool = _has_command("ydotool")
            self._type_char(char)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass

    def _type_ydotool(self, char: str):
        try:
            subprocess.run(["ydotool", "type", "--", char], check=True,
                          timeout=2.0, capture_output=True)
        except FileNotFoundError:
            self._use_ydotool = False
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass

    def _type_backspace(self):
        with self._lock:
            if self._buffer:
                self._buffer.pop()
        if self.on_buffer_change:
            self.on_buffer_change(self.buffer_text)

        if self._use_ydotool:
            try:
                subprocess.run(["ydotool", "key", "14:1", "14:0"],
                              check=True, timeout=2.0, capture_output=True)
            except (FileNotFoundError, subprocess.CalledProcessError,
                    subprocess.TimeoutExpired):
                pass
        elif self._use_wtype:
            try:
                subprocess.run(["wtype", "-k", "BackSpace"],
                              check=True, timeout=2.0, capture_output=True)
            except (FileNotFoundError, subprocess.CalledProcessError,
                    subprocess.TimeoutExpired):
                pass

    def clear(self):
        with self._lock:
            self._buffer.clear()
        self._go_idle()
        if self.on_buffer_change:
            self.on_buffer_change("")

    def delete_last(self):
        self._type_backspace()

    def confirm_and_inject(self):
        """Legacy: inject entire buffer."""
        text = self.buffer_text
        if text:
            for char in text:
                self._type_char(char)
                time.sleep(0.02)
            self.clear()


if __name__ == "__main__":
    ti = TextInjector(fingerspell_hold_ms=400)
    print(f"Session: {ti._session_type}")
    print(f"ydotool: {ti._use_ydotool}")
    print()
    print("Test: typing 'hello' in 3 seconds...")
    print("Click a text field NOW!")
    time.sleep(3)
    for char in "hello":
        ti._type_char(char)
        time.sleep(0.1)
    print("✓ Did 'hello' appear?")

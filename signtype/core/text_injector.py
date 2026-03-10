"""Text injector — types directly into the focused application.

Uses wtype on Wayland, ydotool as fallback, pyautogui for X11.
Each confirmed letter is immediately typed into whatever app/textbox
the user has focused — works everywhere (browser, terminal, chat, etc).
"""

import os
import subprocess
import threading
import time


def _detect_session_type() -> str:
    """Detect whether we're on Wayland or X11."""
    return os.environ.get("XDG_SESSION_TYPE", "x11")


def _has_command(name: str) -> bool:
    """Check if a command exists on PATH."""
    try:
        subprocess.run(["which", name], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


class TextInjector:
    """Types letters directly into the focused application.

    Implements 300ms consistent classification gating — a character is
    only typed after it's consistently classified for the hold duration.
    After typing, there's a cooldown to prevent repeats.
    """

    def __init__(self, fingerspell_hold_ms: int = 400, cooldown_ms: int = 600):
        self._lock = threading.Lock()
        self._hold_ms = fingerspell_hold_ms
        self._cooldown_ms = cooldown_ms
        self._session_type = _detect_session_type()

        # Detect available typing tools (ydotool preferred on KDE Wayland)
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

        # Confidence gating state
        self._current_candidate: str | None = None
        self._candidate_start_time: float | None = None
        self._last_typed_time: float = 0.0
        self._last_typed_char: str | None = None

        # Visual buffer for overlay display
        self._buffer = []

        # Callbacks
        self.on_buffer_change: object | None = None
        self.on_inject: object | None = None

    @property
    def buffer_text(self) -> str:
        """Current visual buffer (last 20 chars for overlay display)."""
        with self._lock:
            return "".join(self._buffer[-20:])

    def process_classification(self, letter: str, confidence: float,
                                min_confidence: float = 0.85):
        """Process a classification result with confidence gating.

        A character is typed into the focused app only after being
        consistently classified for fingerspell_hold_ms milliseconds.
        """
        now = time.time()

        # Handle special gestures
        if letter == "del":
            if (now - self._last_typed_time) * 1000 >= self._cooldown_ms:
                self._type_backspace()
                self._last_typed_time = now
            self._reset_candidate()
            return

        # Skip low confidence or "nothing" class
        if letter in ("nothing", "NOTHING") or confidence < min_confidence:
            self._reset_candidate()
            return

        # Convert label to character
        if letter in ("space", "SPACE"):
            char = " "
        elif len(letter) == 1 and letter.isalpha():
            char = letter.lower()
        else:
            self._reset_candidate()
            return

        # Cooldown check — don't repeat same char too fast
        if (char == self._last_typed_char and
                (now - self._last_typed_time) * 1000 < self._cooldown_ms):
            return

        # Hold-to-confirm logic
        if self._current_candidate == char:
            elapsed_ms = (now - self._candidate_start_time) * 1000
            if elapsed_ms >= self._hold_ms:
                # Confirmed! Type it directly into focused app
                self._type_char(char)
                self._last_typed_char = char
                self._last_typed_time = now

                # Update visual buffer
                with self._lock:
                    self._buffer.append(char)
                if self.on_buffer_change:
                    self.on_buffer_change(self.buffer_text)
                if self.on_inject:
                    self.on_inject(char)

                self._reset_candidate()
        else:
            # New character — start tracking
            self._current_candidate = char
            self._candidate_start_time = now

    def _reset_candidate(self):
        self._current_candidate = None
        self._candidate_start_time = None

    def _type_char(self, char: str):
        """Type a single character into the focused application."""
        if self._use_ydotool:
            self._type_ydotool(char)
        elif self._use_wtype:
            self._type_wtype(char)
        else:
            print(f"  [WOULD TYPE] {char}")

    def _type_wtype(self, char: str):
        """Type using wtype (Wayland native)."""
        try:
            subprocess.run(
                ["wtype", char],
                check=True,
                timeout=2.0,
                capture_output=True,
            )
        except FileNotFoundError:
            print("[TextInjector] wtype not found, falling back to ydotool")
            self._use_wtype = False
            self._use_ydotool = _has_command("ydotool")
            self._type_char(char)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode() if e.stderr else ""
            if stderr:
                print(f"[TextInjector] wtype error: {stderr.strip()}")
        except subprocess.TimeoutExpired:
            pass

    def _type_ydotool(self, char: str):
        """Type using ydotool (works on both Wayland and X11)."""
        try:
            subprocess.run(
                ["ydotool", "type", "--", char],
                check=True,
                timeout=2.0,
                capture_output=True,
            )
        except FileNotFoundError:
            print("[TextInjector] ydotool not found")
            self._use_ydotool = False
        except subprocess.CalledProcessError:
            pass
        except subprocess.TimeoutExpired:
            pass

    def _type_backspace(self):
        """Send a backspace keystroke."""
        with self._lock:
            if self._buffer:
                self._buffer.pop()
        if self.on_buffer_change:
            self.on_buffer_change(self.buffer_text)

        if self._use_wtype:
            try:
                subprocess.run(
                    ["wtype", "-k", "BackSpace"],
                    check=True, timeout=2.0, capture_output=True,
                )
            except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass
        elif self._use_ydotool:
            try:
                subprocess.run(
                    ["ydotool", "key", "14:1", "14:0"],
                    check=True, timeout=2.0, capture_output=True,
                )
            except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass

    def clear(self):
        """Clear the visual buffer."""
        with self._lock:
            self._buffer.clear()
        self._reset_candidate()
        if self.on_buffer_change:
            self.on_buffer_change("")

    def delete_last(self):
        """Delete the last character (for backward compat)."""
        self._type_backspace()

    def confirm_and_inject(self):
        """Legacy: inject entire buffer (no longer primary method)."""
        text = self.buffer_text
        if text:
            for char in text:
                self._type_char(char)
                time.sleep(0.02)
            self.clear()


# Quick self-test
if __name__ == "__main__":
    ti = TextInjector(fingerspell_hold_ms=300)
    print(f"Session type: {ti._session_type}")
    print(f"Using wtype: {ti._use_wtype}")
    print(f"Using ydotool: {ti._use_ydotool}")
    print()
    print("Test: typing 'hello' into focused app in 3 seconds...")
    print("Click on a text field NOW!")
    time.sleep(3)

    for char in "hello":
        ti._type_char(char)
        time.sleep(0.1)

    print()
    print("✓ Did 'hello' appear in your text field?")

"""Text injector — manages text buffer and injects into active application.

Uses wtype on Wayland, pyautogui fallback on X11.
"""

import os
import subprocess
import threading
import time


def _detect_session_type() -> str:
    """Detect whether we're on Wayland or X11."""
    return os.environ.get("XDG_SESSION_TYPE", "x11")


class TextInjector:
    """Manages the typing buffer and injects confirmed text.

    Implements 300ms consistent classification gating — a character is
    only appended after it's consistently classified for the hold duration.
    """

    def __init__(self, fingerspell_hold_ms: int = 300):
        self._buffer = []
        self._lock = threading.Lock()
        self._hold_ms = fingerspell_hold_ms
        self._session_type = _detect_session_type()

        # Confidence gating state
        self._current_candidate = None
        self._candidate_start_time = None

        # Callbacks
        self.on_buffer_change = None  # (buffer_text) -> None
        self.on_inject = None         # (injected_text) -> None

    @property
    def buffer_text(self) -> str:
        with self._lock:
            return "".join(self._buffer)

    def process_classification(self, letter: str, confidence: float,
                                min_confidence: float = 0.85):
        """Process a classification result with confidence gating.

        A character is appended only after being consistently classified
        for fingerspell_hold_ms milliseconds.
        """
        if letter == "__DELETE__":
            self.delete_last()
            self._reset_candidate()
            return

        if letter == "__CONFIRM__":
            self.confirm_and_inject()
            self._reset_candidate()
            return

        if letter in ("NOTHING", "DELETE") or confidence < min_confidence:
            self._reset_candidate()
            return

        # Convert to lowercase for typing
        char = " " if letter == "SPACE" else letter.lower()

        now = time.time()
        if self._current_candidate == char:
            # Same character — check if hold duration met
            elapsed_ms = (now - self._candidate_start_time) * 1000
            if elapsed_ms >= self._hold_ms:
                self.append(char)
                self._reset_candidate()
        else:
            # New character — start tracking
            self._current_candidate = char
            self._candidate_start_time = now

    def _reset_candidate(self):
        self._current_candidate = None
        self._candidate_start_time = None

    def append(self, char: str):
        """Append a character to the buffer."""
        with self._lock:
            self._buffer.append(char)
        if self.on_buffer_change:
            self.on_buffer_change(self.buffer_text)

    def delete_last(self):
        """Remove the last character from the buffer."""
        with self._lock:
            if self._buffer:
                self._buffer.pop()
        if self.on_buffer_change:
            self.on_buffer_change(self.buffer_text)

    def clear(self):
        """Clear the entire buffer."""
        with self._lock:
            self._buffer.clear()
        if self.on_buffer_change:
            self.on_buffer_change(self.buffer_text)

    def confirm_and_inject(self):
        """Inject buffer text into the active application and clear buffer."""
        text = self.buffer_text
        if not text:
            return

        self._inject_text(text)

        if self.on_inject:
            self.on_inject(text)

        self.clear()

    def _inject_text(self, text: str):
        """Inject text using the appropriate method for the display server."""
        if self._session_type == "wayland":
            self._inject_wtype(text)
        else:
            self._inject_pyautogui(text)

    def _inject_wtype(self, text: str):
        """Inject text using wtype (Wayland)."""
        try:
            subprocess.run(
                ["wtype", text],
                check=True,
                timeout=5.0,
                capture_output=True,
            )
        except FileNotFoundError:
            print("ERROR: wtype not found. Install with: sudo pacman -S wtype")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: wtype failed: {e.stderr.decode()}")
        except subprocess.TimeoutExpired:
            print("ERROR: wtype timed out")

    def _inject_pyautogui(self, text: str):
        """Inject text using pyautogui (X11 fallback)."""
        try:
            import pyautogui
            pyautogui.typewrite(text, interval=0.02)
        except ImportError:
            print("ERROR: pyautogui not installed (needed for X11)")


# Quick self-test
if __name__ == "__main__":
    ti = TextInjector(fingerspell_hold_ms=300)
    print(f"Session type: {ti._session_type}")

    ti.append("h")
    ti.append("e")
    ti.append("l")
    ti.append("l")
    ti.append("o")
    print(f"Buffer: '{ti.buffer_text}'")

    ti.delete_last()
    print(f"After delete: '{ti.buffer_text}'")

    ti.clear()
    print(f"After clear: '{ti.buffer_text}'")
    print("✓ Text injector test complete")

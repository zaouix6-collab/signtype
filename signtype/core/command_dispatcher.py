"""Command dispatcher — executes system commands bound to gestures.

Uses ydotool for hotkeys on Wayland, xdg-open for apps/URLs/files,
subprocess for shell commands.
"""

import json
import os
import subprocess
import threading
import time


def _detect_session_type() -> str:
    return os.environ.get("XDG_SESSION_TYPE", "x11")


class CommandDispatcher:
    """Dispatches commands based on gesture recognition with confidence gating.

    Supports command types:
        - shell: run a shell command
        - hotkey: simulate a keyboard shortcut
        - launch: open an application
        - url: open a URL in default browser
        - file: open a file with default application
    """

    def __init__(
        self,
        config_path: str,
        confidence_threshold: float = 0.85,
        hold_duration_ms: int = 500,
        cooldown_ms: int = 1500,
    ):
        self._config_path = config_path
        self._confidence_threshold = confidence_threshold
        self._hold_duration_ms = hold_duration_ms
        self._cooldown_ms = cooldown_ms
        self._session_type = _detect_session_type()

        # Gating state
        self._current_gesture = None
        self._gesture_start_time = None
        self._last_dispatch_time = 0
        self._lock = threading.Lock()

        # Load gesture bindings
        self._bindings = {}
        self.reload_config()

        # Callbacks
        self.on_dispatch = None  # (gesture_name, command_type, payload) -> None

    def reload_config(self):
        """Reload gesture-to-command bindings from config file."""
        try:
            with open(self._config_path, "r") as f:
                config = json.load(f)
            self._bindings = config.get("custom_gestures", {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load config: {e}")
            self._bindings = {}

    def process_gesture(self, gesture_name: str, confidence: float):
        """Process a gesture classification result with confidence gating.

        A command fires only when:
        1. Confidence >= threshold (0.85)
        2. Same gesture held for hold_duration (500ms)
        3. Cooldown period (1500ms) has elapsed since last dispatch
        """
        with self._lock:
            now = time.time()

            # Check cooldown
            if (now - self._last_dispatch_time) * 1000 < self._cooldown_ms:
                return

            # Check confidence
            if confidence < self._confidence_threshold:
                self._reset_gating()
                return

            # Check if it's a bound gesture
            if gesture_name not in self._bindings:
                return

            # Same gesture? Check hold duration
            if self._current_gesture == gesture_name:
                elapsed_ms = (now - self._gesture_start_time) * 1000
                if elapsed_ms >= self._hold_duration_ms:
                    self._dispatch(gesture_name)
                    self._reset_gating()
                    self._last_dispatch_time = now
            else:
                # New gesture — start tracking
                self._current_gesture = gesture_name
                self._gesture_start_time = now

    def _reset_gating(self):
        self._current_gesture = None
        self._gesture_start_time = None

    def _dispatch(self, gesture_name: str):
        """Execute the command bound to a gesture."""
        binding = self._bindings.get(gesture_name)
        if not binding:
            return

        cmd_type = binding.get("type", "")
        payload = binding.get("payload", "")
        label = binding.get("label", gesture_name)

        try:
            if cmd_type == "shell":
                self._run_shell(payload)
            elif cmd_type == "hotkey":
                self._run_hotkey(payload)
            elif cmd_type == "launch":
                self._run_launch(payload)
            elif cmd_type == "url":
                self._run_url(payload)
            elif cmd_type == "file":
                self._run_file(payload)
            else:
                print(f"Warning: Unknown command type '{cmd_type}' for {gesture_name}")
                return

            if self.on_dispatch:
                self.on_dispatch(gesture_name, cmd_type, payload)

        except Exception as e:
            print(f"Error dispatching {gesture_name}: {e}")

    def _run_shell(self, command: str):
        """Execute a shell command."""
        subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _run_hotkey(self, hotkey: str):
        """Simulate a keyboard shortcut.

        Args:
            hotkey: Keys separated by '+', e.g. 'ctrl+shift+t'
        """
        if self._session_type == "wayland":
            self._hotkey_ydotool(hotkey)
        else:
            self._hotkey_pyautogui(hotkey)

    def _hotkey_ydotool(self, hotkey: str):
        """Simulate hotkey using ydotool (Wayland)."""
        # ydotool uses key names like 'KEY_LEFTCTRL', 'KEY_T'
        key_map = {
            "ctrl": "29",    # KEY_LEFTCTRL
            "shift": "42",   # KEY_LEFTSHIFT
            "alt": "56",     # KEY_LEFTALT
            "super": "125",  # KEY_LEFTMETA
            "enter": "28",
            "tab": "15",
            "escape": "1",
            "space": "57",
            "backspace": "14",
            "delete": "111",
            "up": "103",
            "down": "108",
            "left": "105",
            "right": "106",
        }

        parts = hotkey.lower().split("+")
        keycodes = []
        for part in parts:
            part = part.strip()
            if part in key_map:
                keycodes.append(key_map[part])
            elif len(part) == 1 and part.isalpha():
                # Single letter — lookup scancode (a=30, b=48, etc.)
                # Using evdev scancodes for QWERTY layout
                alpha_map = "qwertyuiopasdfghjklzxcvbnm"
                scancode_start = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                                  30, 31, 32, 33, 34, 35, 36, 37, 38,
                                  44, 45, 46, 47, 48, 49, 50]
                idx = alpha_map.index(part) if part in alpha_map else -1
                if idx >= 0:
                    keycodes.append(str(scancode_start[idx]))

        if keycodes:
            # Press all keys, then release all
            # ydotool key: keycode:state (1=press, 0=release)
            args = []
            for kc in keycodes:
                args.append(f"{kc}:1")  # press
            for kc in reversed(keycodes):
                args.append(f"{kc}:0")  # release

            try:
                subprocess.run(
                    ["ydotool", "key"] + args,
                    check=True,
                    timeout=5.0,
                    capture_output=True,
                )
            except FileNotFoundError:
                print("ERROR: ydotool not found. Install with: sudo pacman -S ydotool")
            except subprocess.CalledProcessError as e:
                print(f"ERROR: ydotool failed: {e.stderr.decode()}")

    def _hotkey_pyautogui(self, hotkey: str):
        """Simulate hotkey using pyautogui (X11 fallback)."""
        try:
            import pyautogui
            keys = [k.strip() for k in hotkey.split("+")]
            pyautogui.hotkey(*keys)
        except ImportError:
            print("ERROR: pyautogui not installed (needed for X11 hotkeys)")

    def _run_launch(self, app: str):
        """Launch an application using xdg-open."""
        subprocess.Popen(
            ["xdg-open", app],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _run_url(self, url: str):
        """Open a URL in the default browser."""
        subprocess.Popen(
            ["xdg-open", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _run_file(self, filepath: str):
        """Open a file with the default application."""
        subprocess.Popen(
            ["xdg-open", filepath],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def update_settings(self, confidence: float = None, hold_ms: int = None,
                        cooldown_ms: int = None):
        """Update gating parameters at runtime."""
        if confidence is not None:
            self._confidence_threshold = confidence
        if hold_ms is not None:
            self._hold_duration_ms = hold_ms
        if cooldown_ms is not None:
            self._cooldown_ms = cooldown_ms


# Quick self-test
if __name__ == "__main__":
    dispatcher = CommandDispatcher(
        config_path=os.path.join(os.path.dirname(__file__), "..", "config.json")
    )
    print(f"Session type: {dispatcher._session_type}")
    print(f"Bindings loaded: {len(dispatcher._bindings)}")
    print("✓ Command dispatcher test complete")

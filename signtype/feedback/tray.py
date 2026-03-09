"""System tray icon — runs in a separate subprocess to avoid GTK3/GTK4 conflict.

pystray requires GTK 3.0, but our overlay uses GTK 4.0. They cannot coexist
in the same process. So the tray icon runs as a lightweight subprocess that
communicates via a simple socket.

If pystray is not available, the tray is silently disabled — the app
still works fine without it (you just don't get a tray icon).
"""

import threading
import subprocess
import os
import signal


class TrayIcon:
    """System tray icon showing current SignType status.

    Since pystray needs GTK 3.0 and the overlay uses GTK 4.0,
    the tray runs in a separate subprocess. If it fails to start,
    the app continues without a tray icon.

    Colors:
        Green  = TYPING mode
        Orange = COMMAND mode
        Gray   = IDLE
        Red    = RECORDING
    """

    def __init__(self):
        self._process = None
        self._current_mode = "idle"

        # Callbacks — set by the application
        self.on_open_settings = None  # () -> None
        self.on_toggle = None         # () -> None
        self.on_quit = None           # () -> None

    def start(self):
        """Start the tray icon in a separate subprocess."""
        tray_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "_tray_subprocess.py"
        )

        # Create the subprocess script if it doesn't exist
        if not os.path.exists(tray_script):
            self._write_tray_script(tray_script)

        try:
            self._process = subprocess.Popen(
                ["python3.12", tray_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            # Read stdout in background for callbacks
            self._reader_thread = threading.Thread(
                target=self._read_commands, daemon=True
            )
            self._reader_thread.start()
            print("[Tray] Started in subprocess")
        except Exception as e:
            print(f"[Tray] Could not start tray icon: {e}")
            print("[Tray] App will run without system tray icon.")
            self._process = None

    def _read_commands(self):
        """Read commands from the tray subprocess."""
        if not self._process or not self._process.stdout:
            return
        try:
            for line in self._process.stdout:
                cmd = line.decode().strip()
                if cmd == "SETTINGS" and self.on_open_settings:
                    self.on_open_settings()
                elif cmd == "TOGGLE" and self.on_toggle:
                    self.on_toggle()
                elif cmd == "QUIT" and self.on_quit:
                    self.on_quit()
        except (OSError, ValueError):
            pass

    def update_mode(self, mode: str):
        """Send mode update to the tray subprocess."""
        self._current_mode = mode.lower()
        self._send(f"MODE:{mode}")

    def _send(self, message: str):
        """Send a message to the tray subprocess."""
        if self._process and self._process.stdin:
            try:
                self._process.stdin.write(f"{message}\n".encode())
                self._process.stdin.flush()
            except (OSError, BrokenPipeError):
                pass

    def stop(self):
        """Stop the tray subprocess."""
        if self._process:
            try:
                self._send("QUIT")
                self._process.wait(timeout=2)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    self._process.kill()
                except OSError:
                    pass
            self._process = None

    def _write_tray_script(self, path: str):
        """Write the standalone tray subprocess script."""
        script = '''#!/usr/bin/env python3
"""Standalone tray icon subprocess — uses GTK 3.0 via pystray."""
import sys
import threading

try:
    from pystray import Icon, Menu, MenuItem
    from PIL import Image, ImageDraw
except ImportError:
    print("QUIT", flush=True)
    sys.exit(1)

COLORS = {
    "idle": (149, 165, 166),
    "typing": (46, 204, 113),
    "command": (230, 126, 34),
    "recording": (231, 76, 60),
}

def create_icon_image(color):
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([4, 4, size - 4, size - 4], fill=color)
    inner = tuple(min(c + 60, 255) for c in color)
    draw.ellipse([18, 18, size - 18, size - 18], fill=inner)
    return img

icon = None

def on_settings(ic, item):
    print("SETTINGS", flush=True)

def on_toggle(ic, item):
    print("TOGGLE", flush=True)

def on_quit(ic, item):
    print("QUIT", flush=True)
    if icon:
        icon.stop()

menu = Menu(
    MenuItem("SignType", None, enabled=False),
    Menu.SEPARATOR,
    MenuItem("Open Settings", on_settings),
    MenuItem("Toggle System", on_toggle),
    Menu.SEPARATOR,
    MenuItem("Quit", on_quit),
)

icon = Icon("signtype", create_icon_image(COLORS["idle"]), "SignType — IDLE", menu)

def stdin_reader():
    """Read commands from the main process."""
    global icon
    try:
        for line in sys.stdin:
            cmd = line.strip()
            if cmd.startswith("MODE:"):
                mode = cmd.split(":", 1)[1].lower()
                color = COLORS.get(mode, COLORS["idle"])
                if icon:
                    icon.icon = create_icon_image(color)
                    icon.title = f"SignType — {mode.upper()}"
            elif cmd == "QUIT":
                if icon:
                    icon.stop()
                break
    except (EOFError, OSError):
        if icon:
            icon.stop()

reader = threading.Thread(target=stdin_reader, daemon=True)
reader.start()
icon.run()
'''
        with open(path, "w") as f:
            f.write(script)


# Quick self-test
if __name__ == "__main__":
    import time

    tray = TrayIcon()
    tray.on_quit = lambda: print("Quit clicked")
    tray.on_open_settings = lambda: print("Settings clicked")
    tray.on_toggle = lambda: print("Toggle clicked")

    tray.start()
    print("Tray icon started — check system tray")

    for mode in ["typing", "command", "idle", "recording"]:
        time.sleep(2)
        tray.update_mode(mode)
        print(f"  Mode: {mode}")

    time.sleep(3)
    tray.stop()
    print("✓ Tray icon test complete")

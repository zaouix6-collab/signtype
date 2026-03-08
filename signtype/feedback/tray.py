"""System tray icon — pystray with mode-colored status.

Uses StatusNotifierItem protocol on KDE Wayland.
"""

import threading

try:
    from pystray import Icon, Menu, MenuItem
    from PIL import Image, ImageDraw
    HAS_TRAY = True
except ImportError:
    HAS_TRAY = False


class TrayIcon:
    """System tray icon showing current SignType status.

    Colors:
        Green  = TYPING mode
        Orange = COMMAND mode
        Gray   = IDLE
        Red    = RECORDING
    """

    COLORS = {
        "idle": (149, 165, 166),      # Gray
        "typing": (46, 204, 113),     # Green
        "command": (230, 126, 34),    # Orange
        "recording": (231, 76, 60),   # Red
    }

    def __init__(self):
        if not HAS_TRAY:
            raise ImportError(
                "pystray or Pillow not installed. "
                "Install with: pip install pystray Pillow"
            )
        self._icon = None
        self._thread = None
        self._current_mode = "idle"

        # Callbacks — set by the application
        self.on_open_settings = None  # () -> None
        self.on_toggle = None         # () -> None
        self.on_quit = None           # () -> None

    def _create_icon_image(self, color: tuple) -> Image.Image:
        """Create a colored circle icon."""
        size = 64
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Outer circle
        draw.ellipse([4, 4, size - 4, size - 4], fill=color)

        # Inner "S" shape approximated with a smaller circle
        inner_color = tuple(min(c + 60, 255) for c in color)
        draw.ellipse([18, 18, size - 18, size - 18], fill=inner_color)

        return img

    def start(self):
        """Start the tray icon."""
        menu = Menu(
            MenuItem("SignType", None, enabled=False),
            Menu.SEPARATOR,
            MenuItem("Open Settings", self._on_settings),
            MenuItem("Toggle System", self._on_toggle),
            Menu.SEPARATOR,
            MenuItem("Quit", self._on_quit),
        )

        color = self.COLORS.get(self._current_mode, self.COLORS["idle"])
        icon_image = self._create_icon_image(color)

        self._icon = Icon(
            "signtype",
            icon_image,
            "SignType — IDLE",
            menu,
        )

        self._thread = threading.Thread(target=self._icon.run, daemon=True)
        self._thread.start()

    def update_mode(self, mode: str):
        """Update the tray icon color and tooltip for the current mode."""
        self._current_mode = mode.lower()
        color = self.COLORS.get(self._current_mode, self.COLORS["idle"])

        if self._icon:
            self._icon.icon = self._create_icon_image(color)
            self._icon.title = f"SignType — {mode.upper()}"

    def _on_settings(self, icon, item):
        if self.on_open_settings:
            self.on_open_settings()

    def _on_toggle(self, icon, item):
        if self.on_toggle:
            self.on_toggle()

    def _on_quit(self, icon, item):
        if self.on_quit:
            self.on_quit()
        if self._icon:
            self._icon.stop()

    def stop(self):
        if self._icon:
            self._icon.stop()


# Quick self-test
if __name__ == "__main__":
    import time

    print(f"Tray available: {HAS_TRAY}")
    tray = TrayIcon()
    tray.on_quit = lambda: print("Quit clicked")
    tray.on_open_settings = lambda: print("Settings clicked")
    tray.on_toggle = lambda: print("Toggle clicked")

    tray.start()
    print("Tray icon started — check system tray")

    # Cycle through modes
    for mode in ["typing", "command", "idle", "recording"]:
        time.sleep(2)
        tray.update_mode(mode)
        print(f"  Mode: {mode}")

    time.sleep(3)
    tray.stop()
    print("✓ Tray icon test complete")

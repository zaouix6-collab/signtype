#!/usr/bin/env python3
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

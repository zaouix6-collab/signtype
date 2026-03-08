"""Floating text buffer overlay — GTK4 + gtk4-layer-shell for Wayland.

Displays the current typing buffer above other windows using the
Wayland layer-shell protocol. Falls back to a basic GTK4 window
if layer-shell is not available.
"""

import os
import threading
import sys

# Must be set before importing Gtk
os.environ.setdefault("GDK_BACKEND", "wayland")

try:
    import gi
    gi.require_version("Gtk", "4.0")
    from gi.repository import Gtk, Gdk, GLib, Pango

    # Try to load gtk4-layer-shell
    try:
        gi.require_version("Gtk4LayerShell", "1.0")
        from gi.repository import Gtk4LayerShell
        HAS_LAYER_SHELL = True
    except (ValueError, ImportError):
        Gtk4LayerShell = None
        HAS_LAYER_SHELL = False

    HAS_GTK = True
except ImportError:
    HAS_GTK = False
    HAS_LAYER_SHELL = False


class BufferOverlay:
    """Frameless floating buffer window using GTK4.

    Shows buffer text, mode indicator (TYPING=green border, COMMAND=orange).
    Uses gtk4-layer-shell to sit in the overlay layer on Wayland.
    """

    # Mode colors
    TYPING_COLOR = "#2ecc71"   # Green
    COMMAND_COLOR = "#e67e22"  # Orange
    IDLE_COLOR = "#95a5a6"     # Gray
    RECORDING_COLOR = "#e74c3c"  # Red

    def __init__(self):
        if not HAS_GTK:
            raise ImportError(
                "PyGObject with GTK4 is not available. "
                "Install with: sudo pacman -S python-gobject gtk4"
            )

        self._app = None
        self._window = None
        self._label = None
        self._mode_label = None
        self._notification_label = None
        self._thread = None
        self._buffer_text = ""
        self._mode = "idle"

    def start(self):
        """Start the overlay in a separate thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        """Run the GTK4 application."""
        self._app = Gtk.Application(application_id="com.signtype.overlay")
        self._app.connect("activate", self._on_activate)
        self._app.run(None)

    def _on_activate(self, app):
        """Build the overlay window."""
        self._window = Gtk.ApplicationWindow(application=app)
        self._window.set_title("SignType Buffer")
        self._window.set_default_size(400, 60)

        # Layer shell setup (Wayland overlay)
        if HAS_LAYER_SHELL:
            Gtk4LayerShell.init_for_window(self._window)
            Gtk4LayerShell.set_layer(self._window, Gtk4LayerShell.Layer.OVERLAY)
            Gtk4LayerShell.set_anchor(self._window, Gtk4LayerShell.Edge.BOTTOM, True)
            Gtk4LayerShell.set_margin(self._window, Gtk4LayerShell.Edge.BOTTOM, 80)
            Gtk4LayerShell.set_keyboard_mode(
                self._window, Gtk4LayerShell.KeyboardMode.NONE
            )
        else:
            # Fallback: basic window hints
            self._window.set_decorated(False)

        # Make the window transparent
        self._apply_css()

        # Layout
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        box.set_halign(Gtk.Align.CENTER)
        box.set_valign(Gtk.Align.CENTER)
        box.set_margin_start(16)
        box.set_margin_end(16)
        box.set_margin_top(8)
        box.set_margin_bottom(8)

        # Mode indicator
        self._mode_label = Gtk.Label()
        self._mode_label.set_markup(self._mode_markup("IDLE"))
        self._mode_label.add_css_class("mode-label")
        box.append(self._mode_label)

        # Buffer text
        self._label = Gtk.Label()
        self._label.set_markup('<span font="24" foreground="white">_</span>')
        self._label.set_wrap(True)
        self._label.set_max_width_chars(40)
        self._label.add_css_class("buffer-label")
        box.append(self._label)

        # Notification label (toast messages)
        self._notification_label = Gtk.Label()
        self._notification_label.set_markup("")
        self._notification_label.add_css_class("notification-label")
        self._notification_label.set_visible(False)
        self._notification_label.set_wrap(True)
        self._notification_label.set_max_width_chars(50)
        box.append(self._notification_label)

        # Container with border
        frame = Gtk.Frame()
        frame.set_child(box)
        frame.add_css_class("overlay-frame")

        self._window.set_child(frame)
        self._window.present()

    def _apply_css(self):
        """Apply custom CSS for the overlay appearance."""
        css = """
        window {
            background-color: rgba(30, 30, 30, 0.85);
            border-radius: 12px;
        }
        .overlay-frame {
            border: 3px solid #95a5a6;
            border-radius: 12px;
            background-color: rgba(30, 30, 30, 0.85);
            padding: 8px;
        }
        .overlay-frame.typing {
            border-color: #2ecc71;
        }
        .overlay-frame.command {
            border-color: #e67e22;
        }
        .overlay-frame.recording {
            border-color: #e74c3c;
        }
        .mode-label {
            font-size: 10px;
        }
        .buffer-label {
            font-size: 24px;
            color: white;
        }
        .notification-label {
            font-size: 12px;
            color: #ecf0f1;
            padding: 4px 8px;
            margin-top: 4px;
        }
        .notification-label.error {
            color: #e74c3c;
        }
        .notification-label.success {
            color: #2ecc71;
        }
        .notification-label.warning {
            color: #f39c12;
        }
        """
        provider = Gtk.CssProvider()
        provider.load_from_string(css)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

    def _mode_markup(self, mode_text: str) -> str:
        colors = {
            "IDLE": self.IDLE_COLOR,
            "TYPING": self.TYPING_COLOR,
            "COMMAND": self.COMMAND_COLOR,
            "RECORDING": self.RECORDING_COLOR,
        }
        color = colors.get(mode_text.upper(), self.IDLE_COLOR)
        return f'<span font="10" foreground="{color}" weight="bold">{mode_text.upper()}</span>'

    def update_buffer(self, text: str):
        """Thread-safe buffer text update."""
        self._buffer_text = text
        if self._label is not None:
            GLib.idle_add(self._do_update_buffer, text)

    def _do_update_buffer(self, text: str):
        display_text = text if text else "_"
        escaped = GLib.markup_escape_text(display_text)
        self._label.set_markup(f'<span font="24" foreground="white">{escaped}</span>')
        return False

    def update_mode(self, mode: str):
        """Thread-safe mode update. Changes border color and mode label."""
        self._mode = mode
        if self._mode_label is not None:
            GLib.idle_add(self._do_update_mode, mode)

    def _do_update_mode(self, mode: str):
        self._mode_label.set_markup(self._mode_markup(mode))

        # Update frame CSS class
        if self._window:
            frame = self._window.get_child()
            if frame:
                for cls in ("typing", "command", "recording"):
                    frame.remove_css_class(cls)
                if mode.lower() in ("typing", "command", "recording"):
                    frame.add_css_class(mode.lower())
        return False

    def show_notification(self, message: str, level: str = "info", duration_ms: int = 3000):
        """Show a visual toast notification on the overlay."""
        if self._notification_label is not None:
            GLib.idle_add(self._do_show_notification, message, level, duration_ms)

    def _do_show_notification(self, message: str, level: str, duration_ms: int):
        escaped = GLib.markup_escape_text(message)
        color = {"error": "#e74c3c", "success": "#2ecc71", "warning": "#f39c12"}.get(
            level, "#ecf0f1"
        )
        self._notification_label.set_markup(
            f'<span font="12" foreground="{color}">{escaped}</span>'
        )
        self._notification_label.set_visible(True)
        # Auto-hide after duration
        GLib.timeout_add(duration_ms, self._do_hide_notification)
        return False

    def _do_hide_notification(self):
        if self._notification_label:
            self._notification_label.set_visible(False)
        return False  # Don't repeat

    def hide(self):
        """Hide the overlay."""
        if self._window:
            GLib.idle_add(self._window.hide)

    def show(self):
        """Show the overlay."""
        if self._window:
            GLib.idle_add(self._window.present)

    def quit(self):
        """Quit the GTK application."""
        if self._app:
            GLib.idle_add(self._app.quit)


# Quick self-test
if __name__ == "__main__":
    if not HAS_GTK:
        print("✗ GTK4 not available")
        sys.exit(1)

    print(f"GTK4 available: {HAS_GTK}")
    print(f"Layer shell available: {HAS_LAYER_SHELL}")

    overlay = BufferOverlay()

    def test_updates():
        import time
        time.sleep(2)
        overlay.update_mode("TYPING")
        overlay.update_buffer("hello")
        time.sleep(2)
        overlay.update_mode("COMMAND")
        overlay.update_buffer("cmd mode")
        time.sleep(2)
        overlay.quit()

    t = threading.Thread(target=test_updates, daemon=True)
    t.start()

    # Must run GTK on main thread for self-test
    overlay._app = Gtk.Application(application_id="com.signtype.overlay.test")
    overlay._app.connect("activate", overlay._on_activate)
    overlay._app.run(None)
    print("✓ Buffer overlay test complete")

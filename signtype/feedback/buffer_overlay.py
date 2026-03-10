"""Floating text buffer overlay — GTK4 + gtk4-layer-shell for Wayland.

Displays:
- Current mode (TYPING / COMMAND / IDLE) — always visible
- Live confidence bar showing detection strength
- Detected letter in large font, green flash on confirm
- Typing buffer (last 20 characters)
- Toast notifications

Dark glassmorphism theme with semi-transparent background.
Falls back to a basic GTK4 window if layer-shell is not available.
"""

import os
import threading
import sys
import time

# Must be set before importing Gtk
os.environ.setdefault("GDK_BACKEND", "wayland")

try:
    import gi
    gi.require_version("Gtk", "4.0")
    from gi.repository import Gtk, Gdk, GLib, Pango

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


# ── Mode colors ────────────────────────────────────────────────────

MODE_CONFIG = {
    "TYPING":    {"color": "#2ecc71", "label": "✋ TYPING",    "border": "#2ecc71"},
    "COMMAND":   {"color": "#e67e22", "label": "👆 COMMAND",   "border": "#e67e22"},
    "IDLE":      {"color": "#95a5a6", "label": "⏸ IDLE",      "border": "#555"},
    "RECORDING": {"color": "#e74c3c", "label": "⏺ RECORDING", "border": "#e74c3c"},
}


class BufferOverlay:
    """Frameless floating overlay with live confidence feedback."""

    def __init__(self):
        if not HAS_GTK:
            raise ImportError(
                "PyGObject with GTK4 is not available. "
                "Install with: sudo pacman -S python-gobject gtk4"
            )

        self._app = None
        self._window = None
        self._buffer_label = None
        self._mode_label = None
        self._detected_label = None
        self._confidence_bar = None
        self._notification_label = None
        self._thread = None

        self._buffer_text = ""
        self._mode = "TYPING"
        self._confidence = 0.0
        self._detected_letter = ""
        self._notification_timeout_id = None

    def start(self):
        """Start the overlay in a separate thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        time.sleep(0.3)  # Let GTK initialize

    def _run(self):
        self._app = Gtk.Application(application_id="com.signtype.overlay")
        self._app.connect("activate", self._on_activate)
        self._app.run(None)

    def _on_activate(self, app):
        """Build the overlay window."""
        self._window = Gtk.ApplicationWindow(application=app)
        self._window.set_title("SignType")
        self._window.set_default_size(380, 140)

        # Layer shell setup
        if HAS_LAYER_SHELL:
            Gtk4LayerShell.init_for_window(self._window)
            Gtk4LayerShell.set_layer(self._window, Gtk4LayerShell.Layer.OVERLAY)
            Gtk4LayerShell.set_anchor(self._window, Gtk4LayerShell.Edge.BOTTOM, True)
            Gtk4LayerShell.set_anchor(self._window, Gtk4LayerShell.Edge.RIGHT, True)
            Gtk4LayerShell.set_margin(self._window, Gtk4LayerShell.Edge.BOTTOM, 40)
            Gtk4LayerShell.set_margin(self._window, Gtk4LayerShell.Edge.RIGHT, 40)
            Gtk4LayerShell.set_keyboard_mode(
                self._window, Gtk4LayerShell.KeyboardMode.NONE
            )
        else:
            self._window.set_decorated(False)

        self._apply_css()

        # ── Main layout ──
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        main_box.add_css_class("main-container")

        # Row 1: Mode indicator (always visible)
        self._mode_label = Gtk.Label()
        self._mode_label.add_css_class("mode-label")
        self._mode_label.set_halign(Gtk.Align.START)
        self._update_mode_label("TYPING")
        main_box.append(self._mode_label)

        # Row 2: Detected letter + confidence bar
        detect_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        detect_box.set_margin_top(6)
        detect_box.set_margin_bottom(6)

        self._detected_label = Gtk.Label(label="—")
        self._detected_label.add_css_class("detected-letter")
        detect_box.append(self._detected_label)

        # Confidence bar (vertical progress bar)
        conf_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        conf_box.set_valign(Gtk.Align.CENTER)
        conf_box.set_hexpand(True)

        self._conf_percent_label = Gtk.Label(label="0%")
        self._conf_percent_label.add_css_class("conf-percent")
        self._conf_percent_label.set_halign(Gtk.Align.END)
        conf_box.append(self._conf_percent_label)

        self._confidence_bar = Gtk.ProgressBar()
        self._confidence_bar.set_fraction(0.0)
        self._confidence_bar.add_css_class("confidence-bar")
        self._confidence_bar.set_hexpand(True)
        conf_box.append(self._confidence_bar)

        detect_box.append(conf_box)
        main_box.append(detect_box)

        # Row 3: Typing buffer
        self._buffer_label = Gtk.Label(label="_")
        self._buffer_label.add_css_class("buffer-label")
        self._buffer_label.set_wrap(True)
        self._buffer_label.set_max_width_chars(35)
        self._buffer_label.set_halign(Gtk.Align.START)
        main_box.append(self._buffer_label)

        # Row 4: Toast notification (hidden by default)
        self._notification_label = Gtk.Label()
        self._notification_label.add_css_class("notification-label")
        self._notification_label.set_visible(False)
        self._notification_label.set_wrap(True)
        self._notification_label.set_max_width_chars(40)
        main_box.append(self._notification_label)

        self._window.set_child(main_box)
        self._window.present()

    def _apply_css(self):
        """Apply glassmorphism dark theme."""
        css = Gtk.CssProvider()
        css.load_from_string("""
            .main-container {
                background: rgba(15, 15, 25, 0.88);
                border-radius: 16px;
                border: 1px solid rgba(255, 255, 255, 0.08);
                padding: 14px 18px;
            }

            .mode-label {
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 2px;
                padding: 3px 10px;
                border-radius: 4px;
            }

            .detected-letter {
                font-size: 42px;
                font-weight: 800;
                color: #fff;
                min-width: 56px;
                font-family: monospace;
            }

            .conf-percent {
                font-size: 10px;
                color: rgba(255, 255, 255, 0.5);
            }

            .confidence-bar {
                min-height: 6px;
                border-radius: 3px;
            }
            .confidence-bar trough {
                background: rgba(255, 255, 255, 0.08);
                border-radius: 3px;
                min-height: 6px;
            }
            .confidence-bar progress {
                background: #2ecc71;
                border-radius: 3px;
                min-height: 6px;
            }

            .confidence-bar.low progress { background: #e74c3c; }
            .confidence-bar.medium progress { background: #f39c12; }
            .confidence-bar.high progress { background: #2ecc71; }

            .buffer-label {
                font-size: 18px;
                font-family: monospace;
                color: rgba(255, 255, 255, 0.9);
                margin-top: 4px;
                padding: 4px 0;
                border-top: 1px solid rgba(255, 255, 255, 0.06);
            }

            .notification-label {
                font-size: 12px;
                color: #2ecc71;
                margin-top: 6px;
                padding: 4px 8px;
                border-radius: 6px;
                background: rgba(46, 204, 113, 0.1);
            }
            .notification-label.error {
                color: #e74c3c;
                background: rgba(231, 76, 60, 0.1);
            }
            .notification-label.warning {
                color: #f39c12;
                background: rgba(243, 156, 18, 0.1);
            }
        """)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

    def _update_mode_label(self, mode: str):
        """Update the always-visible mode indicator."""
        cfg = MODE_CONFIG.get(mode.upper(), MODE_CONFIG["IDLE"])
        if self._mode_label:
            self._mode_label.set_markup(
                f'<span foreground="{cfg["color"]}" font="11">'
                f'{cfg["label"]}</span>'
            )

    # ── Public update methods (thread-safe via GLib.idle_add) ──

    def update_buffer(self, text: str):
        """Update the typing buffer display."""
        self._buffer_text = text
        GLib.idle_add(self._do_update_buffer)

    def _do_update_buffer(self):
        if self._buffer_label:
            display = self._buffer_text if self._buffer_text else "_"
            self._buffer_label.set_markup(
                f'<span font="18" foreground="rgba(255,255,255,0.9)">'
                f'{GLib.markup_escape_text(display)}</span>'
            )

    def update_confidence(self, confidence: float, letter: str):
        """Update the confidence bar and detected letter."""
        self._confidence = confidence
        self._detected_letter = letter
        GLib.idle_add(self._do_update_confidence)

    def _do_update_confidence(self):
        if self._confidence_bar:
            self._confidence_bar.set_fraction(min(self._confidence, 1.0))

            # Color the bar based on confidence level
            bar = self._confidence_bar
            bar.remove_css_class("low")
            bar.remove_css_class("medium")
            bar.remove_css_class("high")
            if self._confidence < 0.5:
                bar.add_css_class("low")
            elif self._confidence < 0.8:
                bar.add_css_class("medium")
            else:
                bar.add_css_class("high")

        if self._conf_percent_label:
            self._conf_percent_label.set_text(f"{self._confidence:.0%}")

        if self._detected_label:
            if self._detected_letter:
                color = "#2ecc71" if self._confidence >= 0.8 else "#fff"
                self._detected_label.set_markup(
                    f'<span font="42" foreground="{color}" weight="heavy">'
                    f'{GLib.markup_escape_text(self._detected_letter.upper())}</span>'
                )
            else:
                self._detected_label.set_markup(
                    '<span font="42" foreground="rgba(255,255,255,0.2)">—</span>'
                )

    def update_mode(self, mode: str):
        """Update the always-visible mode indicator."""
        self._mode = mode.upper()
        GLib.idle_add(self._do_update_mode)

    def _do_update_mode(self):
        self._update_mode_label(self._mode)

    def show_notification(self, message: str, level: str = "info",
                          duration_ms: int = 3000):
        """Show a toast notification."""
        GLib.idle_add(self._do_show_notification, message, level, duration_ms)

    def _do_show_notification(self, message, level, duration_ms):
        if not self._notification_label:
            return

        self._notification_label.remove_css_class("error")
        self._notification_label.remove_css_class("warning")
        if level == "error":
            self._notification_label.add_css_class("error")
        elif level == "warning":
            self._notification_label.add_css_class("warning")

        self._notification_label.set_markup(
            f'<span font="12">{GLib.markup_escape_text(message)}</span>'
        )
        self._notification_label.set_visible(True)

        if self._notification_timeout_id:
            GLib.source_remove(self._notification_timeout_id)
        self._notification_timeout_id = GLib.timeout_add(
            duration_ms, self._hide_notification
        )

    def _hide_notification(self):
        if self._notification_label:
            self._notification_label.set_visible(False)
        self._notification_timeout_id = None
        return False

    def show(self):
        GLib.idle_add(self._do_show)

    def _do_show(self):
        if self._window:
            self._window.present()

    def hide(self):
        GLib.idle_add(self._do_hide)

    def _do_hide(self):
        if self._window:
            self._window.set_visible(False)

    def quit(self):
        GLib.idle_add(self._do_quit)

    def _do_quit(self):
        if self._app:
            self._app.quit()


# Quick self-test
if __name__ == "__main__":
    print(f"GTK4: {HAS_GTK}, Layer Shell: {HAS_LAYER_SHELL}")
    overlay = BufferOverlay()
    overlay.start()

    import time
    time.sleep(1)

    overlay.update_mode("TYPING")
    for i in range(20):
        conf = (i + 1) / 20
        letter = chr(65 + i % 26)
        overlay.update_confidence(conf, letter)
        time.sleep(0.3)

    overlay.update_buffer("hello world")
    overlay.show_notification("Test notification!", "success", 2000)
    time.sleep(3)

    overlay.quit()
    print("✓ Overlay test complete")

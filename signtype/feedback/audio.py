"""Visual notification system — all feedback for deaf/HoH users.

Replaces audio/TTS entirely. All system state communication happens
through visual notifications displayed on the overlay or logged to
the terminal. This project is designed for deaf and hard-of-hearing
users — audio output is never used.
"""

import queue
import threading
import time
from dataclasses import dataclass, field


@dataclass
class Notification:
    """A visual notification to display."""
    message: str
    level: str = "info"  # "info", "success", "warning", "error"
    duration_ms: int = 3000
    timestamp: float = field(default_factory=time.time)


class VisualFeedback:
    """Visual notification system for deaf/HoH users.

    All feedback is visual:
    - Overlay notifications (toast-style messages on the buffer overlay)
    - Terminal logging (for debugging and setup flows)

    No audio is ever produced.
    """

    def __init__(self):
        self._queue: queue.Queue[Notification | None] = queue.Queue()
        self._running = False
        self._thread = None
        self._active_notifications: list[Notification] = []
        self._lock = threading.Lock()

        # Callback to display notifications on the overlay
        # Set by the application: (Notification) -> None
        self.on_notification: None | callable = None

    def start(self):
        """Start the notification processor thread."""
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        """Process notifications from the queue."""
        while self._running:
            try:
                notification = self._queue.get(timeout=0.5)
            except queue.Empty:
                self._expire_notifications()
                continue

            if notification is None:  # Shutdown signal
                break

            notif: Notification = notification  # type narrowing

            with self._lock:
                self._active_notifications.append(notif)

            # Send to overlay
            if self.on_notification:
                self.on_notification(notif)

            # Also log to terminal for debugging
            icon = {"info": "ℹ", "success": "✓", "warning": "⚠", "error": "✗"}.get(
                notif.level, "•"
            )
            print(f"[SignType] {icon} {notif.message}")

            self._expire_notifications()

    def _expire_notifications(self):
        """Remove expired notifications."""
        now = time.time()
        with self._lock:
            self._active_notifications = [
                n for n in self._active_notifications
                if (now - n.timestamp) * 1000 < n.duration_ms
            ]

    def notify(self, message: str, level: str = "info", duration_ms: int = 3000):
        """Queue a visual notification."""
        self._queue.put(Notification(message=message, level=level, duration_ms=duration_ms))

    # --- Pre-built notifications ---

    def announce_mode_switch(self, mode: str):
        self.notify(f"→ {mode.upper()} mode", level="info", duration_ms=2000)

    def announce_startup(self):
        self.notify("SignType starting up...", level="info", duration_ms=3000)

    def announce_ready(self):
        self.notify(
            "Ready! Show both open palms to enter typing mode.",
            level="success", duration_ms=5000,
        )

    def announce_first_run(self):
        self.notify(
            "Welcome to SignType — first-time setup needed.",
            level="info", duration_ms=5000,
        )

    def announce_recording_start(self, gesture_name: str):
        self.notify(
            f"Recording: {gesture_name} — perform gesture for 5 seconds",
            level="info", duration_ms=5000,
        )

    def announce_recording_done(self, gesture_name: str):
        self.notify(
            f"✓ Recorded: {gesture_name}",
            level="success", duration_ms=3000,
        )

    def announce_training_start(self):
        self.notify("Training model — please wait...", level="info", duration_ms=5000)

    def announce_training_done(self, accuracy: float = None):
        msg = f"Training complete — accuracy: {accuracy:.0%}" if accuracy else "Training complete"
        self.notify(msg, level="success", duration_ms=4000)

    def announce_error(self, message: str):
        self.notify(f"Error: {message}", level="error", duration_ms=5000)

    def announce_command_fired(self, gesture_name: str, command_label: str):
        self.notify(f"⚡ {command_label}", level="success", duration_ms=2000)

    def announce_text_injected(self, text: str):
        self.notify(f"Typed: {text}", level="success", duration_ms=2000)

    @property
    def active_notifications(self) -> list[Notification]:
        with self._lock:
            return list(self._active_notifications)

    def stop(self):
        self._running = False
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=2.0)


# Quick self-test
if __name__ == "__main__":
    feedback = VisualFeedback()
    feedback.start()

    feedback.announce_startup()
    time.sleep(1)
    feedback.announce_mode_switch("typing")
    time.sleep(1)
    feedback.announce_recording_start("wave")
    time.sleep(1)
    feedback.announce_recording_done("wave")
    time.sleep(1)
    feedback.announce_error("test error message")
    time.sleep(2)

    feedback.stop()
    print("✓ Visual feedback test complete (all output was visual/text)")

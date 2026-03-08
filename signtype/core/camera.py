"""Camera capture module — threaded webcam reader with frame queue."""

import threading
import queue
import cv2


class Camera:
    """Threaded webcam capture. Drops old frames to keep latency low."""

    def __init__(self, camera_index: int = 0, max_queue_size: int = 2):
        self.camera_index = camera_index
        self.max_queue_size = max_queue_size
        self._cap = None
        self._frame_queue = queue.Queue(maxsize=max_queue_size)
        self._running = False
        self._thread = None

    def start(self):
        """Open camera and start capture thread."""
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {self.camera_index}. "
                "Check config.json 'camera_index' setting."
            )
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        """Continuously read frames and push to queue, dropping old frames."""
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue
            # Drop oldest frame if queue is full
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self._frame_queue.put(frame)

    def read(self, timeout: float = 1.0):
        """Get the latest frame. Returns numpy array (BGR) or None on timeout."""
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """Stop capture thread and release camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_running(self) -> bool:
        return self._running


# Quick self-test
if __name__ == "__main__":
    cam = Camera(0)
    cam.start()
    frame = cam.read()
    if frame is not None:
        print(f"✓ Camera working — frame shape: {frame.shape}")
    else:
        print("✗ No frame received within timeout")
    cam.stop()

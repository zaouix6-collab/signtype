"""Dynamic gesture classifier — LSTM for motion-based gestures."""

import os
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


class DynamicGestureLSTM(nn.Module if nn else object):
    """Small LSTM network for classifying gesture sequences.

    Operates on a rolling buffer of 30 frames of 63-dim landmark vectors.
    Input shape: (batch, 30, 63)
    """

    SEQUENCE_LENGTH = 30

    def __init__(self, input_size: int = 63, hidden_size: int = 64,
                 num_layers: int = 2, num_classes: int = 4):
        if nn is None:
            raise ImportError("PyTorch is not installed. Install with: "
                              "pip install torch --index-url https://download.pytorch.org/whl/cpu")
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        # Take the last time step output
        out = self.fc(out[:, -1, :])
        return out


class DynamicClassifier:
    """Wrapper for the LSTM dynamic gesture classifier.

    Built-in gesture classes:
        0: mode_switch  (both open palms)
        1: delete       (downward swipe with index finger)
        2: confirm      (closed fist hold)
        3: enter_idle   (double open palm)
    """

    BUILTIN_CLASSES = ["mode_switch", "delete", "confirm", "enter_idle"]

    def __init__(self, model_path: str = None, num_classes: int = 4):
        self._model = None
        self._model_path = model_path
        self._classes = self.BUILTIN_CLASSES[:num_classes]
        self._frame_buffer = []

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, model_path: str):
        """Load a trained LSTM model."""
        if torch is None:
            return
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        num_classes = checkpoint.get("num_classes", 4)
        self._model = DynamicGestureLSTM(num_classes=num_classes)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()
        self._classes = checkpoint.get("classes", self.BUILTIN_CLASSES[:num_classes])
        self._model_path = model_path

    def save(self, model_path: str):
        """Save model to disk."""
        if self._model is None:
            return
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "num_classes": len(self._classes),
            "classes": self._classes,
        }, model_path)

    def add_frame(self, landmarks: np.ndarray):
        """Add a frame of landmarks to the rolling buffer.

        Args:
            landmarks: 63-dim numpy array from LandmarkExtractor.
        """
        self._frame_buffer.append(landmarks)
        if len(self._frame_buffer) > DynamicGestureLSTM.SEQUENCE_LENGTH:
            self._frame_buffer.pop(0)

    def predict(self) -> tuple[str, float]:
        """Classify the current frame buffer.

        Returns:
            Tuple of (gesture_name, confidence).
            Returns ("none", 0.0) if buffer is not full or model not loaded.
        """
        if not self.is_loaded or torch is None:
            return ("none", 0.0)

        if len(self._frame_buffer) < DynamicGestureLSTM.SEQUENCE_LENGTH:
            return ("none", 0.0)

        sequence = np.array(self._frame_buffer)  # (30, 63)
        tensor = torch.FloatTensor(sequence).unsqueeze(0)  # (1, 30, 63)

        with torch.no_grad():
            output = self._model(tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            confidence, predicted = torch.max(probabilities, 0)

        idx = predicted.item()
        label = self._classes[idx] if idx < len(self._classes) else "none"
        return (label, float(confidence))

    def clear_buffer(self):
        """Clear the frame buffer."""
        self._frame_buffer.clear()

    @property
    def buffer_length(self) -> int:
        return len(self._frame_buffer)


# Quick self-test
if __name__ == "__main__":
    clf = DynamicClassifier()
    print(f"Model loaded: {clf.is_loaded}")
    print(f"Classes: {clf._classes}")
    print(f"Buffer length: {clf.buffer_length}")

    # Fill buffer with dummy data
    for _ in range(30):
        clf.add_frame(np.random.rand(63))
    print(f"Buffer full: {clf.buffer_length}")

    label, conf = clf.predict()
    print(f"Prediction (no model): {label} ({conf:.2f})")
    print("✓ Dynamic classifier test complete")

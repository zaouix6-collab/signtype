"""Static command gesture classifier — MLP for user-defined gestures."""

import os
import pickle
import numpy as np


class GestureClassifier:
    """MLP classifier for static command gestures.

    Similar to FingerspellClassifier but trained on user-recorded
    custom gesture samples. Classes are dynamically defined by
    the user's gesture bindings.
    """

    def __init__(self, model_path: str = None):
        self._model = None
        self._model_path = model_path
        self._classes = []
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def classes(self) -> list[str]:
        return self._classes

    def load(self, model_path: str):
        """Load a trained model and its class list from disk."""
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            self._model = data["model"]
            self._classes = data.get("classes", [])
        else:
            # Legacy format: just the model
            self._model = data
            self._classes = []

        self._model_path = model_path

    def save(self, model_path: str):
        """Save model and class list to disk."""
        with open(model_path, "wb") as f:
            pickle.dump({"model": self._model, "classes": self._classes}, f)
        self._model_path = model_path

    def predict(self, landmarks: np.ndarray) -> tuple[str, float]:
        """Classify a 63-dim landmark vector.

        Returns:
            Tuple of (gesture_name, confidence). Returns ("unknown", 0.0) if
            model is not loaded or no classes defined.
        """
        if not self.is_loaded or not self._classes:
            return ("unknown", 0.0)

        X = landmarks.reshape(1, -1)
        prediction = self._model.predict(X)[0]
        probabilities = self._model.predict_proba(X)[0]
        confidence = float(probabilities.max())

        if isinstance(prediction, (int, np.integer)):
            label = self._classes[prediction] if prediction < len(self._classes) else "unknown"
        else:
            label = str(prediction)

        return (label, confidence)

    def set_model(self, model, classes: list[str]):
        """Set a freshly trained model and its class list."""
        self._model = model
        self._classes = classes


# Quick self-test
if __name__ == "__main__":
    clf = GestureClassifier()
    print(f"Model loaded: {clf.is_loaded}")
    print(f"Classes: {clf.classes}")

    dummy = np.random.rand(63)
    label, conf = clf.predict(dummy)
    print(f"Prediction (no model): {label} ({conf:.2f})")
    print("✓ Gesture classifier test complete")

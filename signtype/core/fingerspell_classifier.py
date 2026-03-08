"""Finger spelling classifier — MLP for ASL letter recognition."""

import os
import pickle
import numpy as np


class FingerspellClassifier:
    """MLP classifier for ASL finger spelling using preprocessed landmarks.

    Expects 63-dimensional landmark vectors as input.
    Returns (letter, confidence) predictions.
    """

    # ASL Alphabet classes (26 letters + space, delete, nothing)
    CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE", "DELETE", "NOTHING"]

    def __init__(self, model_path: str = None):
        self._model = None
        self._model_path = model_path
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, model_path: str):
        """Load a trained model from disk."""
        with open(model_path, "rb") as f:
            self._model = pickle.load(f)
        self._model_path = model_path

    def predict(self, landmarks: np.ndarray) -> tuple[str, float]:
        """Classify a 63-dim landmark vector.

        Args:
            landmarks: Normalized 63-dim numpy array from LandmarkExtractor.

        Returns:
            Tuple of (predicted_letter, confidence_score).
            Returns ("NOTHING", 0.0) if model is not loaded.
        """
        if not self.is_loaded:
            return ("NOTHING", 0.0)

        # Reshape for single sample prediction
        X = landmarks.reshape(1, -1)
        prediction = self._model.predict(X)[0]
        probabilities = self._model.predict_proba(X)[0]
        confidence = float(probabilities.max())

        # Map class index to label
        if isinstance(prediction, (int, np.integer)):
            label = self.CLASSES[prediction] if prediction < len(self.CLASSES) else "NOTHING"
        else:
            label = str(prediction)

        return (label, confidence)

    def predict_top_k(self, landmarks: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """Return top-k predictions with confidence scores."""
        if not self.is_loaded:
            return [("NOTHING", 0.0)]

        X = landmarks.reshape(1, -1)
        probabilities = self._model.predict_proba(X)[0]
        top_indices = np.argsort(probabilities)[::-1][:k]

        results = []
        for idx in top_indices:
            label = self.CLASSES[idx] if idx < len(self.CLASSES) else "NOTHING"
            results.append((label, float(probabilities[idx])))

        return results


# Quick self-test
if __name__ == "__main__":
    clf = FingerspellClassifier()
    print(f"Model loaded: {clf.is_loaded}")

    # Test with dummy data
    dummy = np.random.rand(63)
    label, conf = clf.predict(dummy)
    print(f"Prediction (no model): {label} ({conf:.2f})")
    print("✓ Fingerspell classifier test complete")

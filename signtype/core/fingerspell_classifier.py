"""Finger spelling classifier — Random Forest for ASL letter recognition.

Uses enhanced 129-feature vectors computed from MediaPipe landmarks:
- 63 raw landmark coordinates
- 21 wrist distances
- 20 consecutive joint distances
- 5 fingertip-to-palm distances  
- 20 inter-joint angles
"""

import os
import pickle
import numpy as np


def _compute_features(landmarks_63: np.ndarray) -> np.ndarray:
    """Convert raw 63-dim landmarks into 129-dim feature vector."""
    pts = landmarks_63.reshape(21, 3)
    features = []
    
    # Raw coordinates (63)
    features.extend(landmarks_63)
    
    # Distance from each joint to wrist (21)
    wrist = pts[0]
    for i in range(21):
        features.append(np.linalg.norm(pts[i] - wrist))
    
    # Consecutive joint distances (20)
    for i in range(20):
        features.append(np.linalg.norm(pts[i + 1] - pts[i]))
    
    # Fingertip to palm center distance (5)
    palm_center = np.mean(pts[[0, 1, 5, 9, 13, 17]], axis=0)
    for tip_idx in [4, 8, 12, 16, 20]:
        features.append(np.linalg.norm(pts[tip_idx] - palm_center))
    
    # Angles between finger segments (20)
    finger_joints = [
        [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
        [13, 14, 15, 16], [17, 18, 19, 20],
    ]
    for finger in finger_joints:
        for j in range(len(finger) - 1):
            v1 = pts[finger[0]] - wrist if j == 0 else pts[finger[j]] - pts[finger[j - 1]]
            v2 = pts[finger[j + 1]] - pts[finger[j]]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            features.append(np.arccos(np.clip(cos_a, -1, 1)))
    
    return np.array(features, dtype=np.float64)


class FingerspellClassifier:
    """Random Forest classifier for ASL finger spelling.
    
    Supports both the new 129-feature enhanced model and the
    legacy 63-dim MLP model for backward compatibility.
    """

    CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "del", "nothing"]

    def __init__(self, model_path: str = None):
        self._model = None
        self._model_path = model_path
        self._feature_type = "raw_63"  # or "enhanced_129"
        self._classes = self.CLASSES
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, model_path: str):
        """Load a trained model from disk."""
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        
        # Support both old (plain model) and new (dict with metadata) formats
        if isinstance(data, dict):
            self._model = data["model"]
            self._feature_type = data.get("feature_type", "raw_63")
            self._classes = data.get("classes", self.CLASSES)
        else:
            # Legacy format: model object directly
            self._model = data
            self._feature_type = "raw_63"
        
        self._model_path = model_path

    def predict(self, landmarks: np.ndarray) -> tuple[str, float]:
        """Classify a 63-dim landmark vector.

        Returns (predicted_letter, confidence_score).
        """
        if not self.is_loaded:
            return ("nothing", 0.0)

        # Compute features based on model type
        if self._feature_type == "enhanced_129":
            X = _compute_features(landmarks).reshape(1, -1)
        else:
            X = landmarks.reshape(1, -1)
        
        # Handle NaN/inf from bad landmarks
        if not np.isfinite(X).all():
            return ("nothing", 0.0)
        
        prediction = self._model.predict(X)[0]
        probabilities = self._model.predict_proba(X)[0]
        confidence = float(probabilities.max())

        if isinstance(prediction, (int, np.integer)):
            label = self._classes[prediction] if prediction < len(self._classes) else "nothing"
        else:
            label = str(prediction)

        return (label, confidence)

    def predict_top_k(self, landmarks: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """Return top-k predictions with confidence scores."""
        if not self.is_loaded:
            return [("nothing", 0.0)]

        if self._feature_type == "enhanced_129":
            X = _compute_features(landmarks).reshape(1, -1)
        else:
            X = landmarks.reshape(1, -1)
        
        if not np.isfinite(X).all():
            return [("nothing", 0.0)]
        
        probabilities = self._model.predict_proba(X)[0]
        top_indices = np.argsort(probabilities)[::-1][:k]

        results = []
        for idx in top_indices:
            label = self._classes[idx] if idx < len(self._classes) else "nothing"
            results.append((label, float(probabilities[idx])))
        return results


if __name__ == "__main__":
    clf = FingerspellClassifier()
    print(f"Model loaded: {clf.is_loaded}")
    dummy = np.random.rand(63)
    label, conf = clf.predict(dummy)
    print(f"Prediction (no model): {label} ({conf:.2f})")
    print("✓ Fingerspell classifier test complete")

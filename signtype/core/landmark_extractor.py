"""MediaPipe Hands landmark extractor — outputs normalized 63-dim vectors.

Uses the new MediaPipe Tasks API (v0.10.14+).
Requires the hand_landmarker.task model file, which is free and
licensed under Apache 2.0 by Google.
"""

import os
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    mp = None


class LandmarkExtractor:
    """Wraps MediaPipe HandLandmarker to extract 21 keypoint landmarks from BGR frames.

    Returns a 63-dimensional numpy array (21 joints × 3 coordinates: x, y, z)
    normalized to [0, 1] range relative to the hand bounding box.
    """

    # Default model location relative to project root
    DEFAULT_MODEL_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "hand_landmarker.task"
    )

    # Free download URL (Apache 2.0 license)
    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    )

    def __init__(
        self,
        model_path: str = "",
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ):
        if mp is None:
            raise ImportError(
                "mediapipe is not installed. "
                "Install it with: pip install mediapipe "
                "(requires Python 3.12 or earlier)"
            )

        model_path = model_path or self.DEFAULT_MODEL_PATH

        if not os.path.exists(model_path):
            self._download_model(model_path)

        # Use IMAGE mode for static images, VIDEO mode for sequential frames
        if static_image_mode:
            running_mode = mp_vision.RunningMode.IMAGE
        else:
            running_mode = mp_vision.RunningMode.VIDEO

        base_options = mp_python.BaseOptions(
            model_asset_path=model_path
        )
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._static_mode = static_image_mode
        self._frame_count = 0

    def _download_model(self, dest_path: str):
        """Download the free hand_landmarker.task model from Google."""
        import urllib.request
        print(f"Downloading hand landmarker model to {dest_path}...")
        print(f"  Source: {self.MODEL_URL}")
        print("  License: Apache 2.0 (free)")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        urllib.request.urlretrieve(self.MODEL_URL, dest_path)
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  ✓ Downloaded ({size_mb:.1f} MB)")

    def extract(self, bgr_frame: np.ndarray) -> list[np.ndarray]:
        """Extract landmarks from a BGR frame.

        Args:
            bgr_frame: OpenCV BGR image (numpy array).

        Returns:
            List of 63-dim numpy arrays, one per detected hand.
            Empty list if no hands detected.
        """
        import cv2

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        if self._static_mode:
            results = self._landmarker.detect(mp_image)
        else:
            self._frame_count += 1
            results = self._landmarker.detect_for_video(
                mp_image, self._frame_count
            )

        if not results.hand_landmarks:
            return []

        hands = []
        for hand_landmarks in results.hand_landmarks:
            landmarks = self._normalize_landmarks(hand_landmarks)
            hands.append(landmarks)

        return hands

    def _normalize_landmarks(self, hand_landmarks) -> np.ndarray:
        """Convert MediaPipe hand landmarks to a normalized 63-dim vector.

        Normalizes relative to the hand's bounding box to make the
        representation invariant to hand position and scale in the frame.
        """
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
        )  # shape: (21, 3)

        # Normalize to bounding box of the hand
        min_vals = coords.min(axis=0)
        max_vals = coords.max(axis=0)
        range_vals = max_vals - min_vals
        # Avoid division by zero
        range_vals[range_vals == 0] = 1.0
        normalized = (coords - min_vals) / range_vals

        return normalized.flatten()  # shape: (63,)

    def extract_single(self, bgr_frame: np.ndarray) -> np.ndarray | None:
        """Extract landmarks from the first detected hand only.

        Returns:
            63-dim numpy array, or None if no hand detected.
        """
        hands = self.extract(bgr_frame)
        return hands[0] if hands else None

    def close(self):
        """Release MediaPipe resources."""
        self._landmarker.close()


# Quick self-test
if __name__ == "__main__":
    import cv2 as cv

    print("Testing landmark extraction on webcam...")
    cap = cv.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        extractor = LandmarkExtractor(static_image_mode=True)
        landmarks = extractor.extract_single(frame)
        if landmarks is not None:
            print(f"✓ Landmarks extracted — shape: {landmarks.shape}")
            print(f"  Values range: [{landmarks.min():.3f}, {landmarks.max():.3f}]")
        else:
            print("✗ No hand detected in frame (try holding hand in front of camera)")
        extractor.close()
    else:
        print("✗ Could not read frame from camera")

"""MediaPipe Hands landmark extractor — outputs normalized 63-dim vectors."""

import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None


class LandmarkExtractor:
    """Wraps MediaPipe Hands to extract 21 keypoint landmarks from BGR frames.

    Returns a 63-dimensional numpy array (21 joints × 3 coordinates: x, y, z)
    normalized to [0, 1] range relative to the hand bounding box.
    """

    def __init__(
        self,
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
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

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
        results = self._hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return []

        hands = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = self._normalize_landmarks(hand_landmarks)
            hands.append(landmarks)

        return hands

    def _normalize_landmarks(self, hand_landmarks) -> np.ndarray:
        """Convert MediaPipe hand landmarks to a normalized 63-dim vector.

        Normalizes relative to the hand's bounding box to make the
        representation invariant to hand position and scale in the frame.
        """
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
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
        self._hands.close()


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

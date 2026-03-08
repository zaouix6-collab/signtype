"""Gesture recorder — captures live landmark samples for custom gestures.

Records ~5 seconds of landmarks at full frame rate from the webcam,
storing 30-frame sequences for LSTM training or single-frame samples
for MLP training.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.camera import Camera
from core.landmark_extractor import LandmarkExtractor


class GestureRecorder:
    """Records gesture samples from the webcam.

    For static gestures: captures individual landmark frames.
    For dynamic gestures: captures 30-frame sliding window sequences.
    """

    SEQUENCE_LENGTH = 30  # Frames per dynamic gesture sequence

    def __init__(self, camera: Camera = None, extractor: LandmarkExtractor = None):
        self._camera = camera
        self._extractor = extractor
        self._owns_camera = camera is None

    def _ensure_resources(self):
        """Create camera and extractor if not provided."""
        if self._camera is None:
            self._camera = Camera(0)
            self._camera.start()
        if self._extractor is None:
            self._extractor = LandmarkExtractor(static_image_mode=False)

    def record_static(
        self,
        gesture_name: str,
        duration_seconds: float = 5.0,
        output_dir: str = None,
    ) -> str:
        """Record static gesture samples (individual frames).

        Args:
            gesture_name: Name of the gesture class.
            duration_seconds: How long to record.
            output_dir: Where to save samples.

        Returns:
            Path to saved .npy file, or None on failure.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if output_dir is None:
            output_dir = os.path.join(base_dir, "data", "custom_gestures", gesture_name)

        os.makedirs(output_dir, exist_ok=True)
        self._ensure_resources()

        print(f"Recording '{gesture_name}' for {duration_seconds}s...")
        print("Hold the gesture steady in front of the camera.")

        samples = []
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            frame = self._camera.read(timeout=0.5)
            if frame is None:
                continue

            landmark = self._extractor.extract_single(frame)
            if landmark is not None:
                samples.append(landmark)

            elapsed = time.time() - start_time
            print(f"  {elapsed:.1f}s / {duration_seconds}s — {len(samples)} samples", end="\r")

        print()

        if not samples:
            print(f"✗ No landmarks captured for '{gesture_name}'")
            return None

        # Save
        output_path = os.path.join(output_dir, f"samples_{int(time.time())}.npy")
        np.save(output_path, np.array(samples))
        print(f"✓ Saved {len(samples)} samples to {output_path}")

        return output_path

    def record_dynamic(
        self,
        gesture_name: str,
        num_sequences: int = 10,
        duration_seconds: float = 5.0,
        output_dir: str = None,
    ) -> str:
        """Record dynamic gesture sequences (30-frame windows).

        Captures a continuous stream and extracts sliding-window sequences.

        Args:
            gesture_name: Name of the gesture class.
            num_sequences: Target number of 30-frame sequences to capture.
            duration_seconds: Recording duration.
            output_dir: Where to save sequences.

        Returns:
            Path to output directory, or None on failure.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if output_dir is None:
            output_dir = os.path.join(base_dir, "data", "custom_gestures", gesture_name)

        os.makedirs(output_dir, exist_ok=True)
        self._ensure_resources()

        print(f"Recording '{gesture_name}' for {duration_seconds}s...")
        print("Perform the motion gesture repeatedly.")

        all_frames = []
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            frame = self._camera.read(timeout=0.5)
            if frame is None:
                continue

            landmark = self._extractor.extract_single(frame)
            if landmark is not None:
                all_frames.append(landmark)

            elapsed = time.time() - start_time
            print(f"  {elapsed:.1f}s / {duration_seconds}s — {len(all_frames)} frames", end="\r")

        print()

        if len(all_frames) < self.SEQUENCE_LENGTH:
            print(f"✗ Not enough frames ({len(all_frames)} < {self.SEQUENCE_LENGTH})")
            return None

        # Extract sliding window sequences
        sequences = []
        step = max(1, (len(all_frames) - self.SEQUENCE_LENGTH) // num_sequences)
        for i in range(0, len(all_frames) - self.SEQUENCE_LENGTH + 1, step):
            seq = all_frames[i:i + self.SEQUENCE_LENGTH]
            sequences.append(np.array(seq))
            if len(sequences) >= num_sequences:
                break

        # Save each sequence as a separate .npy file
        saved = 0
        timestamp = int(time.time())
        for idx, seq in enumerate(sequences):
            output_path = os.path.join(output_dir, f"seq_{timestamp}_{idx:03d}.npy")
            np.save(output_path, seq)
            saved += 1

        print(f"✓ Saved {saved} sequences ({self.SEQUENCE_LENGTH} frames each) to {output_dir}")
        return output_dir

    def cleanup(self):
        """Release resources if we own them."""
        if self._owns_camera and self._camera:
            self._camera.stop()
        if self._extractor:
            self._extractor.close()


if __name__ == "__main__":
    recorder = GestureRecorder()

    import argparse
    parser = argparse.ArgumentParser(description="Record gesture samples")
    parser.add_argument("name", help="Gesture name")
    parser.add_argument("--type", choices=["static", "dynamic"], default="static")
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()

    if args.type == "static":
        recorder.record_static(args.name, args.duration)
    else:
        recorder.record_dynamic(args.name, duration_seconds=args.duration)

    recorder.cleanup()

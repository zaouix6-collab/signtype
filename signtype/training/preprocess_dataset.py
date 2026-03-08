"""Dataset preprocessing — extract MediaPipe landmarks from ASL Alphabet images.

Walks data/asl_alphabet_raw/ subdirectories (one per letter class),
runs MediaPipe Hands on each image, extracts the 21 keypoint landmark
vector, and saves the result as numpy arrays in data/landmarks/.
"""

import os
import sys
import numpy as np
import cv2

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.landmark_extractor import LandmarkExtractor


def preprocess_dataset(
    raw_dir: str = None,
    output_dir: str = None,
    max_per_class: int = None,
):
    """Extract landmarks from all images in the ASL Alphabet dataset.

    Args:
        raw_dir: Path to asl_alphabet_raw/ with subdirectories per class.
        output_dir: Path to save landmark .npy files.
        max_per_class: Optional limit on samples per class (for testing).
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if raw_dir is None:
        raw_dir = os.path.join(base_dir, "data", "asl_alphabet_raw")
    if output_dir is None:
        output_dir = os.path.join(base_dir, "data", "landmarks")

    if not os.path.exists(raw_dir):
        print(f"✗ Dataset not found at: {raw_dir}")
        print("  Download the ASL Alphabet dataset from Kaggle:")
        print("  https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        print(f"  Extract it to: {raw_dir}")
        return False

    # Auto-detect Kaggle's nested structure:
    # asl_alphabet_raw/asl_alphabet_train/asl_alphabet_train/A/
    nested = os.path.join(raw_dir, "asl_alphabet_train", "asl_alphabet_train")
    if os.path.isdir(nested):
        print(f"Auto-detected Kaggle nested structure, using: {nested}")
        raw_dir = nested

    os.makedirs(output_dir, exist_ok=True)

    extractor = LandmarkExtractor(static_image_mode=True, max_num_hands=1)

    # Get class directories
    classes = sorted([
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    ])

    if not classes:
        print(f"✗ No class subdirectories found in {raw_dir}")
        return False

    print(f"Found {len(classes)} classes: {', '.join(classes)}")
    print()

    total_extracted = 0
    total_failed = 0

    for class_name in classes:
        # Resume support: skip classes that already have a .npy file
        output_path = os.path.join(output_dir, f"{class_name}.npy")
        if os.path.exists(output_path):
            existing = np.load(output_path)
            count = len(existing)
            print(f"  ✓ {class_name}: {count} landmarks already saved — skipping")
            total_extracted += count
            continue

        class_dir = os.path.join(raw_dir, class_name)
        images = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        if max_per_class:
            images = images[:max_per_class]

        landmarks_list = []
        failed = 0

        for i, img_name in enumerate(images):
            img_path = os.path.join(class_dir, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                failed += 1
                continue

            landmark = extractor.extract_single(frame)
            if landmark is not None:
                landmarks_list.append(landmark)
            else:
                failed += 1

            # Progress indicator
            if (i + 1) % 100 == 0 or i == len(images) - 1:
                print(
                    f"  {class_name}: {i + 1}/{len(images)} "
                    f"({len(landmarks_list)} extracted, {failed} failed)",
                    end="\r",
                )

        print()

        if landmarks_list:
            output_path = os.path.join(output_dir, f"{class_name}.npy")
            np.save(output_path, np.array(landmarks_list))
            print(f"  ✓ {class_name}: {len(landmarks_list)} landmarks saved")
        else:
            print(f"  ✗ {class_name}: No landmarks extracted!")

        total_extracted += len(landmarks_list)
        total_failed += failed

    extractor.close()

    print()
    print(f"=== Preprocessing complete ===")
    print(f"Total extracted: {total_extracted}")
    print(f"Total failed: {total_failed}")
    print(f"Success rate: {total_extracted / (total_extracted + total_failed):.1%}")
    print(f"Saved to: {output_dir}")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess ASL Alphabet dataset")
    parser.add_argument("--raw-dir", help="Path to raw dataset directory")
    parser.add_argument("--output-dir", help="Path to save landmarks")
    parser.add_argument("--max-per-class", type=int, help="Max samples per class")
    args = parser.parse_args()

    success = preprocess_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        max_per_class=args.max_per_class,
    )
    sys.exit(0 if success else 1)

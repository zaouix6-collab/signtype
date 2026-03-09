"""Train finger spelling MLP — trains on preprocessed landmark data.

Loads landmark vectors from data/landmarks/, splits 80/20,
trains a scikit-learn MLPClassifier, validates accuracy,
and saves to data/model_fingerspell.pkl if accuracy >= 97%.
"""

import os
import sys
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def train_fingerspell(
    landmarks_dir: str = None,
    output_path: str = None,
    min_accuracy: float = 0.97,
):
    """Train the finger spelling MLP classifier.

    Args:
        landmarks_dir: Path to directory with class .npy files.
        output_path: Path to save the trained model.
        min_accuracy: Minimum accuracy to save model (default 97%).

    Returns:
        Tuple of (accuracy, model_path) if successful, (accuracy, None) if below threshold.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if landmarks_dir is None:
        landmarks_dir = os.path.join(base_dir, "data", "landmarks")
    if output_path is None:
        output_path = os.path.join(base_dir, "data", "model_fingerspell.pkl")

    if not os.path.exists(landmarks_dir):
        print(f"✗ Landmarks directory not found: {landmarks_dir}")
        print("  Run preprocess_dataset.py first.")
        return (0.0, None)

    # Load all landmark files
    X_all = []
    y_all = []
    classes = []

    npy_files = sorted([f for f in os.listdir(landmarks_dir) if f.endswith(".npy")])

    if not npy_files:
        print(f"✗ No .npy files found in {landmarks_dir}")
        return (0.0, None)

    for npy_file in npy_files:
        class_name = os.path.splitext(npy_file)[0]
        classes.append(class_name)
        data = np.load(os.path.join(landmarks_dir, npy_file))
        X_all.append(data)
        y_all.extend([class_name] * len(data))
        print(f"  Loaded {class_name}: {len(data)} samples")

    X = np.vstack(X_all).astype(np.float64)
    y = np.array(y_all)

    print(f"\nTotal samples: {len(X)}")
    print(f"Classes: {len(classes)}")
    print(f"Feature dimension: {X.shape[1]}")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Train MLP
    print("\nTraining MLPClassifier...")
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42,
        early_stopping=False,
        tol=1e-4,
        verbose=True,
    )
    clf.fit(X_train, y_train)

    # Validate
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"\n=== Results ===")
    print(f"Validation accuracy: {accuracy:.4f} ({accuracy:.1%})")
    print()
    print(classification_report(y_val, y_pred))

    if accuracy >= min_accuracy:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(clf, f)
        print(f"✓ Model saved to {output_path}")
        print(f"  Accuracy {accuracy:.1%} meets threshold {min_accuracy:.0%}")
        return (accuracy, output_path)
    else:
        print(f"✗ Accuracy {accuracy:.1%} below threshold {min_accuracy:.0%}")
        print("  Model NOT saved. Check dataset integrity.")
        print("  Tips:")
        print("  - Ensure all images have detectable hands")
        print("  - Try re-running preprocessing with higher detection confidence")
        print("  - Consider augmenting the dataset")
        return (accuracy, None)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train finger spelling MLP")
    parser.add_argument("--landmarks-dir", help="Path to landmarks directory")
    parser.add_argument("--output", help="Path to save model")
    parser.add_argument("--min-accuracy", type=float, default=0.97,
                        help="Minimum accuracy threshold")
    args = parser.parse_args()

    accuracy, path = train_fingerspell(
        landmarks_dir=args.landmarks_dir,
        output_path=args.output,
        min_accuracy=args.min_accuracy,
    )
    sys.exit(0 if path else 1)

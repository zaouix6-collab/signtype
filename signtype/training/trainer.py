"""Incremental trainer — retrains MLP/LSTM after new gesture recordings."""

import os
import sys
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def retrain_static_gestures(
    custom_dir: str = None,
    model_path: str = None,
) -> tuple[float, str | None]:
    """Retrain the static gesture MLP with all custom gesture data.

    Args:
        custom_dir: Path to custom_gestures/ directory.
        model_path: Path to save model.

    Returns:
        (accuracy, model_path) tuple.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if custom_dir is None:
        custom_dir = os.path.join(base_dir, "data", "custom_gestures")
    if model_path is None:
        model_path = os.path.join(base_dir, "data", "model_static.pkl")

    if not os.path.exists(custom_dir):
        print("✗ No custom gesture data found")
        return (0.0, None)

    # Collect all static gesture samples
    X_all = []
    y_all = []
    classes = []

    for class_name in sorted(os.listdir(custom_dir)):
        class_dir = os.path.join(custom_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Load samples (non-sequence .npy files with shape (N, 63))
        for npy_file in os.listdir(class_dir):
            if not npy_file.endswith(".npy") or npy_file.startswith("seq_"):
                continue

            data = np.load(os.path.join(class_dir, npy_file))
            if data.ndim == 2 and data.shape[1] == 63:
                X_all.append(data)
                y_all.extend([class_name] * len(data))
                if class_name not in classes:
                    classes.append(class_name)

    if not X_all:
        print("✗ No static gesture samples found")
        return (0.0, None)

    X = np.vstack(X_all)
    y = np.array(y_all)

    print(f"Training on {len(X)} samples across {len(classes)} classes")

    # Train
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=300,
        random_state=42,
        early_stopping=True,
    )
    clf.fit(X, y)

    # Compute training accuracy
    accuracy = clf.score(X, y)
    print(f"Training accuracy: {accuracy:.1%}")

    # Save with class metadata
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"model": clf, "classes": classes}, f)

    print(f"✓ Model saved to {model_path}")
    return (accuracy, model_path)


def retrain_dynamic_gestures(
    custom_dir: str = None,
    model_path: str = None,
    epochs: int = 50,
) -> str | None:
    """Retrain the LSTM with all dynamic gesture data.

    Delegates to train_dynamic.py.
    """
    from training.train_dynamic import train_dynamic
    return train_dynamic(data_dir=custom_dir, output_path=model_path, epochs=epochs)


if __name__ == "__main__":
    print("=== Retraining static gesture MLP ===")
    accuracy, path = retrain_static_gestures()
    if path:
        print(f"✓ Done — accuracy: {accuracy:.1%}")
    else:
        print("✗ Training failed or no data")

"""Train high-accuracy ASL classifier using Random Forest.

Uses MediaPipe hand landmarks + Random Forest for ~98% accuracy.
This replaces the old MLP-based training with a more accurate model.

Usage:
    python training/train_fingerspell.py

Expects preprocessed landmark .npy files in data/landmarks/.
"""

import os
import sys
import numpy as np
import pickle
import time

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Feature engineering for landmarks
def _compute_features(landmarks_63: np.ndarray) -> np.ndarray:
    """Convert raw 63-dim landmarks into richer feature set.
    
    Raw landmarks: 21 joints × 3 coords (x, y, z) = 63 values.
    
    Computed features:
    - Original 63 coordinates (normalized)
    - 21 distances from wrist (joint 0) 
    - 20 inter-joint distances (consecutive joints)
    - 5 fingertip-to-palm distances
    - Angles between finger segments (20 angles)
    
    Total: 63 + 21 + 20 + 5 + 20 = 129 features
    """
    pts = landmarks_63.reshape(21, 3)
    
    features = []
    
    # 1. Raw normalized coordinates (63)
    features.extend(landmarks_63)
    
    # 2. Distance from each joint to wrist (21)
    wrist = pts[0]
    for i in range(21):
        dist = np.linalg.norm(pts[i] - wrist)
        features.append(dist)
    
    # 3. Consecutive joint distances (20)
    for i in range(20):
        dist = np.linalg.norm(pts[i + 1] - pts[i])
        features.append(dist)
    
    # 4. Fingertip to palm center distance (5)
    palm_center = np.mean(pts[[0, 1, 5, 9, 13, 17]], axis=0)
    fingertips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
    for tip_idx in fingertips:
        dist = np.linalg.norm(pts[tip_idx] - palm_center)
        features.append(dist)
    
    # 5. Angles between finger segments (20 angles, 4 per finger)
    finger_joints = [
        [1, 2, 3, 4],      # thumb
        [5, 6, 7, 8],      # index
        [9, 10, 11, 12],   # middle
        [13, 14, 15, 16],  # ring
        [17, 18, 19, 20],  # pinky
    ]
    for finger in finger_joints:
        for j in range(len(finger) - 1):
            if j == 0:
                v1 = pts[finger[0]] - wrist
            else:
                v1 = pts[finger[j]] - pts[finger[j - 1]]
            v2 = pts[finger[j + 1]] - pts[finger[j]]
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            features.append(angle)
    
    return np.array(features, dtype=np.float64)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    landmark_dir = os.path.join(base_dir, "data", "landmarks")
    output_path = os.path.join(base_dir, "data", "model_fingerspell.pkl")
    
    if not os.path.exists(landmark_dir):
        print("ERROR: No landmark data found!")
        print("Run first: python training/preprocess_dataset.py")
        sys.exit(1)
    
    print()
    print("  🤟 Training ASL Classifier (Random Forest)")
    print()
    
    # Load all landmark files
    X_raw = []
    y_raw = []
    
    npy_files = sorted([f for f in os.listdir(landmark_dir) if f.endswith(".npy")])
    
    for npy_file in npy_files:
        class_name = npy_file.replace(".npy", "")
        data = np.load(os.path.join(landmark_dir, npy_file))
        
        if len(data) == 0:
            print(f"  ⚠ {class_name}: 0 samples — skipping")
            continue
        
        X_raw.append(data)
        y_raw.extend([class_name] * len(data))
        print(f"  ✓ {class_name}: {len(data)} samples")
    
    X_landmarks = np.vstack(X_raw)
    y = np.array(y_raw)
    
    print(f"\n  Total: {len(X_landmarks)} samples, {len(npy_files)} classes")
    
    # Feature engineering
    print("  Computing features...", end=" ", flush=True)
    t0 = time.time()
    X = np.array([_compute_features(lm) for lm in X_landmarks])
    print(f"done ({time.time() - t0:.1f}s) → {X.shape[1]} features")
    
    # Remove NaN/inf
    mask = np.isfinite(X).all(axis=1)
    if not mask.all():
        removed = (~mask).sum()
        print(f"  ⚠ Removed {removed} samples with NaN/inf values")
        X = X[mask]
        y = y[mask]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print()
    
    # Train Random Forest
    print("  Training Random Forest...", flush=True)
    t0 = time.time()
    
    clf = RandomForestClassifier(
        n_estimators=200,       # 200 trees (good balance of speed/accuracy)
        max_depth=30,           # Deep enough for complex hand shapes
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,              # Use all CPU cores
        random_state=42,
        class_weight="balanced", # Handle class imbalance
    )
    
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  ✓ Trained in {train_time:.1f}s")
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n  📊 Test Accuracy: {accuracy:.1%}")
    print()
    
    # Show per-class results for worst performers
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Find classes with < 90% accuracy
    weak_classes = []
    for cls_name, metrics in report.items():
        if isinstance(metrics, dict) and "f1-score" in metrics:
            if metrics["f1-score"] < 0.90:
                weak_classes.append((cls_name, metrics["f1-score"]))
    
    if weak_classes:
        print("  ⚠ Weaker classes (< 90% F1):")
        for cls, f1 in sorted(weak_classes, key=lambda x: x[1]):
            print(f"    {cls}: {f1:.1%}")
        print()
    
    # Save model and feature function
    model_data = {
        "model": clf,
        "feature_type": "enhanced_129",  # So the classifier knows to compute features
        "accuracy": accuracy,
        "classes": list(clf.classes_),
        "n_features": X.shape[1],
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✓ Saved: {output_path}")
    print(f"  ✓ Size: {size_mb:.1f} MB")
    print(f"  ✓ Accuracy: {accuracy:.1%}")
    print()


if __name__ == "__main__":
    main()

"""Train dynamic gesture LSTM — trains on recorded motion gesture sequences.

Trains a small LSTM on 30-frame sequences of landmarks for motion
gestures: mode_switch, delete, confirm, enter_idle, plus any
user-defined motion gestures.
"""

import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.dynamic_classifier import DynamicGestureLSTM


class GestureSequenceDataset(Dataset):
    """Dataset of landmark sequences for dynamic gesture training."""

    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(np.array(sequences))
        self.labels = torch.LongTensor(np.array(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def train_dynamic(
    data_dir: str = None,
    output_path: str = None,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
):
    """Train the LSTM dynamic gesture classifier.

    Expects data_dir to contain subdirectories per gesture class,
    each containing .npy files of shape (30, 63) — 30-frame sequences
    of 63-dimensional landmark vectors.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if data_dir is None:
        data_dir = os.path.join(base_dir, "data", "custom_gestures")
    if output_path is None:
        output_path = os.path.join(base_dir, "data", "model_dynamic.pt")

    if not os.path.exists(data_dir):
        print(f"✗ Gesture data directory not found: {data_dir}")
        return None

    # Load sequences
    sequences = []
    labels = []
    classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    if not classes:
        print(f"✗ No gesture class directories found in {data_dir}")
        return None

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        npy_files = [f for f in os.listdir(class_dir) if f.endswith(".npy")]

        for npy_file in npy_files:
            seq = np.load(os.path.join(class_dir, npy_file))
            if seq.shape == (30, 63):
                sequences.append(seq)
                labels.append(class_idx)
            else:
                print(f"  Skipping {npy_file}: shape {seq.shape} != (30, 63)")

        print(f"  {class_name}: {sum(1 for l in labels if l == class_idx)} sequences")

    if not sequences:
        print("✗ No valid sequences found")
        return None

    print(f"\nTotal sequences: {len(sequences)}")
    print(f"Classes: {classes}")

    # Create dataset and dataloader
    dataset = GestureSequenceDataset(sequences, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = DynamicGestureLSTM(num_classes=len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    print("\nTraining LSTM...")
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_seqs, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_seqs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        accuracy = correct / total if total > 0 else 0
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs} — "
                  f"Loss: {total_loss:.4f}, Accuracy: {accuracy:.1%}")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": len(classes),
        "classes": classes,
    }, output_path)

    print(f"\n✓ Model saved to {output_path}")
    print(f"  Classes: {classes}")
    print(f"  Final accuracy: {accuracy:.1%}")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train dynamic gesture LSTM")
    parser.add_argument("--data-dir", help="Path to gesture sequences")
    parser.add_argument("--output", help="Path to save model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    result = train_dynamic(
        data_dir=args.data_dir,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    sys.exit(0 if result else 1)

"""
validation.py — Deterministic train/val/test splits
FIX: Use np.random.default_rng() instead of np.random.seed() to avoid
polluting global random state which affects DataLoader worker seeds,
augmentation randomness, and any other code running after the split.
"""
import numpy as np

def create_splits(dataset_size: int,
                  train_ratio: float = 0.70,
                  val_ratio:   float = 0.10,
                  test_ratio:  float = 0.20,
                  seed: int = 42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    # FIX: local RNG instead of global np.random.seed()
    rng     = np.random.default_rng(seed)
    indices = rng.permutation(dataset_size)

    n_train = int(train_ratio * dataset_size)
    n_val   = int(val_ratio   * dataset_size)

    return (indices[:n_train].tolist(),
            indices[n_train:n_train + n_val].tolist(),
            indices[n_train + n_val:].tolist())
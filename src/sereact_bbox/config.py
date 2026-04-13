"""
config.py — Centralized Configuration
Using plain classes (not @dataclass) so attributes are class-level and
accessible as ModelConfig.in_channels without instantiation.
"""

DEFAULT_CHECKPOINT = "best_model.pth"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_ONNX_DIR   = "onnx_export"


class ModelConfig:
    in_channels : int   = 7
    k_neighbors : int   = 20
    dropout     : float = 0.4


class DataConfig:
    num_points      : int   = 1024
    is_train_split  : float = 0.70
    val_split       : float = 0.10
    test_split      : float = 0.20
    depth_threshold : float = 0.01


class TrainConfig:
    batch_size    : int   = 32
    epochs        : int   = 100
    learning_rate : float = 1e-3   # peak LR after warm-up — DO NOT exceed 2e-3
    weight_decay  : float = 1e-4
    warmup_epochs : int   = 5      # ramp from lr/warmup_epochs to learning_rate
    min_lr        : float = 1e-6
    grad_clip     : float = 1.0
    num_workers   : int   = 24
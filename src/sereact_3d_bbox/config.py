import os
from dataclasses import dataclass, field

@dataclass
class DataConfig:
    num_points: int = 1024
    mad_threshold: float = 5.0      # For the residual MAD filter
    min_points_threshold: int = 10  # Minimum points required to process an object

@dataclass
class ModelConfig:
    in_channels: int = 7            # XYZ (3) + RGB (3) + Mask (1)

@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    save_path: str = "best_model.pth"

@dataclass
class InferenceConfig:
    weights: str = "best_model.pth"
    out_dir: str = "output"

@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

# Global default configuration instance
cfg = AppConfig()

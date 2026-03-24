"""
config.py — Central configuration for the Deepfake Detection project.
All hyperparameters, paths, and model settings live here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
@dataclass
class PathConfig:
    # Root dataset directory
    # Expected structure:
    #   data_root/
    #     train/real/*.mp4
    #     train/fake/*.mp4
    #     val/real/*.mp4
    #     val/fake/*.mp4
    #     test/real/*.mp4
    #     test/fake/*.mp4
    data_root: Path = Path("data")

    # Where to save model checkpoints
    checkpoint_dir: Path = Path("checkpoints")

    # TensorBoard / CSV logs
    log_dir: Path = Path("logs")

    # Results from test.py
    results_dir: Path = Path("results")

    def __post_init__(self):
        for p in [self.checkpoint_dir, self.log_dir, self.results_dir]:
            p.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Data / Preprocessing
# ─────────────────────────────────────────────
@dataclass
class DataConfig:
    # Number of frames sampled per video
    num_frames: int = 16

    # Face crop resolution fed to EfficientNet
    face_size: int = 224

    # Margin around detected face bbox (fraction)
    face_margin: float = 0.3

    # DataLoader workers
    num_workers: int = 4

    # Use MTCNN face detection (recommended), else full frame resize
    use_face_detection: bool = True

    # Fraction for train split when no explicit val folder exists
    val_split: float = 0.15

    # ImageNet stats (used for EfficientNet pretrained)
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)

    # Supported video extensions
    video_extensions: List[str] = field(
        default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv"]
    )

    # Augmentation probabilities (train only)
    aug_hflip_p: float = 0.5
    aug_color_jitter_p: float = 0.4
    aug_blur_p: float = 0.2
    aug_noise_p: float = 0.2
    aug_compression_p: float = 0.3    # simulate re-compression artifacts
    aug_cutout_p: float = 0.2

    # Mixup alpha (0 = disabled)
    mixup_alpha: float = 0.2


# ─────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────
@dataclass
class ModelConfig:
    # EfficientNet variant: b0 … b7 (b4 is the sweet spot)
    backbone: str = "efficientnet_b4"

    # Pretrained weights source: "imagenet" or path to .pth file
    pretrained: str = "imagenet"

    # Freeze backbone layers for first N epochs (0 = never freeze)
    freeze_backbone_epochs: int = 3

    # ─── GRU ───
    gru_hidden_dim: int = 512
    gru_num_layers: int = 2
    gru_dropout: float = 0.3
    gru_bidirectional: bool = True          # doubles effective hidden size

    # ─── Attention ───
    use_temporal_attention: bool = True     # self-attention over GRU outputs
    attention_heads: int = 8

    # ─── Classification Head ───
    # Dimensions of FC layers after GRU (last layer = 1)
    classifier_dims: List[int] = field(
        default_factory=lambda: [512, 256, 128]
    )
    classifier_dropout: float = 0.5

    # ─── Multi-scale features ───
    # Extract features from multiple EfficientNet stages and fuse
    use_multi_scale: bool = True
    # Stage indices inside EfficientNet to tap (0-indexed)
    multi_scale_stages: List[int] = field(default_factory=lambda: [2, 4, 6])

    # ─── Auxiliary head ───
    # Frame-level auxiliary loss to improve feature learning
    use_aux_loss: bool = True
    aux_loss_weight: float = 0.3


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Batch size (number of video clips)
    batch_size: int = 2

    # Total epochs
    epochs: int = 30

    # ─── Optimizer ───
    optimizer: str = "adamw"          # "adamw" | "sgd"
    lr: float = 1e-4
    backbone_lr_multiplier: float = 0.1   # backbone gets lr * multiplier
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # ─── Scheduler ───
    scheduler: str = "cosine_warmup"  # "cosine_warmup" | "step" | "plateau"
    warmup_epochs: int = 3
    min_lr: float = 1e-6
    step_size: int = 10               # for StepLR
    step_gamma: float = 0.1

    # ─── Loss ───
    loss: str = "bce"                 # "bce" | "focal"
    label_smoothing: float = 0.05
    focal_gamma: float = 2.0          # used only when loss == "focal"

    # Class weights for imbalanced datasets [real_weight, fake_weight]
    # Set to None to disable
    class_weights: Optional[List[float]] = None

    # ─── Regularisation ───
    use_gradient_checkpointing: bool = False   # saves VRAM, slows training

    # ─── Evaluation ───
    eval_every_n_epochs: int = 1
    early_stopping_patience: int = 7

    # ─── Logging ───
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "deepfake-detection"
    wandb_run_name: Optional[str] = None

    # ─── Reproducibility ───
    seed: int = 42

    # ─── Hardware ───
    device: str = "auto"       # "auto" | "cuda" | "cpu" | "mps"
    amp: bool = True           # Automatic Mixed Precision (FP16)
    num_gpus: int = 1          # >1 enables DataParallel


# ─────────────────────────────────────────────
# Test / Inference
# ─────────────────────────────────────────────
@dataclass
class TestConfig:
    # Path to checkpoint for evaluation / prediction
    checkpoint_path: Optional[Path] = None

    # TTA: test-time augmentation (horizontal flip ensemble)
    use_tta: bool = True
    tta_flips: int = 2         # number of augmented views to ensemble

    # Sliding window inference for long videos
    use_sliding_window: bool = False
    window_size: int = 16      # frames per window
    window_stride: int = 8     # overlap

    # Decision threshold
    threshold: float = 0.5

    # Save per-frame scores to CSV
    save_frame_scores: bool = True

    # Output visualisation (annotated video)
    save_annotated_video: bool = False
    annotated_video_fps: int = 10


# ─────────────────────────────────────────────
# Master Config
# ─────────────────────────────────────────────
@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    test: TestConfig = field(default_factory=TestConfig)


# ── Singleton for import convenience ──────────
cfg = Config()


if __name__ == "__main__":
    import json, dataclasses

    def _to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, Path):
            return str(obj)
        return obj

    print(json.dumps(_to_dict(cfg), indent=2))
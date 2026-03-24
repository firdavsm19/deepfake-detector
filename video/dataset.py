"""
dataset.py — Video dataset for deepfake detection.

Supports:
  • Automatic real/fake label discovery from folder structure
  • MTCNN face detection + crop with margin
  • Uniform frame sampling + sliding window
  • Rich augmentation pipeline (train) / clean pipeline (val/test)
  • Mixup collate function
  • CSV manifest support
"""

import csv
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("[dataset] facenet-pytorch not found — falling back to full-frame resize.")

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import cfg, DataConfig

# ──────────────────────────────────────────────────────────────
# Face Extractor
# ──────────────────────────────────────────────────────────────

class FaceExtractor:
    """
    Extracts a single face crop from an RGB numpy frame.
    Falls back to centre-crop resize if MTCNN is unavailable or no face found.
    """

    def __init__(self, face_size: int = 224, margin: float = 0.3, device: str = "cpu"):
        self.face_size = face_size
        self.margin = margin
        self.mtcnn = None

        if MTCNN_AVAILABLE:
            self.mtcnn = MTCNN(
                image_size=face_size,
                margin=int(face_size * margin),
                keep_all=False,
                select_largest=True,
                device=device,
                post_process=False,   # return uint8 tensor
            )

    def extract(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Returns a (H, W, 3) uint8 numpy face crop, or resized full frame.
        """
        if self.mtcnn is not None:
            try:
                from PIL import Image
                pil = Image.fromarray(frame_rgb)
                face_tensor = self.mtcnn(pil)   # (3, H, W) or None
                if face_tensor is not None:
                    face_np = face_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
                    return face_np
            except Exception:
                pass  # silent fallback

        # Fallback: resize full frame
        return cv2.resize(frame_rgb, (self.face_size, self.face_size))


# ──────────────────────────────────────────────────────────────
# Augmentation Pipelines
# ──────────────────────────────────────────────────────────────

def _build_train_transform(dcfg: DataConfig) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=dcfg.aug_hflip_p),
        A.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.05,
            p=dcfg.aug_color_jitter_p,
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=5),
        ], p=dcfg.aug_blur_p),
        A.GaussNoise(var_limit=(10, 50), p=dcfg.aug_noise_p),
        # Simulate JPEG compression artifacts common in deepfakes
        A.ImageCompression(quality_lower=40, quality_upper=95, p=dcfg.aug_compression_p),
        A.CoarseDropout(
            max_holes=4, max_height=32, max_width=32,
            fill_value=0, p=dcfg.aug_cutout_p,
        ),
        A.Normalize(mean=dcfg.mean, std=dcfg.std),
        ToTensorV2(),
    ])


def _build_eval_transform(dcfg: DataConfig) -> A.Compose:
    return A.Compose([
        A.Normalize(mean=dcfg.mean, std=dcfg.std),
        ToTensorV2(),
    ])


def _build_tta_transforms(dcfg: DataConfig) -> List[A.Compose]:
    """Two TTA views: original + horizontal flip."""
    base = [A.Normalize(mean=dcfg.mean, std=dcfg.std), ToTensorV2()]
    return [
        A.Compose(base),
        A.Compose([A.HorizontalFlip(p=1.0)] + base),
    ]


# ──────────────────────────────────────────────────────────────
# Video Sampler
# ──────────────────────────────────────────────────────────────

def sample_frame_indices(
    total_frames: int,
    num_frames: int,
    mode: str = "uniform",
    stride: int = 1,
    window_start: Optional[int] = None,
) -> List[int]:
    """
    mode:
      'uniform'  — evenly spaced across the whole video
      'random'   — uniform with small random jitter (for training)
      'window'   — fixed-size sliding window starting at window_start
    """
    if mode == "window" and window_start is not None:
        end = min(window_start + num_frames * stride, total_frames)
        indices = list(range(window_start, end, stride))
    elif mode == "random" and total_frames > num_frames:
        step = total_frames / num_frames
        indices = [
            min(int(i * step + random.uniform(0, step * 0.5)), total_frames - 1)
            for i in range(num_frames)
        ]
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    # Pad if not enough frames
    while len(indices) < num_frames:
        indices.append(indices[-1])

    return indices[:num_frames]


# ──────────────────────────────────────────────────────────────
# Core Dataset
# ──────────────────────────────────────────────────────────────

class DeepfakeVideoDataset(Dataset):
    """
    Loads videos and returns (frames_tensor, label) pairs.

    Expected folder structure:
        split/real/*.mp4
        split/fake/*.mp4

    Or pass a CSV manifest with columns: path, label  (0=real, 1=fake)
    """

    LABEL_MAP = {"real": 0, "fake": 1}

    def __init__(
        self,
        root: Path,
        split: str = "train",               # "train" | "val" | "test"
        dcfg: DataConfig = None,
        transform: Optional[Callable] = None,
        manifest_csv: Optional[Path] = None,
        face_extractor: Optional[FaceExtractor] = None,
        use_sliding_window: bool = False,
        window_size: int = 16,
        window_stride: int = 8,
    ):
        self.dcfg = dcfg or cfg.data
        self.split = split
        self.transform = transform or (
            _build_train_transform(self.dcfg) if split == "train"
            else _build_eval_transform(self.dcfg)
        )
        self.face_extractor = face_extractor
        self.use_sliding_window = use_sliding_window
        self.window_size = window_size
        self.window_stride = window_stride

        # ── Load manifest ──────────────────────────────────
        if manifest_csv is not None:
            self.samples = self._load_csv(manifest_csv)
        else:
            split_dir = root / split
            self.samples = self._discover_videos(split_dir)

        if len(self.samples) == 0:
            raise RuntimeError(f"No videos found in {root / split}")

        # Sliding-window expansion
        if self.use_sliding_window:
            self.samples = self._expand_sliding_window(self.samples)

        print(
            f"[dataset] {split}: {len(self.samples)} samples "
            f"({sum(1 for _, l in self.samples if l == 1)} fake, "
            f"{sum(1 for _, l in self.samples if l == 0)} real)"
        )

    # ── Discovery helpers ─────────────────────────────────────
    def _discover_videos(self, split_dir: Path) -> List[Tuple[Path, int]]:
        samples = []
        for cls_name, label in self.LABEL_MAP.items():
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                print(f"[dataset] Warning: {cls_dir} not found, skipping.")
                continue
            for ext in self.dcfg.video_extensions:
                for p in cls_dir.glob(f"*{ext}"):
                    samples.append((p, label))
        return samples

    def _load_csv(self, csv_path: Path) -> List[Tuple[Path, int]]:
        samples = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append((Path(row["path"]), int(row["label"])))
        return samples

    def _expand_sliding_window(
        self, samples: List[Tuple[Path, int]]
    ) -> List[Tuple]:
        expanded = []
        for path, label in samples:
            cap = cv2.VideoCapture(str(path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if total <= 0:
                continue
            starts = range(0, max(1, total - self.window_size), self.window_stride)
            for s in starts:
                expanded.append((path, label, s))
        return expanded

    # ── Video loading ─────────────────────────────────────────
    def _load_frames(self, video_path: Path, window_start: Optional[int] = None) -> torch.Tensor:
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            cap.release()
            return torch.zeros(self.dcfg.num_frames, 3, self.dcfg.face_size, self.dcfg.face_size)

        sampling_mode = "random" if self.split == "train" else "uniform"
        indices = sample_frame_indices(
            total_frames=total,
            num_frames=self.window_size if self.use_sliding_window else self.dcfg.num_frames,
            mode="window" if window_start is not None else sampling_mode,
            window_start=window_start,
        )

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frames.append(frames[-1] if frames else np.zeros(
                    (self.dcfg.face_size, self.dcfg.face_size, 3), dtype=np.uint8
                ))
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Face detection
            if self.face_extractor is not None:
                frame_rgb = self.face_extractor.extract(frame_rgb)
            else:
                frame_rgb = cv2.resize(frame_rgb, (self.dcfg.face_size, self.dcfg.face_size))

            frames.append(frame_rgb)

        cap.release()

        # Apply augmentation per-frame
        tensor_frames = []
        for f in frames:
            aug = self.transform(image=f)["image"]   # (C, H, W)
            tensor_frames.append(aug)

        return torch.stack(tensor_frames)   # (T, C, H, W)

    # ── Dataset interface ─────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.samples[idx]
        if len(item) == 3:
            path, label, window_start = item
        else:
            path, label = item
            window_start = None

        frames = self._load_frames(path, window_start)
        return frames, torch.tensor(label, dtype=torch.float32)

    # ── Class weights for imbalanced datasets ─────────────────
    def class_counts(self) -> Dict[int, int]:
        counts = {0: 0, 1: 0}
        for item in self.samples:
            lbl = item[1]
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts

    def make_sampler(self) -> WeightedRandomSampler:
        counts = self.class_counts()
        total = sum(counts.values())
        weights_per_class = {c: total / (len(counts) * n) for c, n in counts.items()}
        sample_weights = [weights_per_class[item[1]] for item in self.samples]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )


# ──────────────────────────────────────────────────────────────
# Mixup Collate
# ──────────────────────────────────────────────────────────────

def mixup_collate(alpha: float = 0.2):
    """
    Returns a collate_fn that applies Mixup augmentation to a batch.
    If alpha == 0, returns standard collate.
    """
    from torch.utils.data.dataloader import default_collate

    def _collate(batch):
        frames, labels = default_collate(batch)  # (B, T, C, H, W), (B,)
        if alpha <= 0 or not torch.is_grad_enabled():
            return frames, labels
        lam = np.random.beta(alpha, alpha)
        perm = torch.randperm(frames.size(0))
        mixed_frames = lam * frames + (1 - lam) * frames[perm]
        mixed_labels = lam * labels + (1 - lam) * labels[perm]
        return mixed_frames, mixed_labels

    return _collate


# ──────────────────────────────────────────────────────────────
# DataLoader Factory
# ──────────────────────────────────────────────────────────────

def build_dataloaders(
    dcfg: DataConfig = None,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders.
    """
    dcfg = dcfg or cfg.data
    root = cfg.paths.data_root

    face_extractor = FaceExtractor(
        face_size=dcfg.face_size,
        margin=dcfg.face_margin,
    ) if dcfg.use_face_detection else None

    train_ds = DeepfakeVideoDataset(
        root=root, split="train", dcfg=dcfg, face_extractor=face_extractor
    )
    val_ds = DeepfakeVideoDataset(
        root=root, split="val", dcfg=dcfg,
        transform=_build_eval_transform(dcfg),
        face_extractor=face_extractor,
    )
    test_ds = DeepfakeVideoDataset(
        root=root, split="test", dcfg=dcfg,
        transform=_build_eval_transform(dcfg),
        face_extractor=face_extractor,
    )

    sampler = train_ds.make_sampler() if use_weighted_sampler else None
    tcfg = cfg.train

    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=dcfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=mixup_collate(dcfg.mixup_alpha),
        persistent_workers=dcfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg.batch_size,
        shuffle=False,
        num_workers=dcfg.num_workers,
        pin_memory=True,
        persistent_workers=dcfg.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=tcfg.batch_size,
        shuffle=False,
        num_workers=dcfg.num_workers,
        pin_memory=True,
        persistent_workers=dcfg.num_workers > 0,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick dataset smoke-test (no actual data required for import)
    print("Dataset module loaded successfully.")
    print(f"MTCNN available: {MTCNN_AVAILABLE}")
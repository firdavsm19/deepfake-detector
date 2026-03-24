%%writefile /content/train.py

"""
train.py — Full training pipeline for the EfficientNet + GRU deepfake detector.

Features:
  • Cosine annealing with linear warmup
  • Differential learning rates (backbone vs head)
  • Backbone freeze-then-unfreeze schedule
  • Gradient clipping + AMP (FP16)
  • Early stopping
  • Best-model & latest-checkpoint saving
  • TensorBoard + optional W&B logging
  • Reproducible seeding
"""

import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast          # ← FIXED: was torch.cuda.amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import cfg, TrainConfig, ModelConfig
from dataset import build_dataloaders
from model import DeepfakeDetector, DeepfakeLoss


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    metrics: dict,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    scaler: Optional[GradScaler] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, dict]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt.get("epoch", 0), ckpt.get("metrics", {})


# ──────────────────────────────────────────────────────────────
# Scheduler with Warmup
# ──────────────────────────────────────────────────────────────

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup for `warmup_epochs` then cosine decay to `min_lr`.
    """

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-6):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            scale = (e + 1) / max(self.warmup_epochs, 1)
        else:
            progress = (e - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        return [
            max(base_lr * scale, self.min_lr)
            for base_lr in self.base_lrs
        ]


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────

class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self._data: Dict[str, list] = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self._data.setdefault(k, []).append(float(v))

    def mean(self, key: str) -> float:
        vals = self._data.get(key, [])
        return sum(vals) / len(vals) if vals else 0.0

    def summary(self) -> dict:
        return {k: self.mean(k) for k in self._data}


def compute_metrics(
    all_logits: torch.Tensor,
    all_labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    probs = torch.sigmoid(all_logits)
    preds = (probs >= threshold).long()
    labels = all_labels.long()

    acc = (preds == labels).float().mean().item()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    # AUC via trapezoidal approximation
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels.numpy(), probs.numpy())
    except Exception:
        auc = 0.0

    return dict(
        acc=acc, precision=precision, recall=recall,
        f1=f1, specificity=specificity, auc=auc,
    )


# ──────────────────────────────────────────────────────────────
# Train / Eval Loops
# ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    loss_fn: DeepfakeLoss,
    scaler: GradScaler,
    device: torch.device,
    tcfg: TrainConfig,
    epoch: int,
) -> dict:
    model.train()
    tracker = MetricTracker()
    t0 = time.time()

    for step, (frames, labels) in enumerate(loader):
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=tcfg.amp and device.type == "cuda"):  # ← FIXED
            clip_logits, frame_logits = model(frames)
            loss, loss_info = loss_fn(clip_logits, frame_logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        preds = (torch.sigmoid(clip_logits.detach()) >= 0.5).float()
        acc = (preds == labels).float().mean().item()

        tracker.update(
            loss=loss.item(),
            acc=acc,
            **loss_info,
        )

        if step % 20 == 0:
            lr = optimizer.param_groups[-1]["lr"]
            elapsed = time.time() - t0
            print(
                f"  [train] epoch={epoch} step={step}/{len(loader)}"
                f"  loss={loss.item():.4f}  acc={acc:.3f}"
                f"  lr={lr:.2e}  elapsed={elapsed:.1f}s"
            )

    return tracker.summary()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: DeepfakeLoss,
    device: torch.device,
    tcfg: TrainConfig,
) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0

    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast("cuda", enabled=tcfg.amp and device.type == "cuda"):  # ← FIXED
            clip_logits, _ = model(frames)
            loss, _ = loss_fn(clip_logits, None, labels)

        total_loss += loss.item()
        all_logits.append(clip_logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(loader)
    return metrics


# ──────────────────────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 7, mode: str = "max", delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best = -math.inf if mode == "max" else math.inf
        self.counter = 0
        self.triggered = False

    def step(self, value: float) -> bool:
        """Returns True if training should stop."""
        improved = (
            (self.mode == "max" and value > self.best + self.delta) or
            (self.mode == "min" and value < self.best - self.delta)
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ──────────────────────────────────────────────────────────────
# Main Training Function
# ──────────────────────────────────────────────────────────────

def train(resume_from: Optional[Path] = None, loaders=None):  # ← FIXED: added loaders param
    tcfg = cfg.train
    mcfg = cfg.model
    dcfg = cfg.data
    paths = cfg.paths

    set_seed(tcfg.seed)
    device = resolve_device(tcfg.device)
    print(f"\n{'='*60}")
    print(f"  Deepfake Detector — Training")
    print(f"  Device : {device}")
    print(f"  Backbone: {mcfg.backbone}")
    print(f"  Epochs : {tcfg.epochs}")
    print(f"{'='*60}\n")

    # ── DataLoaders ───────────────────────────────────────────
    # FIXED: use provided loaders (e.g. fast-run subset) if given
    if loaders is not None:
        train_loader, val_loader, _ = loaders
        print(f"  Using provided loaders: {len(train_loader.dataset)} train samples")
    else:
        train_loader, val_loader, _ = build_dataloaders(dcfg)

    # ── Model ─────────────────────────────────────────────────
    model = DeepfakeDetector(mcfg).to(device)
    counts = model.param_count()
    print(f"Parameters — total: {counts['total']:,}  trainable: {counts['trainable']:,}\n")

    if tcfg.num_gpus > 1 and device.type == "cuda":
        model = nn.DataParallel(model)

    # ── Loss ──────────────────────────────────────────────────
    loss_fn = DeepfakeLoss(
        loss_type=tcfg.loss,
        label_smoothing=tcfg.label_smoothing,
        aux_weight=mcfg.aux_loss_weight,
        focal_gamma=tcfg.focal_gamma,
        class_weights=tcfg.class_weights,
    )

    # ── Optimizer with differential LR ────────────────────────
    raw_model = model.module if hasattr(model, "module") else model
    param_groups = raw_model.get_param_groups(
        lr=tcfg.lr,
        backbone_multiplier=tcfg.backbone_lr_multiplier,
    )
    if tcfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(param_groups, weight_decay=tcfg.weight_decay)
    else:
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=tcfg.weight_decay)

    # ── Scheduler ─────────────────────────────────────────────
    if tcfg.scheduler == "cosine_warmup":
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_epochs=tcfg.warmup_epochs,
            total_epochs=tcfg.epochs,
            min_lr=tcfg.min_lr,
        )
    elif tcfg.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=tcfg.step_size, gamma=tcfg.step_gamma
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5
        )

    scaler = GradScaler("cuda", enabled=tcfg.amp and device.type == "cuda")  # ← FIXED
    early_stop = EarlyStopping(patience=tcfg.early_stopping_patience, mode="max")

    # ── Resume ────────────────────────────────────────────────
    start_epoch = 0
    best_val_auc = 0.0
    if resume_from and resume_from.exists():
        start_epoch, prev_metrics = load_checkpoint(
            resume_from, raw_model, optimizer, scheduler, scaler, device
        )
        best_val_auc = prev_metrics.get("auc", 0.0)
        print(f"Resumed from {resume_from} at epoch {start_epoch}, best AUC={best_val_auc:.4f}")

    # ── Logging ───────────────────────────────────────────────
    writer = None
    if tcfg.use_tensorboard:
        writer = SummaryWriter(log_dir=str(paths.log_dir))

    wandb_run = None
    if tcfg.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=tcfg.wandb_project,
                name=tcfg.wandb_run_name,
                config={"model": mcfg.__dict__, "train": tcfg.__dict__, "data": dcfg.__dict__},
            )
        except ImportError:
            print("[train] wandb not installed — skipping W&B logging.")

    # ── Training Loop ─────────────────────────────────────────
    history = []

    for epoch in range(start_epoch, tcfg.epochs):
        print(f"\n── Epoch {epoch + 1}/{tcfg.epochs} " + "─" * 40)

        # Backbone freeze schedule
        if epoch < mcfg.freeze_backbone_epochs:
            raw_model.freeze_backbone()
        elif epoch == mcfg.freeze_backbone_epochs:
            print("  Unfreezing backbone weights.")
            raw_model.unfreeze_backbone()
            for pg in optimizer.param_groups:
                pg["initial_lr"] = pg["lr"]

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device, tcfg, epoch + 1
        )

        # Step scheduler
        if tcfg.scheduler == "plateau":
            pass  # stepped after eval
        else:
            scheduler.step()

        # Evaluate
        if (epoch + 1) % tcfg.eval_every_n_epochs == 0:
            val_metrics = evaluate(model, val_loader, loss_fn, device, tcfg)

            if tcfg.scheduler == "plateau":
                scheduler.step(val_metrics["auc"])

            print(
                f"  [val]  loss={val_metrics['loss']:.4f}  "
                f"acc={val_metrics['acc']:.3f}  "
                f"auc={val_metrics['auc']:.4f}  "
                f"f1={val_metrics['f1']:.4f}"
            )

            # TensorBoard
            if writer:
                for k, v in train_metrics.items():
                    writer.add_scalar(f"train/{k}", v, epoch)
                for k, v in val_metrics.items():
                    writer.add_scalar(f"val/{k}", v, epoch)
                writer.add_scalar("lr", optimizer.param_groups[-1]["lr"], epoch)

            # W&B
            if wandb_run:
                wandb_run.log(
                    {"train": train_metrics, "val": val_metrics, "epoch": epoch + 1}
                )

            # Save best
            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_path = paths.checkpoint_dir / "best_model.pth"
                save_checkpoint(best_path, raw_model, optimizer, scheduler, scaler, epoch, val_metrics)
                print(f"  ✓ New best AUC={best_val_auc:.4f} — saved to {best_path}")

            # Early stopping
            if early_stop.step(val_metrics["auc"]):
                print(f"\n  Early stopping triggered after {epoch + 1} epochs.")
                break

            # History
            row = {"epoch": epoch + 1, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
            history.append(row)

        # Always save latest checkpoint
        latest_path = paths.checkpoint_dir / "latest.pth"
        save_checkpoint(latest_path, raw_model, optimizer, scheduler, scaler, epoch, {})

    # ── Wrap up ───────────────────────────────────────────────
    if writer:
        writer.close()
    if wandb_run:
        wandb_run.finish()

    history_path = paths.log_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Best val AUC: {best_val_auc:.4f}")
    print(f"History saved to {history_path}")
    print(f"Best model saved to {paths.checkpoint_dir / 'best_model.pth'}")


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from dataset import build_dataloaders
    import random

    parser = argparse.ArgumentParser(description="Train Deepfake Detector")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--data", type=str, default=None, help="Override data root path")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone", type=str, default=None, help="e.g. efficientnet_b4")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--fast-run", action="store_true", help="Quick debug run on small dataset")
    args = parser.parse_args()

    # Override config from CLI
    if args.data:
        cfg.paths.data_root = Path(args.data)
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.lr:
        cfg.train.lr = args.lr
    if args.backbone:
        cfg.model.backbone = args.backbone
    if args.no_amp:
        cfg.train.amp = False

    resume_path = Path(args.resume) if args.resume else None

    # ── Fast-run overrides ────────────────────────────────────
    if args.fast_run:
        print("[train] FAST-RUN mode enabled: using tiny dataset and fewer steps")
        cfg.train.epochs = 2
        cfg.train.batch_size = 8
        cfg.data.num_frames = 4
        cfg.model.backbone = "efficientnet_b0"
        cfg.data.use_face_detection = False      # ← FIXED: skip MTCNN so data stays on GPU path

        # Build dataloaders with face detection already disabled
        train_loader, val_loader, test_loader = build_dataloaders(cfg.data)

        # Subsample train to 200 samples
        full_dataset = train_loader.dataset
        subset_size = min(200, len(full_dataset))
        print(f"[train] Subsetting train dataset: {subset_size} / {len(full_dataset)} samples")

        subset_indices = random.sample(range(len(full_dataset)), subset_size)
        subset_dataset = torch.utils.data.Subset(full_dataset, subset_indices)

        train_loader = DataLoader(
            subset_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )

        # FIXED: pass loaders directly — no monkey-patching build_dataloaders
        train(resume_from=resume_path, loaders=(train_loader, val_loader, test_loader))

    else:
        # ── Normal training ───────────────────────────────────
        train(resume_from=resume_path)
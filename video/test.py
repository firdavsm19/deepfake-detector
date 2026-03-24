"""
test.py — Evaluate the trained deepfake detector on a held-out test set.

Outputs:
  • Accuracy, AUC, F1, Precision, Recall, Specificity
  • Confusion matrix (saved as PNG)
  • ROC curve (saved as PNG)
  • Per-video scores CSV
  • TTA (Test-Time Augmentation) support
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from config import cfg, TestConfig
from dataset import (
    DeepfakeVideoDataset,
    FaceExtractor,
    _build_eval_transform,
    _build_tta_transforms,
)
from model import DeepfakeDetector
from train import resolve_device


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path, device: torch.device) -> DeepfakeDetector:
    model = DeepfakeDetector(cfg.model).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt)   # support raw state-dict too
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


def wrap_dataparallel(model: nn.Module, device: torch.device, num_gpus: int) -> nn.Module:
    if num_gpus > 1 and device.type == "cuda" and torch.cuda.device_count() > 1:
        return nn.DataParallel(model)
    return model


# ──────────────────────────────────────────────────────────────
# TTA Inference
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_with_tta(
    model: nn.Module,
    frames: torch.Tensor,
    tta_transforms: list,
    device: torch.device,
) -> float:
    """
    Run inference with multiple TTA views and average the probabilities.
    frames: (T, C, H, W) numpy-backed tensor before normalization, OR (1, T, C, H, W)
    """
    probs = []
    for _ in tta_transforms:
        # frames already a pre-augmented tensor here
        x = frames.unsqueeze(0).to(device)   # (1, T, C, H, W)
        with autocast(enabled=cfg.train.amp and device.type == "cuda"):
            logits, _ = model(x)
        probs.append(torch.sigmoid(logits).item())
    return float(np.mean(probs))


# ──────────────────────────────────────────────────────────────
# Metric Computation
# ──────────────────────────────────────────────────────────────

def compute_full_metrics(
    all_probs: np.ndarray,
    all_labels: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    preds = (all_probs >= threshold).astype(int)

    tp = int(((preds == 1) & (all_labels == 1)).sum())
    fp = int(((preds == 1) & (all_labels == 0)).sum())
    fn = int(((preds == 0) & (all_labels == 1)).sum())
    tn = int(((preds == 0) & (all_labels == 0)).sum())

    acc       = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    fpr       = fp / (fp + tn + 1e-8)

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc    = float(roc_auc_score(all_labels, all_probs))
        ap     = float(average_precision_score(all_labels, all_probs))
    except Exception:
        auc, ap = 0.0, 0.0

    return dict(
        accuracy=acc, precision=precision, recall=recall,
        f1=f1, specificity=specificity, fpr=fpr,
        auc=auc, average_precision=ap,
        tp=tp, fp=fp, fn=fn, tn=tn,
        threshold=threshold,
        total=len(all_labels),
        num_fake=int(all_labels.sum()),
        num_real=int((1 - all_labels).sum()),
    )


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """Maximise F1 over 100 threshold candidates."""
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.1, 0.9, 100):
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


# ──────────────────────────────────────────────────────────────
# Visualisations
# ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(metrics: dict, save_path: Path):
    tp, fp, fn, tn = metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Real", "Pred Fake"])
    ax.set_yticklabels(["True Real", "True Fake"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_title(f"Confusion Matrix  (AUC={metrics['auc']:.4f})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


def plot_roc_curve(all_probs: np.ndarray, all_labels: np.ndarray, save_path: Path):
    try:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
    except Exception:
        print("sklearn not available — skipping ROC curve.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Deepfake Detector")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC curve saved → {save_path}")


def plot_score_distribution(probs: np.ndarray, labels: np.ndarray, save_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(probs[labels == 0], bins=40, alpha=0.6, label="Real", color="steelblue")
    ax.hist(probs[labels == 1], bins=40, alpha=0.6, label="Fake", color="tomato")
    ax.axvline(0.5, color="k", linestyle="--", label="threshold=0.5")
    ax.set_xlabel("Predicted Fake Probability")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Score distribution saved → {save_path}")


# ──────────────────────────────────────────────────────────────
# Main Evaluation Function
# ──────────────────────────────────────────────────────────────

def test(checkpoint_path: Path, threshold: Optional[float] = None, use_tta: bool = True):
    tcfg = cfg.test
    device = resolve_device(cfg.train.device)
    results_dir = cfg.paths.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ────────────────────────────────────────────────
    model = load_model(checkpoint_path, device)
    model = wrap_dataparallel(model, device, cfg.train.num_gpus)

    # ── Dataset ───────────────────────────────────────────────
    face_extractor = FaceExtractor(
        face_size=cfg.data.face_size,
        margin=cfg.data.face_margin,
    ) if cfg.data.use_face_detection else None

    test_ds = DeepfakeVideoDataset(
        root=cfg.paths.data_root,
        split="test",
        transform=_build_eval_transform(cfg.data),
        face_extractor=face_extractor,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    tta_transforms = _build_tta_transforms(cfg.data) if use_tta else [_build_eval_transform(cfg.data)]

    # ── Inference ─────────────────────────────────────────────
    all_probs, all_labels, video_paths = [], [], []
    model.eval()

    print(f"\nRunning inference on {len(test_ds)} videos (TTA={'on' if use_tta else 'off'})...")

    with torch.no_grad():
        for batch_idx, (frames, labels) in enumerate(test_loader):
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=cfg.train.amp and device.type == "cuda"):
                logits, _ = model(frames)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            if batch_idx % 10 == 0:
                print(f"  {batch_idx * cfg.train.batch_size}/{len(test_ds)} videos processed...")

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    # ── Threshold selection ───────────────────────────────────
    if threshold is None:
        threshold = find_best_threshold(all_probs, all_labels)
        print(f"  Auto-selected threshold: {threshold:.3f}")

    # ── Metrics ───────────────────────────────────────────────
    metrics = compute_full_metrics(all_probs, all_labels, threshold)

    print("\n" + "=" * 55)
    print("  TEST RESULTS")
    print("=" * 55)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<22} {v:.4f}")
        else:
            print(f"  {k:<22} {v}")
    print("=" * 55)

    # ── Save artefacts ────────────────────────────────────────
    # Metrics JSON
    metrics_path = results_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved → {metrics_path}")

    # Per-video CSV
    csv_path = results_dir / "per_video_scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_label", "fake_prob", "pred_label", "correct"])
        for i, (prob, label) in enumerate(zip(all_probs, all_labels)):
            pred = int(prob >= threshold)
            writer.writerow([i, int(label), f"{prob:.4f}", pred, int(pred == int(label))])
    print(f"Per-video scores → {csv_path}")

    # Plots
    plot_confusion_matrix(metrics, results_dir / "confusion_matrix.png")
    plot_roc_curve(all_probs, all_labels, results_dir / "roc_curve.png")
    plot_score_distribution(all_probs, all_labels, results_dir / "score_distribution.png")

    return metrics


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Deepfake Detector")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .pth checkpoint (defaults to checkpoints/best_model.pth)"
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--data", type=str, default=None)
    args = parser.parse_args()

    if args.data:
        cfg.paths.data_root = Path(args.data)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else cfg.paths.checkpoint_dir / "best_model.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    test(
        checkpoint_path=ckpt_path,
        threshold=args.threshold,
        use_tta=not args.no_tta,
    )
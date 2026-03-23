# ============================================================
#  evaluate.py — full evaluation on dev or eval split
#  Reports: EER, AUC, min-tDCF, accuracy, confusion matrix
# ============================================================
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, confusion_matrix,
    classification_report,
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import config
from dataset import ASVspoofDataset
from model import VoiceDetector


# ─────────────────────────────────────────────────────────────
#  Metric helpers
# ─────────────────────────────────────────────────────────────
def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """
    Returns (EER, threshold_at_EER).
    EER is the point where FAR == FRR.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = float(interp1d(fpr, thresholds)(eer))
    return float(eer), thresh


def compute_min_tdcf(
    labels: np.ndarray, scores: np.ndarray,
    p_target: float = 0.05,
    c_miss:   float = 1.0,
    c_fa:     float = 10.0,
) -> float:
    """
    Simplified min tandem-DCF (as used in ASVspoof evaluations).
    Default ASV is assumed perfect (t-DCF collapses to CM-DCF).
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    tdcf = c_miss * p_target * fnr + c_fa * (1 - p_target) * fpr
    return float(np.min(tdcf))


# ─────────────────────────────────────────────────────────────
#  Collect predictions
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_predictions(
    model:  VoiceDetector,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (labels, prob_fake, utt_ids).
    """
    model.eval()
    all_labels, all_probs, all_ids = [], [], []

    for wav, labels, utt_ids in loader:
        wav = wav.to(device)
        logits = model(wav)
        probs  = torch.softmax(logits, dim=-1)[:, 1]   # P(fake)

        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        all_ids.extend(utt_ids)

    return (
        np.array(all_labels, dtype=int),
        np.array(all_probs,  dtype=float),
        np.array(all_ids),
    )


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────
def main(split: str, checkpoint: str, threshold: float):
    device = config.DEVICE
    print(f"🚀 Device   : {device}")
    print(f"📂 Split    : {split}")
    print(f"🔍 Checkpoint: {checkpoint}\n")

    # ── Load model ────────────────────────────────────────────
    model = VoiceDetector().to(device)
    state = torch.load(checkpoint, map_location=device)
    # Support both raw state_dict and checkpoint dicts
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    print("✅ Model loaded.\n")

    # ── Dataset ───────────────────────────────────────────────
    ds     = ASVspoofDataset(split)
    loader = DataLoader(
        ds,
        batch_size  = config.BATCH_SIZE,
        shuffle     = False,
        num_workers = config.NUM_WORKERS,
        pin_memory  = True,
    )

    # ── Predict ───────────────────────────────────────────────
    print("⚙️  Running inference...")
    labels, probs, utt_ids = collect_predictions(model, loader, device)
    preds = (probs >= threshold).astype(int)

    # ── Metrics ───────────────────────────────────────────────
    eer,     eer_thresh = compute_eer(labels, probs)
    min_tdcf            = compute_min_tdcf(labels, probs)
    auc                 = roc_auc_score(labels, probs)
    acc                 = accuracy_score(labels, preds)
    cm                  = confusion_matrix(labels, preds)

    print("\n" + "═" * 48)
    print(f"  Split           : {split}")
    print(f"  Samples         : {len(labels)}")
    print(f"  Threshold used  : {threshold:.2f}")
    print("─" * 48)
    print(f"  EER             : {eer * 100:.2f} %   (threshold: {eer_thresh:.4f})")
    print(f"  min-tDCF        : {min_tdcf:.4f}")
    print(f"  AUC             : {auc:.4f}")
    print(f"  Accuracy        : {acc * 100:.2f} %")
    print("─" * 48)
    print("  Confusion matrix  (rows=actual, cols=pred)")
    print(f"              Real   Fake")
    print(f"  Actual Real  {cm[0, 0]:5d}  {cm[0, 1]:5d}")
    print(f"  Actual Fake  {cm[1, 0]:5d}  {cm[1, 1]:5d}")
    print("─" * 48)
    print(classification_report(labels, preds, target_names=["Real", "Fake"]))
    print("═" * 48)

    # ── Save score file (useful for official scoring tools) ───
    score_file = config.OUTPUT_DIR / f"scores_{split}.txt"
    with open(score_file, "w") as f:
        f.write("utt_id  label  prob_fake  decision\n")
        for uid, lbl, prob, pred in zip(utt_ids, labels, probs, preds):
            decision = "spoof" if pred == 1 else "bonafide"
            f.write(f"{uid}  {lbl}  {prob:.6f}  {decision}\n")
    print(f"\n💾 Scores saved to: {score_file}")


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate deepfake detector")
    parser.add_argument(
        "--split",
        choices=["dev", "eval"],
        default="dev",
        help="Which protocol split to evaluate (default: dev)",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(config.BEST_MODEL_PATH),
        help="Path to .pth model file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.THRESHOLD,
        help=f"P(fake) decision threshold (default: {config.THRESHOLD})",
    )
    args = parser.parse_args()
    main(args.split, args.checkpoint, args.threshold)
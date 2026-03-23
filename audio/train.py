%%writefile /content/train.py
# ============================================================
#  train.py — full training + validation loop
# ============================================================
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import config
from dataset import ASVspoofDataset
from model import VoiceDetector


# ─────────────────────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────────────────────
def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────
#  Equal Error Rate
# ─────────────────────────────────────────────────────────────
def compute_eer(labels: list, scores: list) -> float:
    """Lower is better. Returns EER in [0, 1]."""
    fpr, tpr, _ = [], [], []
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(eer)


# ─────────────────────────────────────────────────────────────
#  One training epoch
# ─────────────────────────────────────────────────────────────
def train_epoch(
    model:     VoiceDetector,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:

    model.train()
    total_loss = correct = total = 0

    for step, (wav, labels, _) in enumerate(loader, 1):
        wav, labels = wav.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(wav)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        optimizer.step()

        total_loss += loss.item() * wav.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

        if step % 50 == 0 or step == len(loader):
            print(f"  step [{step:4d}/{len(loader)}]  loss: {loss.item():.4f}")

    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────
#  Validation / evaluation
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model:  VoiceDetector,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Returns (loss, AUC, EER)."""

    model.eval()
    criterion  = nn.CrossEntropyLoss()
    all_probs, all_labels = [], []
    total_loss = total = 0

    for wav, labels, _ in loader:
        wav, labels = wav.to(device), labels.to(device)
        logits = model(wav)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * wav.size(0)
        total      += labels.size(0)

        probs = torch.softmax(logits, dim=-1)[:, 1]  # P(fake)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    auc      = roc_auc_score(all_labels, all_probs)
    eer      = compute_eer(all_labels, all_probs)
    return avg_loss, auc, eer


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────
def main():
    set_seed()
    device = config.DEVICE
    print(f"🚀 Device: {device}")

    # ── Data ──────────────────────────────────────────────────
    print("\n📂 Loading datasets...")
    train_ds  = ASVspoofDataset("train")
    dev_ds    = ASVspoofDataset("dev")

    train_loader = DataLoader(
        train_ds,
        batch_size  = config.BATCH_SIZE,
        shuffle     = True,
        drop_last   = True,
        num_workers = config.NUM_WORKERS,
        pin_memory  = True,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size  = config.BATCH_SIZE,
        shuffle     = False,
        num_workers = config.NUM_WORKERS,
        pin_memory  = True,
    )

    # ── Model ─────────────────────────────────────────────────
    print("\n🧠 Building model...")
    model = VoiceDetector().to(device)
    info  = model.count_params()
    print(f"   Total params    : {info['total']:,}")
    print(f"   Trainable params: {info['trainable']:,}")

    # ── Optimizer + scheduler ─────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.trainable_params(),
        lr           = config.LR_BACKEND,
        weight_decay = config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.SCHEDULER_T_MAX
    )

    # Weighted loss to handle class imbalance
    # ASVspoof 2019 train: ~2580 real, ~22800 spoof → weight real higher
    real_weight = torch.tensor([8.0, 1.0]).to(device)
    criterion   = nn.CrossEntropyLoss(weight=real_weight)

    # ── Training loop ─────────────────────────────────────────
    best_eer   = float("inf")
    history    = []

    print(f"\n🔥 Training for {config.EPOCHS} epochs...\n")
    print(f"{'Epoch':>5}  {'TrainLoss':>10}  {'TrainAcc':>9}  "
          f"{'ValLoss':>8}  {'ValAUC':>7}  {'ValEER':>7}")
    print("─" * 62)

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n── Epoch {epoch}/{config.EPOCHS} ──────────────────────────")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_auc, val_eer = evaluate(model, dev_loader, device)
        scheduler.step()

        # Log
        row = dict(
            epoch      = epoch,
            train_loss = train_loss,
            train_acc  = train_acc,
            val_loss   = val_loss,
            val_auc    = val_auc,
            val_eer    = val_eer,
        )
        history.append(row)

        print(f"\n  {'':>5}  {train_loss:10.4f}  {train_acc:9.4f}  "
              f"{val_loss:8.4f}  {val_auc:7.4f}  {val_eer:7.4f}")

        # Save every epoch
        ckpt = config.OUTPUT_DIR / f"model_epoch_{epoch:02d}.pth"
        torch.save({
            "epoch":       epoch,
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "val_eer":     val_eer,
            "val_auc":     val_auc,
        }, ckpt)

        # Save best
        if val_eer < best_eer:
            best_eer = val_eer
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"  🏆 New best EER = {best_eer:.4f}  → saved best_model.pth")

    # ── Save last model ───────────────────────────────────────
    torch.save(model.state_dict(), config.LAST_MODEL_PATH)

    # ── Print history table ───────────────────────────────────
    print("\n\n📊 Training history")
    print(f"{'Epoch':>5}  {'TrLoss':>8}  {'TrAcc':>7}  "
          f"{'VlLoss':>8}  {'VlAUC':>7}  {'VlEER':>7}")
    print("─" * 52)
    for r in history:
        print(f"{r['epoch']:5d}  {r['train_loss']:8.4f}  {r['train_acc']:7.4f}  "
              f"{r['val_loss']:8.4f}  {r['val_auc']:7.4f}  {r['val_eer']:7.4f}")

    print(f"\n✅ Training complete.  Best EER = {best_eer:.4f}")


if __name__ == "__main__":
    main()
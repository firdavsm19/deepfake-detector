import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from dataset import get_dataloaders
from model import DeepfakeDetector


def train():
    torch.manual_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Device: {device}")

    train_loader, val_loader, _ = get_dataloaders()

    model = DeepfakeDetector().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5
    )

    history      = {"train_loss": [], "train_acc": [], "val_acc": [], "val_auc": []}
    best_val_acc = 0

    for epoch in range(Config.EPOCHS):

        # ── Train ──────────────────────────────────────────
        model.train()
        train_loss, preds_all, labels_all = 0, [], []

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{Config.EPOCHS} [Train]")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()

            train_loss += loss.item()
            preds_all.extend(outputs.argmax(1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
            loop.set_postfix(loss=f"{loss.item():.4f}")

        # ── Validate ───────────────────────────────────────
        model.eval()
        val_preds, val_labels, val_probs = [], [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{Config.EPOCHS} [Val] "):
                images = images.to(device)
                out    = model(images)
                probs  = torch.softmax(out, dim=1)[:, 1]
                val_preds.extend(out.argmax(1).cpu().numpy())
                val_labels.extend(labels.numpy())
                val_probs.extend(probs.cpu().numpy())

        # ── Metrics ────────────────────────────────────────
        train_acc = accuracy_score(labels_all, preds_all)
        val_acc   = accuracy_score(val_labels, val_preds)
        val_auc   = roc_auc_score(val_labels, val_probs)
        avg_loss  = train_loss / len(train_loader)

        # ── Save history ───────────────────────────────────
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        print(f"\n  Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} "
              f"| Val Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")

        scheduler.step(val_acc)

        # ── Save best model ────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  ✅ Saved best model  (Val Acc: {val_acc:.4f})")

    plot_history(history)
    print(f"\n🏁 Training complete. Best Val Acc: {best_val_acc:.4f}")


def plot_history(history):
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], 'b-o')
    axes[0].set_title("Training Loss"); axes[0].grid(True)

    axes[1].plot(epochs, history["train_acc"], 'b-o', label="Train")
    axes[1].plot(epochs, history["val_acc"],   'r-o', label="Val")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True)

    axes[2].plot(epochs, history["val_auc"], 'g-o')
    axes[2].set_title("Val AUC-ROC"); axes[2].grid(True)

    plt.tight_layout()
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    plt.savefig("logs/training_history.png", dpi=150)
    print("📊 Plot saved → logs/training_history.png")


if __name__ == "__main__":
    train()
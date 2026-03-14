import torch
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              classification_report, confusion_matrix)
from tqdm import tqdm

from config import Config
from dataset import get_dataloaders
from model import DeepfakeDetector


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders()

    model = DeepfakeDetector()
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval().to(device)

    all_preds, all_labels, all_probs = [], [], []

    print("\n🧪 Running on Test Set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images  = images.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)[:, 1]
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm  = confusion_matrix(all_labels, all_preds)

    print(f"\n{'═'*45}")
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"  Test AUC-ROC  : {auc:.4f}")
    print(f"{'═'*45}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["REAL", "FAKE"]))
    print("Confusion Matrix:")
    print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")


if __name__ == "__main__":
    test()
%%writefile /content/predict.py
# ============================================================
#  predict.py — inference on a single file or a folder
#  Usage:
#    python predict.py --input audio.wav
#    python predict.py --input ./folder/ --threshold 0.6
#    python predict.py --input audio.wav --checkpoint my_model.pth
# ============================================================
import argparse
import os
import numpy as np
import torch
import librosa

import config
from model import VoiceDetector


# ─────────────────────────────────────────────────────────────
#  Load model
# ─────────────────────────────────────────────────────────────
def load_model(checkpoint: str, device: torch.device) -> VoiceDetector:
    model = VoiceDetector().to(device)
    state = torch.load(checkpoint, map_location=device)
    if "model" in state:        # full checkpoint dict
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    print(f"✅ Loaded model from: {checkpoint}")
    return model


# ─────────────────────────────────────────────────────────────
#  Audio preprocessing
# ─────────────────────────────────────────────────────────────
def load_audio(path: str) -> torch.Tensor:
    """Load → resample → mono → pad/truncate → tensor."""
    wav, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)

    # Pad or truncate
    if len(wav) < config.MAX_LEN:
        wav = np.pad(wav, (0, config.MAX_LEN - len(wav)))
    else:
        wav = wav[: config.MAX_LEN]

    return torch.tensor(wav, dtype=torch.float32).unsqueeze(0)  # (1, T)


# ─────────────────────────────────────────────────────────────
#  Single-file prediction
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_file(
    path:      str,
    model:     VoiceDetector,
    device:    torch.device,
    threshold: float = 0.5,
) -> dict:
    """
    Returns:
        label      : "REAL" or "FAKE"
        prob_fake  : float [0, 1]
        prob_real  : float [0, 1]
        confidence : winning-class probability * 100
    """
    wav    = load_audio(path).to(device)
    logits = model(wav)
    probs  = torch.softmax(logits, dim=-1)[0]

    prob_real = probs[0].item()
    prob_fake = probs[1].item()
    label     = "FAKE" if prob_fake >= threshold else "REAL"
    confidence = (prob_fake if label == "FAKE" else prob_real) * 100

    return {
        "file":       os.path.basename(path),
        "label":      label,
        "prob_fake":  round(prob_fake,  4),
        "prob_real":  round(prob_real,  4),
        "confidence": round(confidence, 1),
    }


# ─────────────────────────────────────────────────────────────
#  Folder prediction
# ─────────────────────────────────────────────────────────────
def predict_folder(
    folder:    str,
    model:     VoiceDetector,
    device:    torch.device,
    threshold: float = 0.5,
) -> list[dict]:

    EXTENSIONS = (".flac", ".wav", ".mp3", ".ogg", ".m4a")
    files = sorted(
        f for f in os.listdir(folder) if f.lower().endswith(EXTENSIONS)
    )
    if not files:
        print(f"⚠️  No audio files found in {folder}")
        return []

    results = []
    for fname in files:
        path = os.path.join(folder, fname)
        try:
            r = predict_file(path, model, device, threshold)
        except Exception as e:
            print(f"⚠️  {fname}: {e}")
            continue

        results.append(r)
        icon = "🔴 FAKE" if r["label"] == "FAKE" else "🟢 REAL"
        print(
            f"{icon}  |  {r['file']:<45}  "
            f"fake={r['prob_fake']:.3f}  conf={r['confidence']:5.1f}%"
        )

    # Summary
    n_fake = sum(1 for r in results if r["label"] == "FAKE")
    n_real = len(results) - n_fake
    print(f"\n📊 Summary: {len(results)} files — 🟢 {n_real} real, 🔴 {n_fake} fake")
    return results


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Audio deepfake detector — inference")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a single audio file or a folder of audio files",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(config.BEST_MODEL_PATH),
        help=f"Path to model checkpoint (default: {config.BEST_MODEL_PATH})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.THRESHOLD,
        help=f"P(fake) decision threshold (default: {config.THRESHOLD})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌  Not found: {args.input}")
        return

    device = config.DEVICE
    model  = load_model(args.checkpoint, device)

    print()
    if os.path.isdir(args.input):
        predict_folder(args.input, model, device, args.threshold)
    else:
        r = predict_file(args.input, model, device, args.threshold)
        icon = "🔴 FAKE" if r["label"] == "FAKE" else "🟢 REAL"
        print(f"\n{icon}")
        print(f"  File       : {r['file']}")
        print(f"  P(fake)    : {r['prob_fake']:.4f}")
        print(f"  P(real)    : {r['prob_real']:.4f}")
        print(f"  Confidence : {r['confidence']}%")


if __name__ == "__main__":
    main()
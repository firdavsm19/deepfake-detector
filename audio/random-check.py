import torch
import numpy as np
import librosa
import sys
sys.path.insert(0, '/content')

import config
from model import VoiceDetector
from dataset import parse_protocol

# ── 1. Pick a random sample from the eval protocol ───────────
samples = parse_protocol(config.EVAL_PROTO)

import random
utt_id, true_label = random.choice(samples)
audio_path = config.EVAL_AUDIO / f"{utt_id}.flac"

print(f"File      : {utt_id}.flac")
print(f"True label: {'REAL' if true_label == 0 else 'FAKE'}  ({true_label})")

# ── 2. Load and preprocess audio ─────────────────────────────
wav, _ = librosa.load(str(audio_path), sr=16000, mono=True)
if len(wav) >= config.MAX_LEN:
    wav = wav[:config.MAX_LEN]
else:
    wav = np.pad(wav, (0, config.MAX_LEN - len(wav)))

tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)

# ── 3. Load best model and predict ───────────────────────────
model = VoiceDetector().to(config.DEVICE)
model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))
model.eval()

with torch.no_grad():
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=-1)[0]

prob_real = probs[0].item()
prob_fake = probs[1].item()
predicted = "FAKE" if prob_fake >= config.THRESHOLD else "REAL"
correct   = predicted == ("REAL" if true_label == 0 else "FAKE")

# ── 4. Print result ───────────────────────────────────────────
print(f"\n{'='*40}")
print(f"  Predicted : {predicted}")
print(f"  True label: {'REAL' if true_label == 0 else 'FAKE'}")
print(f"  P(real)   : {prob_real:.4f}")
print(f"  P(fake)   : {prob_fake:.4f}")
print(f"  Correct   : {'✅ YES' if correct else '❌ NO'}")
print(f"{'='*40}")

%%writefile /content/dataset.py
# ============================================================
#  dataset.py — ASVspoof 2019 LA dataset loader
# ============================================================
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
from pathlib import Path
from typing import Literal
import config


# ── Protocol parser ──────────────────────────────────────────
def parse_protocol(proto_path: Path) -> list[tuple[str, int]]:
    """
    Parse an ASVspoof 2019 LA protocol file.

    Protocol columns:
        0  speaker_id
        1  utterance_id     ← filename (without .flac)
        2  env_id
        3  attack_id
        4  key               bonafide | spoof

    Returns:
        list of (utterance_id, label)   label: 0 = real, 1 = fake
    """
    samples = []
    with open(proto_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            utt_id = parts[1]
            label  = 0 if parts[4] == "bonafide" else 1
            samples.append((utt_id, label))
    return samples


# ── Dataset ───────────────────────────────────────────────────
class ASVspoofDataset(Dataset):
    """
    Loads ASVspoof 2019 LA audio for train / dev / eval splits.

    Args:
        split : one of "train", "dev", "eval"
    """

    SPLITS = {
        "train": (config.TRAIN_AUDIO, config.TRAIN_PROTO),
        "dev":   (config.DEV_AUDIO,   config.DEV_PROTO),
        "eval":  (config.EVAL_AUDIO,  config.EVAL_PROTO),
    }

    def __init__(self, split: Literal["train", "dev", "eval"]):
        assert split in self.SPLITS, f"split must be one of {list(self.SPLITS)}"
        audio_dir, proto_path = self.SPLITS[split]

        self.audio_dir = Path(audio_dir)
        self.max_len   = config.MAX_LEN
        self.samples   = parse_protocol(proto_path)  # [(utt_id, label), ...]

        real  = sum(1 for _, l in self.samples if l == 0)
        fake  = sum(1 for _, l in self.samples if l == 1)
        print(f"[{split:5s}]  {len(self.samples):6d} files  |  real: {real}  fake: {fake}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        utt_id, label = self.samples[idx]
        path = self.audio_dir / f"{utt_id}.flac"

        try:
            wav, _ = librosa.load(str(path), sr=config.SAMPLE_RATE, mono=True)
            wav    = self._fix_length(wav)
            return torch.tensor(wav, dtype=torch.float32), label, utt_id
        except Exception as e:
            # Skip broken files gracefully
            print(f"⚠️  Could not load {path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

    # ── helpers ───────────────────────────────────────────────
    def _fix_length(self, wav: np.ndarray) -> np.ndarray:
        if len(wav) >= self.max_len:
            return wav[: self.max_len]
        return np.pad(wav, (0, self.max_len - len(wav)))


# ── Quick sanity-check ────────────────────────────────────────
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds     = ASVspoofDataset("train")
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    wav, labels, ids = next(iter(loader))
    print(f"wav shape : {wav.shape}")    # (4, 64000)
    print(f"labels    : {labels}")
    print(f"ids       : {ids}")
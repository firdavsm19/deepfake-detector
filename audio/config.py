# ============================================================
#  config.py — single place for all paths & hyperparameters
# ============================================================
import torch
from pathlib import Path

# ── Paths (Kaggle) ───────────────────────────────────────────
DATA_ROOT = Path("/kaggle/input/asvpoof-2019-dataset/LA/LA")

TRAIN_AUDIO = DATA_ROOT / "ASVspoof2019_LA_train/flac"
DEV_AUDIO   = DATA_ROOT / "ASVspoof2019_LA_dev/flac"
EVAL_AUDIO  = DATA_ROOT / "ASVspoof2019_LA_eval/flac"

TRAIN_PROTO = DATA_ROOT / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
DEV_PROTO   = DATA_ROOT / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
EVAL_PROTO  = DATA_ROOT / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"

# ── Output ────────────────────────────────────────────────────
OUTPUT_DIR       = Path("./outputs")
BEST_MODEL_PATH  = OUTPUT_DIR / "best_model.pth"
LAST_MODEL_PATH  = OUTPUT_DIR / "last_model.pth"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Audio ─────────────────────────────────────────────────────
SAMPLE_RATE      = 16000
MAX_DURATION_SEC = 4.0
MAX_LEN          = int(SAMPLE_RATE * MAX_DURATION_SEC)   # 64 000 samples

# ── Model ─────────────────────────────────────────────────────
BACKBONE         = "facebook/wav2vec2-base"   # 768-dim hidden
# BACKBONE       = "facebook/mms-300m"        # 1024-dim — swap if you want multilingual
BACKBONE_DIM     = 768                        # must match the backbone above
PROJ_DIM         = 128                        # projection → AASIST backend
FREEZE_BACKBONE  = True                       # freeze CNN + transformer during training

# ── Training ──────────────────────────────────────────────────
BATCH_SIZE       = 16
EPOCHS           = 20
LR_BACKEND       = 1e-4    # learning rate for projection + AASIST head
LR_BACKBONE      = 5e-6    # (only used when FREEZE_BACKBONE = False)
WEIGHT_DECAY     = 1e-4
MAX_GRAD_NORM    = 1.0
SCHEDULER_T_MAX  = 20      # cosine annealing period (= EPOCHS)

# ── Evaluation ────────────────────────────────────────────────
THRESHOLD        = 0.5     # default P(fake) threshold for binary decision

# ── Misc ──────────────────────────────────────────────────────
SEED             = 42
NUM_WORKERS      = 2
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
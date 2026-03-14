# ═══════════════════════════════════════════════
# ALL SETTINGS IN ONE PLACE — edit these values
# ═══════════════════════════════════════════════

class Config:
    # Paths
    DATA_ROOT        = "data"
    CHECKPOINT_DIR   = "checkpoints"
    LOG_DIR          = "logs"
    BEST_MODEL_PATH  = "checkpoints/best_model.pth"

    # Data
    IMAGE_SIZE       = 224          # reduced from 380 → faster + better on CPU
    BATCH_SIZE       = 32           # increased from 16 → more stable training
    NUM_WORKERS      = 4

    # Training
    EPOCHS           = 1           # increased from 15
    LEARNING_RATE    = 1e-3         # increased from 1e-4 → learn faster
    WEIGHT_DECAY     = 1e-4
    LABEL_SMOOTHING  = 0.0          # removed smoothing → clearer signal
    GRAD_CLIP        = 1.0

    # Model
    NUM_CLASSES      = 2
    DROPOUT_RATE     = 0.3          # reduced from 0.4 → less regularization
    DROP_CONNECT     = 0.1          # reduced from 0.2

    # Reproducibility
    SEED             = 42
"""Global configuration for the beta-VAE feature disentanglement project.

All tunable hyperparameters are centralized here and shared by the training and
evaluation scripts via `import config`, avoiding magic numbers scattered around.
After editing, simply re-run training/evaluation for the changes to take effect.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"          # MNIST is downloaded here automatically
OUTPUT_DIR = ROOT / "outputs"     # Weights, figures and metrics are written here
MODEL_PATH = OUTPUT_DIR / "beta_vae.pt"
HISTORY_PATH = OUTPUT_DIR / "history.json"

# ---------------------------------------------------------------------------
# Data (FactorMNIST: two controllable factors "rotation + stroke thickness"
# are applied on top of MNIST)
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
NUM_WORKERS = 0                   # 0 = load in the main process (avoids Windows multiprocessing issues)
ROTATION_RANGE = (-30.0, 30.0)    # Rotation range in degrees (sampled continuously)
THICKNESS_LEVELS = 3              # Thickness levels: 1=thin / 2=normal / 3=thick

# ---------------------------------------------------------------------------
# Model (beta-VAE: CNN encoder + symmetric transposed-conv decoder)
# ---------------------------------------------------------------------------
LATENT_DIM = 20                   # Latent dimension
HIDDEN_DIMS = [32, 64, 128]       # Encoder conv channels (decoder is symmetric)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
EPOCHS = 40
LR = 1e-3
BETA = 4.0                        # Beta coefficient: =1 reduces to standard VAE; >1 strengthens disentanglement
DEVICE = "auto"                   # auto / cuda / cpu
SEED = 42
LOG_INTERVAL = 100                # Print progress every N batches

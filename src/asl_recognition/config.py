"""Central place for project paths and settings.

Keeping these values here means every script and module uses the same numbers
(for example, the number of features), so a change only has to be made once.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# This file lives in <root>/src/asl_recognition/, so the project root is two
# folders up from the folder containing this file.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_DIR / "asl_recognition.log"

# ---------------------------------------------------------------------------
# Signs the project recognizes
# ---------------------------------------------------------------------------
# J and Z are left out on purpose: in ASL they are drawn in the air, so a
# single still frame cannot represent them. That leaves 24 static letters.
STATIC_SIGN_LABELS = tuple("ABCDEFGHIKLMNOPQRSTUVWXY")

# ---------------------------------------------------------------------------
# Hand landmarks (MediaPipe Hands returns 21 points per hand)
# ---------------------------------------------------------------------------
NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3  # x, y, z
NUM_FEATURES = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63
WRIST_INDEX = 0  # landmark 0 is the wrist; we use it as the origin

# ---------------------------------------------------------------------------
# Camera and detector settings
# ---------------------------------------------------------------------------
DEFAULT_CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Flip the image like a selfie view. Data collection and live prediction must
# use the same setting, because a mirrored hand gives different coordinates.
MIRROR_CAMERA = True

# One hand keeps the problem simple: one hand -> one letter.
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.5

# ---------------------------------------------------------------------------
# Dataset collection
# ---------------------------------------------------------------------------
RAW_DATASET_FILE = RAW_DATA_DIR / "landmarks.csv"

# A small starter set whose hand shapes look clearly different from each other.
# Check each shape against an ASL reference chart before recording it.
DEFAULT_LABELS = ("A", "B", "C", "L", "Y")

DEFAULT_SAMPLES_PER_SESSION = 100
MAX_SAMPLES_PER_SESSION = 1000  # upper limit for one recording; catches typos like 100000
CAPTURE_INTERVAL_SECONDS = 0.1  # about 10 samples per second while recording
COUNTDOWN_SECONDS = 3.0         # time to get your hand in position

# Frames from one recording look almost identical. If some of them ended up in
# the training set and some in the test set, the test score would look better
# than it really is. Every recording therefore gets a session_id, and the
# training step (next batch) will keep whole sessions together. That only
# works if each label has several sessions, so we require a minimum.
MIN_SAMPLES_PER_LABEL = 100
MIN_SESSIONS_PER_LABEL = 3

# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------
MODEL_FILE = MODELS_DIR / "asl_classifier.joblib"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"

# Fractions of each label's SESSIONS (not rows) used for validation and test.
TEST_FRACTION = 0.2
VALIDATION_FRACTION = 0.2

# With only the minimum of 3 sessions, the test set is a single recording per
# letter, which gives a noisy score. Training warns below this number.
RECOMMENDED_SESSIONS_PER_LABEL = 5

# File names written by scripts/evaluate_model.py into the reports folder.
METRICS_FILENAME = "test_metrics.json"
CONFUSION_MATRIX_IMAGE_NAME = "confusion_matrix.png"
CONFUSION_MATRIX_CSV_NAME = "confusion_matrix.csv"

# ---------------------------------------------------------------------------
# Settings used by the training and prediction steps in later milestones
# ---------------------------------------------------------------------------
RANDOM_SEED = 42            # fixed so train/test splits are repeatable
CONFIDENCE_THRESHOLD = 0.70  # below this, the app shows "uncertain"
SMOOTHING_WINDOW = 7         # number of recent frames used to smooth predictions
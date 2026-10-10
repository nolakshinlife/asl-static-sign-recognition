"""Saving, loading and checking the landmark dataset.

The dataset is one CSV file with one row per captured frame:

    session_id, label, x0, y0, z0, x1, y1, z1, ... x20, y20, z20

The 63 feature columns are the NORMALIZED features from
features.landmarks_to_features(), not raw MediaPipe coordinates. Training and
live prediction use the same function, so they always see the same kind of
numbers.

session_id identifies one recording (one press of SPACE in collect_data.py).
Training uses it to keep all frames of a recording on the same side of the
train/test split, which avoids data leakage.
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from asl_recognition import config
from asl_recognition.features import feature_column_names

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

METADATA_COLUMNS = ["session_id", "label"]
SUMMARY_COLUMNS = ["label", "samples", "sessions"]
FLOAT_FORMAT = "%.6f"  # 6 decimals is plenty and keeps the CSV small


class DatasetError(Exception):
    """Raised when the dataset file is missing, damaged or cannot be written."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def csv_columns() -> List[str]:
    """All CSV column names in the order they are stored."""
    return METADATA_COLUMNS + feature_column_names()


def normalize_label(
    label: str, allowed_labels: Iterable[str] = config.STATIC_SIGN_LABELS
) -> str:
    """Return the label as a clean upper-case letter, or raise ValueError.

    " a " becomes "A". Letters that are not static signs (such as J and Z)
    are rejected.
    """
    if not isinstance(label, str):
        raise ValueError(f"Label must be a string, got {type(label).__name__}.")
    cleaned = label.strip().upper()
    allowed = list(allowed_labels)
    if cleaned not in allowed:
        raise ValueError(
            f"Invalid label {label!r}. Allowed labels: {' '.join(allowed)}."
        )
    return cleaned


def normalize_label_list(labels: Sequence[str]) -> List[str]:
    """Validate a list of labels and remove duplicates, keeping the order."""
    cleaned: List[str] = []
    for label in labels:
        value = normalize_label(label)
        if value not in cleaned:
            cleaned.append(value)
    if not cleaned:
        raise ValueError("At least one label is required.")
    return cleaned


def validate_sample_count(
    count: int, maximum: int = config.MAX_SAMPLES_PER_SESSION
) -> int:
    """Check that a sample count is a whole number from 1 to maximum."""
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError(f"Sample count must be a whole number, got {count!r}.")
    if not 1 <= count <= maximum:
        raise ValueError(f"Sample count must be between 1 and {maximum}, got {count}.")
    return count


def new_session_id() -> str:
    """Create an id such as '20261010-193045-123' from the current time."""
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]


def _validate_features(features: np.ndarray) -> np.ndarray:
    """Return features as a float array of shape (63,), or raise ValueError."""
    try:
        array = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Features must be numeric: {error}") from error
    if array.shape != (config.NUM_FEATURES,):
        raise ValueError(
            f"Features must have shape ({config.NUM_FEATURES},), got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Features contain NaN or infinite values.")
    return array


# ---------------------------------------------------------------------------
# Recording one session
# ---------------------------------------------------------------------------
class SessionRecorder:
    """Collects samples for ONE label during ONE recording.

    It has no camera code, so it can be tested without a webcam. The caller
    passes the current time (for example time.monotonic()) to every method.

    The recorder waits for a short countdown, then accepts a sample at most
    once per interval until target_count samples have been collected. Spacing
    the samples out gives more variety than saving every single frame.
    """

    def __init__(
        self,
        label: str,
        target_count: int,
        start_time: float,
        countdown_seconds: float = config.COUNTDOWN_SECONDS,
        interval_seconds: float = config.CAPTURE_INTERVAL_SECONDS,
    ) -> None:
        if countdown_seconds < 0:
            raise ValueError("countdown_seconds cannot be negative.")
        if interval_seconds < 0:
            raise ValueError("interval_seconds cannot be negative.")

        self.label = normalize_label(label)
        self.target_count = validate_sample_count(target_count)
        self.session_id = new_session_id()
        self.samples: List[np.ndarray] = []

        self._start_time = start_time
        self._countdown_seconds = countdown_seconds
        self._interval_seconds = interval_seconds
        self._last_capture_time: Optional[float] = None

    def countdown_remaining(self, now: float) -> float:
        """Seconds left before recording starts (0 once it has started)."""
        return max(0.0, self._start_time + self._countdown_seconds - now)

    @property
    def is_complete(self) -> bool:
        """True once target_count samples have been collected."""
        return len(self.samples) >= self.target_count

    def try_add(self, features: np.ndarray, now: float) -> bool:
        """Store the features if it is time for the next sample.

        Returns:
            True if the sample was stored, False if it was skipped (countdown
            still running, interval not over yet, or recording is complete).
        """
        if self.is_complete or self.countdown_remaining(now) > 0:
            return False
        if (
            self._last_capture_time is not None
            and now - self._last_capture_time < self._interval_seconds
        ):
            return False

        self.samples.append(_validate_features(features))
        self._last_capture_time = now
        return True


# ---------------------------------------------------------------------------
# Reading and writing the CSV file
# ---------------------------------------------------------------------------
def dataset_exists(csv_path: Optional[PathLike] = None) -> bool:
    """True if the dataset file exists and is not empty."""
    path = Path(csv_path) if csv_path is not None else config.RAW_DATASET_FILE
    return path.is_file() and path.stat().st_size > 0


def _read_header(path: Path) -> List[str]:
    """Return the first row of a CSV file (empty list for an empty file)."""
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return next(csv.reader(file), [])


def append_samples(
    csv_path: PathLike,
    session_id: str,
    label: str,
    features_list: Sequence[np.ndarray],
) -> int:
    """Append one session's samples to the CSV file.

    Every sample is checked before anything is written, so a bad sample can
    never leave a half-written session in the file. The header row is written
    only when the file is new or empty.

    Returns:
        The number of rows written.

    Raises:
        ValueError: Bad label, session id or features.
        DatasetError: The existing file has different columns, or writing failed.
    """
    label = normalize_label(label)
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string.")
    rows = [_validate_features(features) for features in features_list]
    if not rows:
        return 0

    path = Path(csv_path)
    columns = csv_columns()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_has_content = path.is_file() and path.stat().st_size > 0
        if file_has_content and _read_header(path) != columns:
            raise DatasetError(
                f"{path} has different columns than expected. Move or delete it "
                "and collect again."
            )

        # newline="" stops Python from adding extra blank lines on Windows.
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_has_content:
                writer.writerow(columns)
            for features in rows:
                writer.writerow(
                    [session_id, label] + [FLOAT_FORMAT % value for value in features]
                )
    except OSError as error:
        raise DatasetError(f"Could not write to {path}: {error}") from error

    logger.info("Saved %d samples for label %s (session %s).", len(rows), label, session_id)
    return len(rows)


def load_dataset(
    csv_path: Optional[PathLike] = None, allow_empty: bool = False
) -> pd.DataFrame:
    """Load the dataset and check that it is valid.

    Checks: file exists, exact columns, no empty ids or labels, only known
    labels, and all features are finite numbers.

    Args:
        csv_path: Dataset file. Defaults to config.RAW_DATASET_FILE.
        allow_empty: If True, a file with only a header row is accepted.

    Raises:
        DatasetError: If the file is missing or invalid.
    """
    path = Path(csv_path) if csv_path is not None else config.RAW_DATASET_FILE
    if not path.is_file():
        raise DatasetError(
            f"Dataset file not found: {path}. Run scripts/collect_data.py first."
        )

    try:
        data = pd.read_csv(
            path, dtype={"session_id": str, "label": str}, encoding="utf-8-sig"
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, OSError) as error:
        raise DatasetError(f"Could not read {path}: {error}") from error

    if list(data.columns) != csv_columns():
        raise DatasetError(
            f"{path} does not have the expected columns "
            f"(session_id, label, x0 ... z20)."
        )
    if data.empty:
        if allow_empty:
            return data
        raise DatasetError(f"{path} contains no samples.")

    if data["session_id"].isna().any() or data["label"].isna().any():
        raise DatasetError(f"{path} has rows with a missing session_id or label.")

    unknown = sorted(set(data["label"]) - set(config.STATIC_SIGN_LABELS))
    if unknown:
        raise DatasetError(f"{path} contains unknown labels: {' '.join(unknown)}.")

    try:
        values = data[feature_column_names()].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise DatasetError(f"{path} has non-numeric feature values: {error}") from error
    if not np.all(np.isfinite(values)):
        raise DatasetError(f"{path} has missing or infinite feature values.")

    return data


def remove_session(csv_path: PathLike, session_id: str) -> int:
    """Delete every row of one session. Used by the 'undo' key.

    The file is rewritten to a temporary file first and then swapped in, so a
    crash in the middle cannot destroy the dataset.

    Returns:
        The number of rows removed (0 if the session was not found).
    """
    path = Path(csv_path)
    data = load_dataset(path, allow_empty=True)
    keep = data["session_id"] != session_id
    removed = int((~keep).sum())
    if removed == 0:
        return 0

    temp_path = path.with_name(path.name + ".tmp")
    try:
        data[keep].to_csv(
            temp_path, index=False, float_format=FLOAT_FORMAT, encoding="utf-8"
        )
        temp_path.replace(path)
    except OSError as error:
        raise DatasetError(f"Could not update {path}: {error}") from error

    logger.info("Removed %d rows of session %s.", removed, session_id)
    return removed


# ---------------------------------------------------------------------------
# Summaries and checks
# ---------------------------------------------------------------------------
def summarize_dataset(
    data: pd.DataFrame, labels: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Count samples and sessions for each label.

    Args:
        data: A dataset returned by load_dataset().
        labels: Labels to include (labels with no data show 0). Defaults to
            the labels found in the data.

    Returns:
        A table with the columns label, samples and sessions.
    """
    if labels is None:
        labels = sorted(data["label"].unique())

    rows = []
    for label in labels:
        subset = data[data["label"] == label]
        rows.append(
            {
                "label": label,
                "samples": int(len(subset)),
                "sessions": int(subset["session_id"].nunique()),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def load_summary(
    csv_path: Optional[PathLike] = None, labels: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Like summarize_dataset(), but reads the file and accepts a missing one."""
    if dataset_exists(csv_path):
        data = load_dataset(csv_path, allow_empty=True)
    else:
        data = pd.DataFrame(columns=csv_columns())
    return summarize_dataset(data, labels)


def find_dataset_problems(
    summary: pd.DataFrame,
    min_samples: int = config.MIN_SAMPLES_PER_LABEL,
    min_sessions: int = config.MIN_SESSIONS_PER_LABEL,
) -> List[str]:
    """Return a list of reasons the dataset is not ready for training.

    An empty list means the minimum counts are met. It does not say anything
    about how good the final model will be.
    """
    problems: List[str] = []
    if len(summary) < 2:
        problems.append("At least 2 different labels are needed to train a classifier.")

    for row in summary.itertuples(index=False):
        if row.samples < min_samples:
            problems.append(
                f"Label {row.label}: {row.samples} samples (need at least {min_samples})."
            )
        if row.sessions < min_sessions:
            problems.append(
                f"Label {row.label}: {row.sessions} sessions (need at least {min_sessions})."
            )
    return problems
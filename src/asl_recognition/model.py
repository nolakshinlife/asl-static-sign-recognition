"""Training, evaluation and saving of the sign classifier.

How the pieces fit together:
  1. validate_training_data()  checks the dataset is big enough and normalized.
  2. split_sessions()          splits whole recording SESSIONS into train,
                               validation and test sets (no data leakage).
  3. select_best_model()       trains a few candidate models on the training set
                               and picks the best one on the VALIDATION set.
  4. train_final_model()       refits the chosen model on train + validation.
  5. save_model()              stores the model plus a JSON file with the label
                               list and the exact session ids of each split.
  6. evaluate_model()          scores a model on the TEST sessions.

The dataset already holds normalized features (see features.py), so nothing is
normalized again here. Live prediction must use the same
features.landmarks_to_features() function.
"""

import hashlib
import json
import logging
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from asl_recognition import config
from asl_recognition.dataset import find_dataset_problems, summarize_dataset
from asl_recognition.features import feature_column_names

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

METADATA_SCHEMA_VERSION = 1
MIN_SESSIONS_TO_SPLIT = 3     # one session each for train, validation and test
MAX_SPLIT_FRACTION = 0.4      # keeps enough sessions left over for training


class ModelError(Exception):
    """Raised when training data, a saved model or its metadata is not usable."""


@dataclass(frozen=True)
class SessionSplit:
    """The session ids that belong to each part of the split."""

    train: List[str]
    validation: List[str]
    test: List[str]


# ---------------------------------------------------------------------------
# Checking the data and splitting it
# ---------------------------------------------------------------------------
def validate_training_data(data: pd.DataFrame) -> None:
    """Raise ModelError if the dataset is not ready for training.

    The size rules come from dataset.find_dataset_problems(). We also check
    that the features look normalized: after normalization the wrist is the
    origin, so the wrist columns must be 0. This catches data that was saved
    with a different preprocessing step.
    """
    if data.empty:
        raise ModelError("The dataset is empty. Collect data with scripts/collect_data.py.")

    summary = summarize_dataset(data)
    problems = find_dataset_problems(summary)

    wrist_columns = [f"{axis}{config.WRIST_INDEX}" for axis in ("x", "y", "z")]
    wrist = data[wrist_columns].to_numpy(dtype=np.float64)
    if not np.allclose(wrist, 0.0, atol=1e-5):
        problems.append(
            f"Features do not look normalized (columns {', '.join(wrist_columns)} "
            "should all be 0). The data may come from a different preprocessing step."
        )

    if problems:
        details = "\n".join(f"  - {problem}" for problem in problems)
        raise ModelError(f"The dataset is not ready for training:\n{details}")

    for row in summary.itertuples(index=False):
        if row.sessions < config.RECOMMENDED_SESSIONS_PER_LABEL:
            logger.warning(
                "Label %s has only %d sessions. %d or more give a more reliable test result.",
                row.label,
                row.sessions,
                config.RECOMMENDED_SESSIONS_PER_LABEL,
            )


def check_no_session_overlap(split: SessionSplit) -> None:
    """Raise ModelError if any session is in more than one part of the split."""
    pairs = (
        ("train", "validation", split.train, split.validation),
        ("train", "test", split.train, split.test),
        ("validation", "test", split.validation, split.test),
    )
    for name_a, name_b, sessions_a, sessions_b in pairs:
        shared = set(sessions_a) & set(sessions_b)
        if shared:
            raise ModelError(
                f"Data leakage: {len(shared)} session(s) are in both the "
                f"{name_a} and {name_b} sets."
            )


def split_sessions(
    data: pd.DataFrame,
    test_fraction: float = config.TEST_FRACTION,
    validation_fraction: float = config.VALIDATION_FRACTION,
    seed: int = config.RANDOM_SEED,
) -> SessionSplit:
    """Split whole sessions into train, validation and test sets.

    Each label is split on its own, so every letter appears in all three sets.
    Whole sessions stay together because frames from one recording are almost
    identical; splitting individual rows would leak near-copies of the test
    data into training and make the test score look better than it is.

    The result is repeatable: the same data and seed give the same split.

    Raises:
        ValueError: A fraction is outside 0 < fraction <= 0.4.
        ModelError: A label has fewer than 3 sessions.
    """
    for name, value in (
        ("test_fraction", test_fraction),
        ("validation_fraction", validation_fraction),
    ):
        if not 0 < value <= MAX_SPLIT_FRACTION:
            raise ValueError(
                f"{name} must be greater than 0 and at most {MAX_SPLIT_FRACTION}, got {value}."
            )

    rng = np.random.default_rng(seed)
    train: List[str] = []
    validation: List[str] = []
    test: List[str] = []

    # Sorting first makes the result independent of the row order in the CSV.
    for label in sorted(data["label"].unique()):
        sessions = sorted(data.loc[data["label"] == label, "session_id"].unique())
        count = len(sessions)
        if count < MIN_SESSIONS_TO_SPLIT:
            raise ModelError(
                f"Label {label} has {count} sessions; at least {MIN_SESSIONS_TO_SPLIT} "
                "are needed to make separate train, validation and test sets."
            )

        shuffled = [str(session) for session in rng.permutation(sessions)]
        n_test = max(1, round(count * test_fraction))
        n_validation = max(1, round(count * validation_fraction))
        # Always leave at least one session for training.
        n_validation = min(n_validation, count - 1 - n_test)

        test.extend(shuffled[:n_test])
        validation.extend(shuffled[n_test : n_test + n_validation])
        train.extend(shuffled[n_test + n_validation :])

    split = SessionSplit(train=train, validation=validation, test=test)
    check_no_session_overlap(split)
    return split


def select_rows(
    data: pd.DataFrame, sessions: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (features, labels) for the rows that belong to the given sessions."""
    subset = data.loc[data["session_id"].isin(list(sessions))]
    if subset.empty:
        raise ModelError("No dataset rows were found for the requested sessions.")
    features = subset[feature_column_names()].to_numpy(dtype=np.float32)
    labels = subset["label"].to_numpy(dtype=str)
    return features, labels


# ---------------------------------------------------------------------------
# Models and predictions
# ---------------------------------------------------------------------------
def get_candidate_models(seed: int = config.RANDOM_SEED) -> Dict[str, BaseEstimator]:
    """The classifiers we compare. All of them can output probabilities.

    StandardScaler is inside a pipeline, so when the pipeline is fitted the
    scaler only learns from the data it is fitted on (never from test data).
    """
    return {
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=seed),
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)
        ),
        "svm": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", probability=True, random_state=seed)
        ),
    }


def predict_with_confidence(
    model: BaseEstimator, features: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict letters and a confidence value for each row of features.

    The prediction is the class with the highest probability, and the
    confidence is that probability. Using the probabilities for both keeps the
    letter and its confidence consistent (an SVM's own predict() can disagree
    with its probabilities). The live predictor must use this same function.

    Args:
        features: Array of shape (n, 63), or a single vector of 63 numbers.

    Raises:
        ValueError: Wrong shape or NaN/infinite values.
    """
    array = np.asarray(features, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != config.NUM_FEATURES:
        raise ValueError(
            f"Features must have shape (n, {config.NUM_FEATURES}), got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Features contain NaN or infinite values.")

    probabilities = model.predict_proba(array)
    best = probabilities.argmax(axis=1)
    letters = np.asarray(model.classes_)[best]
    confidence = probabilities[np.arange(len(best)), best]
    return letters, confidence


def select_best_model(
    candidates: Dict[str, BaseEstimator],
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """Train each candidate on the training set and score it on the validation set.

    The best model is the one with the highest validation macro F1 (which
    treats every letter equally). If two tie, the first one listed wins.

    Returns:
        (name of the best model, scores for every candidate).
    """
    scores: Dict[str, Dict[str, float]] = {}
    for name, estimator in candidates.items():
        fitted = clone(estimator).fit(train_features, train_labels)
        predicted, _ = predict_with_confidence(fitted, validation_features)
        scores[name] = {
            "accuracy": float(accuracy_score(validation_labels, predicted)),
            "macro_f1": float(
                f1_score(validation_labels, predicted, average="macro", zero_division=0)
            ),
        }
        logger.info(
            "Validation %s: accuracy %.3f, macro F1 %.3f",
            name,
            scores[name]["accuracy"],
            scores[name]["macro_f1"],
        )

    best_name = max(scores, key=lambda model_name: scores[model_name]["macro_f1"])
    return best_name, scores


def train_final_model(
    estimator: BaseEstimator, features: np.ndarray, labels: np.ndarray
) -> BaseEstimator:
    """Fit a fresh copy of the chosen model. Pass train + validation data here."""
    return clone(estimator).fit(features, labels)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    model: BaseEstimator,
    features: np.ndarray,
    true_labels: Sequence[str],
    labels: Sequence[str],
    confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """Score a model on a labelled set. Every number is computed from the data.

    Returns a dict with overall accuracy, macro precision/recall/F1,
    per-letter scores, the confusion matrix and how the confidence threshold
    behaves (what fraction of predictions pass it, and how accurate those are).
    """
    predicted, confidence = predict_with_confidence(model, features)
    true_array = np.asarray(true_labels, dtype=str)
    if len(true_array) != len(predicted):
        raise ValueError("true_labels and features must have the same length.")
    labels = [str(label) for label in labels]

    precision, recall, f1_values, support = precision_recall_fscore_support(
        true_array, predicted, labels=labels, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1_values[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    matrix = confusion_matrix(true_array, predicted, labels=labels)

    above_threshold = confidence >= confidence_threshold
    accuracy_above: Optional[float] = None
    if above_threshold.any():
        accuracy_above = float(
            np.mean(predicted[above_threshold] == true_array[above_threshold])
        )

    # What you would score by always guessing the most common letter.
    _, counts = np.unique(true_array, return_counts=True)

    return {
        "num_samples": int(len(true_array)),
        "labels": labels,
        "accuracy": float(accuracy_score(true_array, predicted)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1_values)),
        "majority_class_baseline_accuracy": float(counts.max() / len(true_array)),
        "confidence_threshold": float(confidence_threshold),
        "fraction_above_threshold": float(np.mean(above_threshold)),
        "accuracy_above_threshold": accuracy_above,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def save_confusion_matrix_image(
    matrix: np.ndarray,
    labels: Sequence[str],
    path: PathLike,
    title: str = "Confusion matrix (held-out test sessions)",
) -> None:
    """Draw the confusion matrix and save it as a PNG file.

    Rows are the true letters, columns are the predicted letters. Numbers on
    the diagonal are correct predictions.
    """
    # Figure is used directly (not pyplot) so no window or GUI backend is needed.
    from matplotlib.figure import Figure

    matrix = np.asarray(matrix)
    size = len(labels)
    figure = Figure(figsize=(max(4.0, 0.7 * size + 3.0), max(3.5, 0.7 * size + 2.5)))
    axis = figure.subplots()

    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(size))
    axis.set_xticklabels(labels)
    axis.set_yticks(range(size))
    axis.set_yticklabels(labels)
    axis.set_xlabel("Predicted letter")
    axis.set_ylabel("True letter")
    axis.set_title(title)

    # Use white text on dark cells so the numbers stay readable.
    cutoff = matrix.max() / 2 if matrix.max() > 0 else 0
    for row in range(size):
        for column in range(size):
            value = int(matrix[row, column])
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > cutoff else "black",
            )

    figure.colorbar(image, ax=axis)
    figure.tight_layout()

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------
def file_sha256(path: PathLike) -> str:
    """Fingerprint of a file. It changes whenever the file's content changes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_metadata(
    *,
    model_name: str,
    model: BaseEstimator,
    data: pd.DataFrame,
    data_path: PathLike,
    split: SessionSplit,
    validation_scores: Dict[str, Dict[str, float]],
    seed: int,
    test_fraction: float,
    validation_fraction: float,
) -> Dict[str, Any]:
    """Collect everything we want to remember about a trained model.

    The saved session lists let evaluate_model.py rebuild the exact same test
    set later, even after more data has been collected.
    """
    summary = summarize_dataset(data)

    def sample_count(sessions: Sequence[str]) -> int:
        return int(data["session_id"].isin(list(sessions)).sum())

    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_name": model_name,
        "labels": [str(label) for label in model.classes_],
        "num_features": config.NUM_FEATURES,
        "preprocessing": (
            "features.landmarks_to_features(): wrist moved to the origin, divided by "
            "the largest wrist-to-landmark distance, flattened to 63 numbers"
        ),
        # Live prediction must use the same mirror setting as data collection.
        "mirror_camera": config.MIRROR_CAMERA,
        "random_seed": seed,
        "dataset": {
            "file_name": Path(data_path).name,
            "sha256": file_sha256(data_path),
            "num_samples": int(len(data)),
            "samples_per_label": {row.label: row.samples for row in summary.itertuples()},
            "sessions_per_label": {row.label: row.sessions for row in summary.itertuples()},
        },
        "split": {
            "test_fraction": test_fraction,
            "validation_fraction": validation_fraction,
            "train_sessions": list(split.train),
            "validation_sessions": list(split.validation),
            "test_sessions": list(split.test),
            "train_samples": sample_count(split.train),
            "validation_samples": sample_count(split.validation),
            "test_samples": sample_count(split.test),
        },
        "validation_scores": validation_scores,
        "library_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def save_model(
    model: BaseEstimator,
    metadata: Dict[str, Any],
    model_path: PathLike = config.MODEL_FILE,
    metadata_path: PathLike = config.MODEL_METADATA_FILE,
) -> None:
    """Save the model (joblib) and its metadata (JSON).

    Both files are written under temporary names first and renamed at the end,
    so a failure half-way cannot leave a new model with old metadata.
    """
    model_file = Path(model_path)
    metadata_file = Path(metadata_path)
    temp_model = model_file.with_name(model_file.name + ".tmp")
    temp_metadata = metadata_file.with_name(metadata_file.name + ".tmp")

    try:
        model_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, temp_model)
        temp_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        temp_model.replace(model_file)
        temp_metadata.replace(metadata_file)
    except OSError as error:
        raise ModelError(f"Could not save the model: {error}") from error


def _validate_metadata(model: Any, metadata: Dict[str, Any]) -> None:
    """Check that the metadata is complete and matches the loaded model."""
    required = ("schema_version", "model_name", "labels", "num_features", "split")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ModelError(f"Model metadata is missing: {', '.join(missing)}. Re-run train_model.py.")
    if metadata["schema_version"] != METADATA_SCHEMA_VERSION:
        raise ModelError("The model metadata has an unsupported version. Re-run train_model.py.")
    if metadata["num_features"] != config.NUM_FEATURES:
        raise ModelError("The model expects a different number of features. Re-run train_model.py.")
    for key in ("train_sessions", "validation_sessions", "test_sessions"):
        if key not in metadata["split"]:
            raise ModelError(f"Model metadata split is missing {key}. Re-run train_model.py.")

    if not (hasattr(model, "classes_") and hasattr(model, "predict_proba")):
        raise ModelError("The model file does not contain a usable classifier.")
    if [str(label) for label in model.classes_] != list(metadata["labels"]):
        raise ModelError(
            "The saved label list does not match the model's classes. Re-run train_model.py."
        )

    trained_with = metadata.get("library_versions", {}).get("scikit_learn")
    if trained_with and trained_with != sklearn.__version__:
        logger.warning(
            "Model was trained with scikit-learn %s but %s is installed. "
            "If loading or predictions misbehave, re-run train_model.py.",
            trained_with,
            sklearn.__version__,
        )


def load_model(
    model_path: PathLike = config.MODEL_FILE,
    metadata_path: PathLike = config.MODEL_METADATA_FILE,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """Load a saved model and its metadata, and check they belong together.

    Only load model files that you created yourself: joblib files are Python
    pickles and can run code when loaded.

    Raises:
        ModelError: A file is missing or unreadable, or the two do not match.
    """
    model_file = Path(model_path)
    metadata_file = Path(metadata_path)
    if not model_file.is_file() or not metadata_file.is_file():
        raise ModelError(
            f"Model files not found ({model_file} and {metadata_file}). "
            "Run scripts/train_model.py first."
        )

    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelError(f"Could not read {metadata_file}: {error}") from error

    try:
        model = joblib.load(model_file)
    except Exception as error:  # joblib can raise many kinds of errors here
        raise ModelError(f"Could not load {model_file}: {error}") from error

    _validate_metadata(model, metadata)
    return model, metadata


# ---------------------------------------------------------------------------
# Rebuilding the saved split (used by evaluate_model.py)
# ---------------------------------------------------------------------------
def split_from_metadata(metadata: Dict[str, Any]) -> SessionSplit:
    """Rebuild the SessionSplit that was saved at training time."""
    split = metadata["split"]
    return SessionSplit(
        train=list(split["train_sessions"]),
        validation=list(split["validation_sessions"]),
        test=list(split["test_sessions"]),
    )


def select_saved_test_rows(
    data: pd.DataFrame, metadata: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the features and labels of the test sessions chosen at training time.

    Raises:
        ModelError: The saved split overlaps, or a test session is no longer in
            the dataset (for example, it was removed with the undo key).
    """
    split = split_from_metadata(metadata)
    check_no_session_overlap(split)

    available = set(data["session_id"].unique())
    missing = sorted(set(split.test) - available)
    if missing:
        raise ModelError(
            f"{len(missing)} test session(s) are no longer in the dataset "
            f"(for example {missing[0]}). Re-run train_model.py to train and "
            "evaluate on the current data."
        )
    return select_rows(data, split.test)


def find_unseen_sessions(data: pd.DataFrame, metadata: Dict[str, Any]) -> List[str]:
    """Sessions in the dataset that the model's split does not mention.

    These were collected after training, so they were not used for anything.
    """
    split = split_from_metadata(metadata)
    known = set(split.train) | set(split.validation) | set(split.test)
    return sorted(str(session) for session in set(data["session_id"].unique()) - known)
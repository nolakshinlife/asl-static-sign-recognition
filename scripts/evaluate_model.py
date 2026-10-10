"""Evaluate the saved model on the held-out TEST sessions.

Run from the project root (Windows Command Prompt):
    python scripts\\evaluate_model.py

The test sessions are the ones train_model.py set aside and recorded in the
model metadata, so the model never saw them during training or model choice.

Files written to the reports folder:
    test_metrics.json, confusion_matrix.png, confusion_matrix.csv
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Lets this script run without installing the package first.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asl_recognition import config  # noqa: E402
from asl_recognition.dataset import DatasetError, load_dataset  # noqa: E402
from asl_recognition.logging_setup import setup_logging  # noqa: E402
from asl_recognition.model import (  # noqa: E402
    ModelError,
    evaluate_model,
    file_sha256,
    find_unseen_sessions,
    load_model,
    save_confusion_matrix_image,
    select_saved_test_rows,
)

logger = logging.getLogger("evaluate_model")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(description="Evaluate the ASL classifier on the test set.")
    parser.add_argument(
        "--data", type=Path, default=config.RAW_DATASET_FILE, help="Dataset CSV file."
    )
    parser.add_argument(
        "--model-path", type=Path, default=config.MODEL_FILE, help="Trained model file."
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=config.MODEL_METADATA_FILE,
        help="Model metadata file (JSON).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=config.REPORTS_DIR,
        help="Folder for the metrics file and confusion matrix.",
    )
    return parser.parse_args(argv)


def percent(value: Optional[float]) -> str:
    """Format a 0-1 value as a percentage, or 'n/a' if there is no value."""
    return "n/a" if value is None else f"{value * 100:.1f}%"


def write_reports(
    reports_dir: Path,
    metrics: Dict[str, Any],
    metadata: Dict[str, Any],
    dataset_hash_now: str,
) -> List[Path]:
    """Save the metrics JSON, the confusion matrix CSV and the confusion matrix image."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / config.METRICS_FILENAME
    csv_path = reports_dir / config.CONFUSION_MATRIX_CSV_NAME
    image_path = reports_dir / config.CONFUSION_MATRIX_IMAGE_NAME

    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_name": metadata["model_name"],
        "trained_at": metadata.get("trained_at"),
        "dataset_sha256_at_training": metadata.get("dataset", {}).get("sha256"),
        "dataset_sha256_now": dataset_hash_now,
        "num_test_sessions": len(metadata["split"]["test_sessions"]),
        "metrics": metrics,
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    labels = metrics["labels"]
    matrix = np.array(metrics["confusion_matrix"])
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(csv_path, index_label="true_letter")
    save_confusion_matrix_image(matrix, labels, image_path)
    return [metrics_path, csv_path, image_path]


def print_metrics(metrics: Dict[str, Any], num_sessions: int) -> None:
    """Print the evaluation results."""
    print(f"\nHeld-out test set: {metrics['num_samples']} samples from {num_sessions} sessions")
    print(f"  Accuracy:                {percent(metrics['accuracy'])}")
    print(f"  Macro precision:         {percent(metrics['macro_precision'])}")
    print(f"  Macro recall:            {percent(metrics['macro_recall'])}")
    print(f"  Macro F1:                {percent(metrics['macro_f1'])}")
    print(
        f"  Always-guess-most-common baseline: "
        f"{percent(metrics['majority_class_baseline_accuracy'])}"
    )
    print(
        f"  Confidence >= {metrics['confidence_threshold']:.2f}: "
        f"{percent(metrics['fraction_above_threshold'])} of predictions pass, "
        f"accuracy on those: {percent(metrics['accuracy_above_threshold'])}"
    )

    print("\nPer-letter results:")
    print("  Letter  Precision  Recall     F1  Samples")
    for label, result in metrics["per_class"].items():
        print(
            f"  {label:<6}  {result['precision']:9.3f}  {result['recall']:6.3f}  "
            f"{result['f1']:5.3f}  {result['support']:7d}"
        )

    labels = metrics["labels"]
    matrix = pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels)
    print("\nConfusion matrix (rows = true letter, columns = predicted letter):")
    print(matrix.to_string())

    print(
        "\nNote: the test sessions come from the same person, camera and setup as the\n"
        "training data. This score does not show how well the model works for other\n"
        "people, cameras or lighting."
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Evaluate the saved model. Returns 0 on success and 1 on error."""
    args = parse_args(argv)
    setup_logging()

    try:
        model, metadata = load_model(args.model_path, args.metadata_path)
        data = load_dataset(args.data)
        features, true_labels = select_saved_test_rows(data, metadata)

        unseen = find_unseen_sessions(data, metadata)
        if unseen:
            logger.warning(
                "%d session(s) in the dataset were collected after training and are "
                "ignored here. Re-run train_model.py to include them.",
                len(unseen),
            )
        dataset_hash_now = file_sha256(args.data)
        trained_hash = metadata.get("dataset", {}).get("sha256")
        if trained_hash and trained_hash != dataset_hash_now:
            logger.warning(
                "The dataset file has changed since training. The saved test sessions "
                "are still used, but consider re-training."
            )

        metrics = evaluate_model(
            model, features, true_labels, metadata["labels"], config.CONFIDENCE_THRESHOLD
        )
        written = write_reports(args.reports_dir, metrics, metadata, dataset_hash_now)
    except (DatasetError, ModelError, ValueError, OSError) as error:
        logger.error("%s", error)
        return 1

    print_metrics(metrics, len(metadata["split"]["test_sessions"]))
    print("\nSaved reports:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
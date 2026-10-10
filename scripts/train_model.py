"""Train the sign classifier from the collected dataset.

Run from the project root (Windows Command Prompt):
    python scripts\\train_model.py

What it does:
  1. Loads and validates the dataset (data\\raw\\landmarks.csv by default).
  2. Splits whole recording sessions into train / validation / test sets.
  3. Trains three candidate models and compares them on the VALIDATION set.
  4. Refits the best one on train + validation and saves it to models\\.

The test set is NOT used here. Run scripts\\evaluate_model.py for that, so
the test result is a score the model was never tuned against.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# Lets this script run without installing the package first.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asl_recognition import config  # noqa: E402
from asl_recognition.dataset import DatasetError, load_dataset  # noqa: E402
from asl_recognition.logging_setup import setup_logging  # noqa: E402
from asl_recognition.model import (  # noqa: E402
    ModelError,
    build_metadata,
    get_candidate_models,
    save_model,
    select_best_model,
    select_rows,
    split_sessions,
    train_final_model,
    validate_training_data,
)

logger = logging.getLogger("train_model")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(description="Train the ASL sign classifier.")
    parser.add_argument(
        "--data", type=Path, default=config.RAW_DATASET_FILE, help="Dataset CSV file."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=config.MODEL_FILE,
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=config.MODEL_METADATA_FILE,
        help="Where to save the model metadata (JSON).",
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed.")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=config.TEST_FRACTION,
        help="Fraction of each letter's sessions kept for the test set.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=config.VALIDATION_FRACTION,
        help="Fraction of each letter's sessions used to choose the model.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Train and save the model. Returns 0 on success and 1 on error."""
    args = parse_args(argv)
    setup_logging()

    try:
        data = load_dataset(args.data)
        validate_training_data(data)
        split = split_sessions(
            data,
            test_fraction=args.test_fraction,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )

        train_features, train_labels = select_rows(data, split.train)
        validation_features, validation_labels = select_rows(data, split.validation)

        candidates = get_candidate_models(args.seed)
        best_name, scores = select_best_model(
            candidates, train_features, train_labels, validation_features, validation_labels
        )

        # The choice is made, so the validation data can now be used for
        # training too. The test sessions stay untouched.
        final_model = train_final_model(
            candidates[best_name],
            np.vstack([train_features, validation_features]),
            np.concatenate([train_labels, validation_labels]),
        )

        metadata = build_metadata(
            model_name=best_name,
            model=final_model,
            data=data,
            data_path=args.data,
            split=split,
            validation_scores=scores,
            seed=args.seed,
            test_fraction=args.test_fraction,
            validation_fraction=args.validation_fraction,
        )
        save_model(final_model, metadata, args.model_path, args.metadata_path)
    except (DatasetError, ModelError, ValueError) as error:
        logger.error("%s", error)
        return 1

    print_report(metadata, scores, best_name, args)
    return 0


def print_report(metadata: dict, scores: dict, best_name: str, args: argparse.Namespace) -> None:
    """Print what was trained. The test set was not used, so no test score appears."""
    split = metadata["split"]
    dataset = metadata["dataset"]

    print()
    print(f"Dataset: {dataset['num_samples']} samples, letters {' '.join(metadata['labels'])}")
    print("Split (whole sessions, so one recording never sits in two sets):")
    print(f"  train       {len(split['train_sessions']):3d} sessions, {split['train_samples']:5d} samples")
    print(f"  validation  {len(split['validation_sessions']):3d} sessions, {split['validation_samples']:5d} samples")
    print(f"  test        {len(split['test_sessions']):3d} sessions, {split['test_samples']:5d} samples (not used yet)")

    print("\nModel comparison on the validation set:")
    for name, result in scores.items():
        marker = "  <- chosen" if name == best_name else ""
        print(
            f"  {name:<20} accuracy {result['accuracy']:.3f}   "
            f"macro F1 {result['macro_f1']:.3f}{marker}"
        )
    print(
        "\nValidation scores were used to pick the model, so they are a little "
        "optimistic.\nThe test score from evaluate_model.py is the honest one."
    )

    print(f"\nSaved model:    {args.model_path}")
    print(f"Saved metadata: {args.metadata_path}")
    print("\nNext step:  python scripts\\evaluate_model.py")


if __name__ == "__main__":
    sys.exit(main())
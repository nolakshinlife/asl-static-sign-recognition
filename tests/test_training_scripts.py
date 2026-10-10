"""End-to-end test of train_model.py and evaluate_model.py on synthetic data."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    """Import a file from the scripts folder (it is not part of the package)."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def workspace(tmp_path, make_dataset):
    """A temporary dataset CSV plus output paths, so no real project files are touched."""
    data = make_dataset(sessions_per_label=6, samples_per_session=30)
    csv_path = tmp_path / "landmarks.csv"
    data.to_csv(csv_path, index=False, float_format="%.6f")
    return {
        "csv": csv_path,
        "model": tmp_path / "models" / "model.joblib",
        "metadata": tmp_path / "models" / "metadata.json",
        "reports": tmp_path / "reports",
    }


def train_args(workspace):
    return [
        "--data", str(workspace["csv"]),
        "--model-path", str(workspace["model"]),
        "--metadata-path", str(workspace["metadata"]),
    ]


def evaluate_args(workspace):
    return train_args(workspace) + ["--reports-dir", str(workspace["reports"])]


def test_train_then_evaluate_end_to_end(workspace):
    train_model = load_script("train_model")
    evaluate_model = load_script("evaluate_model")

    assert train_model.main(train_args(workspace)) == 0
    assert workspace["model"].is_file()
    assert workspace["metadata"].is_file()

    assert evaluate_model.main(evaluate_args(workspace)) == 0
    reports = workspace["reports"]
    assert (reports / "test_metrics.json").is_file()
    assert (reports / "confusion_matrix.png").is_file()
    assert (reports / "confusion_matrix.csv").is_file()

    metadata = json.loads(workspace["metadata"].read_text(encoding="utf-8"))
    report = json.loads((reports / "test_metrics.json").read_text(encoding="utf-8"))
    expected_samples = 30 * len(metadata["split"]["test_sessions"])
    assert report["metrics"]["num_samples"] == expected_samples
    assert 0.0 <= report["metrics"]["accuracy"] <= 1.0


def test_train_fails_when_dataset_is_missing(workspace):
    train_model = load_script("train_model")
    workspace["csv"].unlink()
    assert train_model.main(train_args(workspace)) == 1
    assert not workspace["model"].exists()


def test_train_fails_when_dataset_is_too_small(tmp_path, make_dataset):
    train_model = load_script("train_model")
    small = make_dataset(sessions_per_label=2, samples_per_session=60)
    csv_path = tmp_path / "small.csv"
    small.to_csv(csv_path, index=False, float_format="%.6f")
    args = [
        "--data", str(csv_path),
        "--model-path", str(tmp_path / "model.joblib"),
        "--metadata-path", str(tmp_path / "metadata.json"),
    ]
    assert train_model.main(args) == 1
    assert not (tmp_path / "model.joblib").exists()


def test_evaluate_fails_when_no_model_exists(workspace):
    evaluate_model = load_script("evaluate_model")
    assert evaluate_model.main(evaluate_args(workspace)) == 1
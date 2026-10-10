"""Tests for the dataset module. No webcam is needed."""

import numpy as np
import pytest

from asl_recognition import config
from asl_recognition.dataset import (
    DatasetError,
    SessionRecorder,
    append_samples,
    csv_columns,
    find_dataset_problems,
    load_dataset,
    load_summary,
    normalize_label,
    normalize_label_list,
    remove_session,
    summarize_dataset,
    validate_sample_count,
)
from asl_recognition.features import feature_column_names


def make_features(seed: int = 0) -> np.ndarray:
    """Random feature vector of the correct size, used only as test input."""
    rng = np.random.default_rng(seed)
    return rng.random(config.NUM_FEATURES).astype(np.float32)


# ----- labels and counts -----
def test_normalize_label_cleans_input():
    assert normalize_label(" a ") == "A"


@pytest.mark.parametrize("bad_label", ["J", "Z", "AB", "", "1", 5, None])
def test_normalize_label_rejects_invalid(bad_label):
    with pytest.raises(ValueError):
        normalize_label(bad_label)


def test_normalize_label_list_removes_duplicates_and_keeps_order():
    assert normalize_label_list(["b", "A", "B"]) == ["B", "A"]


def test_normalize_label_list_rejects_empty_list():
    with pytest.raises(ValueError):
        normalize_label_list([])


def test_default_labels_are_valid_static_signs():
    assert normalize_label_list(config.DEFAULT_LABELS) == list(config.DEFAULT_LABELS)


@pytest.mark.parametrize("bad_count", [0, -5, 10_000, 2.5, "10", True])
def test_validate_sample_count_rejects_invalid(bad_count):
    with pytest.raises(ValueError):
        validate_sample_count(bad_count)


def test_validate_sample_count_accepts_valid():
    assert validate_sample_count(100) == 100


# ----- SessionRecorder -----
def test_recorder_ignores_frames_during_countdown():
    recorder = SessionRecorder(
        "A", target_count=3, start_time=0.0, countdown_seconds=2.0, interval_seconds=0.1
    )
    assert recorder.countdown_remaining(1.0) == pytest.approx(1.0)
    assert recorder.try_add(make_features(), now=1.0) is False
    assert recorder.try_add(make_features(), now=2.0) is True


def test_recorder_respects_interval_and_stops_when_complete():
    recorder = SessionRecorder(
        "A", target_count=2, start_time=0.0, countdown_seconds=0.0, interval_seconds=0.1
    )
    assert recorder.try_add(make_features(1), now=1.0) is True
    assert recorder.try_add(make_features(2), now=1.05) is False  # too soon
    assert recorder.try_add(make_features(3), now=1.2) is True
    assert recorder.is_complete
    assert recorder.try_add(make_features(4), now=2.0) is False  # already complete
    assert len(recorder.samples) == 2


def test_recorder_rejects_bad_settings():
    with pytest.raises(ValueError):
        SessionRecorder("J", target_count=10, start_time=0.0)
    with pytest.raises(ValueError):
        SessionRecorder("A", target_count=0, start_time=0.0)
    with pytest.raises(ValueError):
        SessionRecorder("A", target_count=10, start_time=0.0, countdown_seconds=-1)


# ----- saving and loading -----
def test_append_and_load_round_trip(tmp_path):
    path = tmp_path / "data.csv"
    features = [make_features(1), make_features(2)]

    saved = append_samples(path, "s1", "a", features)
    data = load_dataset(path)

    assert saved == 2
    assert list(data.columns) == csv_columns()
    assert (data["label"] == "A").all()
    assert (data["session_id"] == "s1").all()
    loaded = data[feature_column_names()].to_numpy()
    assert np.allclose(loaded, np.array(features), atol=1e-6)


def test_header_is_written_only_once(tmp_path):
    path = tmp_path / "data.csv"
    append_samples(path, "s1", "A", [make_features(1)])
    append_samples(path, "s2", "B", [make_features(2), make_features(3)])
    data = load_dataset(path)
    assert len(data) == 3
    assert set(data["label"]) == {"A", "B"}


def test_append_with_no_samples_writes_nothing(tmp_path):
    path = tmp_path / "data.csv"
    assert append_samples(path, "s1", "A", []) == 0
    assert not path.exists()


def test_append_rejects_bad_features_without_writing(tmp_path):
    path = tmp_path / "data.csv"
    nan_features = make_features()
    nan_features[5] = np.nan
    with pytest.raises(ValueError):
        append_samples(path, "s1", "A", [make_features(), nan_features])
    with pytest.raises(ValueError):
        append_samples(path, "s1", "A", [np.zeros(10)])
    assert not path.exists()  # nothing was written


def test_append_rejects_bad_label_and_session_id(tmp_path):
    path = tmp_path / "data.csv"
    with pytest.raises(ValueError):
        append_samples(path, "s1", "Z", [make_features()])
    with pytest.raises(ValueError):
        append_samples(path, "  ", "A", [make_features()])


def test_append_to_file_with_wrong_columns_raises(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(DatasetError):
        append_samples(path, "s1", "A", [make_features()])


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(tmp_path / "missing.csv")


def test_load_rejects_unknown_label(tmp_path):
    path = tmp_path / "data.csv"
    header = ",".join(csv_columns())
    row = "s1,J," + ",".join(["0.1"] * config.NUM_FEATURES)
    path.write_text(header + "\n" + row + "\n", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_dataset(path)


def test_load_rejects_non_numeric_features(tmp_path):
    path = tmp_path / "data.csv"
    header = ",".join(csv_columns())
    values = ["abc"] + ["0.1"] * (config.NUM_FEATURES - 1)
    path.write_text(header + "\ns1,A," + ",".join(values) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_dataset(path)


# ----- summaries -----
def test_summary_and_problems(tmp_path):
    path = tmp_path / "data.csv"
    append_samples(path, "s1", "A", [make_features(1)] * 3)
    append_samples(path, "s2", "A", [make_features(2)] * 3)
    append_samples(path, "s3", "B", [make_features(3)] * 2)

    summary = summarize_dataset(load_dataset(path), ["A", "B", "C"])
    rows = {row.label: (row.samples, row.sessions) for row in summary.itertuples()}
    assert rows == {"A": (6, 2), "B": (2, 1), "C": (0, 0)}

    problems = find_dataset_problems(summary, min_samples=3, min_sessions=2)
    assert not any(problem.startswith("Label A") for problem in problems)
    assert any(problem.startswith("Label B") for problem in problems)
    assert any(problem.startswith("Label C") for problem in problems)


def test_problems_require_two_labels(tmp_path):
    path = tmp_path / "data.csv"
    append_samples(path, "s1", "A", [make_features(1)])
    summary = summarize_dataset(load_dataset(path))
    problems = find_dataset_problems(summary, min_samples=1, min_sessions=1)
    assert any("2 different labels" in problem for problem in problems)


def test_load_summary_without_file_gives_zero_counts(tmp_path):
    summary = load_summary(tmp_path / "missing.csv", ["A", "B"])
    assert list(summary["samples"]) == [0, 0]
    assert list(summary["sessions"]) == [0, 0]


# ----- undo -----
def test_remove_session_only_removes_that_session(tmp_path):
    path = tmp_path / "data.csv"
    append_samples(path, "s1", "A", [make_features(1)] * 2)
    append_samples(path, "s2", "B", [make_features(2)] * 3)

    assert remove_session(path, "s1") == 2
    data = load_dataset(path)
    assert set(data["session_id"]) == {"s2"}
    assert len(data) == 3
    assert remove_session(path, "does-not-exist") == 0


def test_removing_last_session_leaves_valid_empty_file(tmp_path):
    path = tmp_path / "data.csv"
    append_samples(path, "s1", "A", [make_features(1)])
    remove_session(path, "s1")

    assert len(load_dataset(path, allow_empty=True)) == 0
    with pytest.raises(DatasetError):
        load_dataset(path)
    # Collecting again still works after the file was emptied.
    append_samples(path, "s2", "A", [make_features(2)])
    assert len(load_dataset(path)) == 1
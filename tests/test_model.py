"""Tests for splitting, training, evaluation and saving. Uses synthetic data only."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from asl_recognition import config
from asl_recognition.features import feature_column_names
from asl_recognition.model import (
    ModelError,
    SessionSplit,
    build_metadata,
    check_no_session_overlap,
    evaluate_model,
    file_sha256,
    find_unseen_sessions,
    load_model,
    predict_with_confidence,
    save_confusion_matrix_image,
    save_model,
    select_best_model,
    select_rows,
    select_saved_test_rows,
    split_sessions,
    train_final_model,
    validate_training_data,
)


def fast_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def train_small_model(data, csv_path):
    """Split, train and build metadata the same way train_model.py does."""
    data.to_csv(csv_path, index=False, float_format="%.6f")
    split = split_sessions(data)
    features, labels = select_rows(data, split.train + split.validation)
    model = train_final_model(fast_model(), features, labels)
    metadata = build_metadata(
        model_name="logistic_regression",
        model=model,
        data=data,
        data_path=csv_path,
        split=split,
        validation_scores={},
        seed=config.RANDOM_SEED,
        test_fraction=config.TEST_FRACTION,
        validation_fraction=config.VALIDATION_FRACTION,
    )
    return split, model, metadata


# ----- validate_training_data -----
def test_validate_accepts_good_data(make_dataset):
    validate_training_data(make_dataset())


def test_validate_rejects_too_few_sessions(make_dataset):
    with pytest.raises(ModelError, match="sessions"):
        validate_training_data(make_dataset(sessions_per_label=2, samples_per_session=60))


def test_validate_rejects_too_few_samples(make_dataset):
    with pytest.raises(ModelError, match="samples"):
        validate_training_data(make_dataset(sessions_per_label=3, samples_per_session=10))


def test_validate_rejects_single_label(make_dataset):
    with pytest.raises(ModelError, match="2 different labels"):
        validate_training_data(make_dataset(labels=("A",)))


def test_validate_rejects_features_that_are_not_normalized(make_dataset):
    data = make_dataset()
    data["x0"] = 0.5  # the wrist should always be at 0
    with pytest.raises(ModelError, match="normalized"):
        validate_training_data(data)


# ----- split_sessions -----
def test_split_has_no_overlap_and_covers_every_label(make_dataset):
    data = make_dataset()
    split = split_sessions(data)

    parts = [set(split.train), set(split.validation), set(split.test)]
    assert not parts[0] & parts[1]
    assert not parts[0] & parts[2]
    assert not parts[1] & parts[2]
    assert set.union(*parts) == set(data["session_id"].unique())

    for part in parts:
        letters = set(data.loc[data["session_id"].isin(part), "label"])
        assert letters == {"A", "B", "C"}


def test_split_keeps_whole_sessions_together(make_dataset):
    # Every row of a session must land in exactly one part of the split.
    data = make_dataset()
    split = split_sessions(data)
    for session in data["session_id"].unique():
        memberships = [
            session in split.train,
            session in split.validation,
            session in split.test,
        ]
        assert sum(memberships) == 1


def test_split_sizes_with_ten_sessions_per_label(make_dataset):
    data = make_dataset(sessions_per_label=10)
    split = split_sessions(data)
    # 20% of 10 sessions = 2 test and 2 validation sessions for each of 3 labels.
    assert len(split.test) == 6
    assert len(split.validation) == 6
    assert len(split.train) == 18


def test_split_with_minimum_sessions_uses_one_session_each(make_dataset):
    split = split_sessions(make_dataset(sessions_per_label=3))
    assert (len(split.train), len(split.validation), len(split.test)) == (3, 3, 3)


def test_split_is_repeatable(make_dataset):
    data = make_dataset(sessions_per_label=8)
    assert split_sessions(data, seed=7) == split_sessions(data, seed=7)


def test_split_requires_three_sessions_per_label(make_dataset):
    with pytest.raises(ModelError, match="at least 3"):
        split_sessions(make_dataset(sessions_per_label=2))


@pytest.mark.parametrize("bad_fraction", [0.0, -0.1, 0.9])
def test_split_rejects_bad_fractions(make_dataset, bad_fraction):
    data = make_dataset()
    with pytest.raises(ValueError):
        split_sessions(data, test_fraction=bad_fraction)
    with pytest.raises(ValueError):
        split_sessions(data, validation_fraction=bad_fraction)


def test_overlap_check_detects_leakage():
    leaky = SessionSplit(train=["s1", "s2"], validation=["s3"], test=["s2"])
    with pytest.raises(ModelError, match="leakage"):
        check_no_session_overlap(leaky)


# ----- select_rows -----
def test_select_rows_returns_matching_features_and_labels(make_dataset):
    data = make_dataset()
    features, labels = select_rows(data, ["A-session0", "B-session1"])
    assert features.shape == (60, config.NUM_FEATURES)
    assert features.dtype == np.float32
    assert set(labels) == {"A", "B"}


def test_select_rows_rejects_unknown_sessions(make_dataset):
    with pytest.raises(ModelError):
        select_rows(make_dataset(), ["does-not-exist"])


# ----- model selection and prediction -----
def test_select_best_model_scores_every_candidate(make_dataset):
    data = make_dataset()
    split = split_sessions(data)
    train_x, train_y = select_rows(data, split.train)
    val_x, val_y = select_rows(data, split.validation)

    candidates = {
        "logistic_regression": fast_model(),
        "small_forest": RandomForestClassifier(n_estimators=20, random_state=0),
    }
    best_name, scores = select_best_model(candidates, train_x, train_y, val_x, val_y)

    assert set(scores) == set(candidates)
    assert best_name in candidates
    assert scores[best_name]["macro_f1"] == max(s["macro_f1"] for s in scores.values())
    for result in scores.values():
        assert 0.0 <= result["accuracy"] <= 1.0
    # The synthetic letters are far apart, so a sensible model should separate them.
    assert scores[best_name]["accuracy"] > 0.9


def test_predict_with_confidence_shapes_and_range(make_dataset):
    data = make_dataset()
    features, labels = select_rows(data, ["A-session0", "B-session0", "C-session0"])
    model = train_final_model(fast_model(), features, labels)

    letters, confidence = predict_with_confidence(model, features)
    assert letters.shape == (90,)
    assert confidence.shape == (90,)
    # With 3 classes the top probability is at least 1/3 and at most 1.
    assert np.all(confidence >= 1 / 3 - 1e-9)
    assert np.all(confidence <= 1 + 1e-9)


def test_predict_with_confidence_accepts_one_vector_and_rejects_bad_input(make_dataset):
    data = make_dataset()
    features, labels = select_rows(data, ["A-session0", "B-session0", "C-session0"])
    model = train_final_model(fast_model(), features, labels)

    letters, _ = predict_with_confidence(model, features[0])
    assert letters.shape == (1,)

    with pytest.raises(ValueError):
        predict_with_confidence(model, np.zeros((2, 10)))
    bad = features[:1].copy()
    bad[0, 5] = np.nan
    with pytest.raises(ValueError):
        predict_with_confidence(model, bad)


# ----- evaluate_model -----
def test_evaluate_model_reports_consistent_numbers(make_dataset):
    data = make_dataset()
    split = split_sessions(data)
    train_x, train_y = select_rows(data, split.train)
    test_x, test_y = select_rows(data, split.test)
    model = train_final_model(fast_model(), train_x, train_y)

    metrics = evaluate_model(model, test_x, test_y, ["A", "B", "C"], confidence_threshold=0.7)

    matrix = np.array(metrics["confusion_matrix"])
    assert matrix.shape == (3, 3)
    assert matrix.sum() == len(test_y) == metrics["num_samples"]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert set(metrics["per_class"]) == {"A", "B", "C"}
    assert sum(item["support"] for item in metrics["per_class"].values()) == len(test_y)
    # The accuracy must equal the diagonal of the confusion matrix over the total.
    assert metrics["accuracy"] == pytest.approx(np.trace(matrix) / matrix.sum())


def test_evaluate_model_confidence_threshold_extremes(make_dataset):
    data = make_dataset()
    features, labels = select_rows(data, ["A-session0", "B-session0", "C-session0"])
    model = train_final_model(fast_model(), features, labels)

    everything = evaluate_model(model, features, labels, ["A", "B", "C"], confidence_threshold=0.0)
    assert everything["fraction_above_threshold"] == 1.0

    nothing = evaluate_model(model, features, labels, ["A", "B", "C"], confidence_threshold=1.01)
    assert nothing["fraction_above_threshold"] == 0.0
    assert nothing["accuracy_above_threshold"] is None


def test_confusion_matrix_image_is_saved(tmp_path):
    path = tmp_path / "reports" / "matrix.png"
    save_confusion_matrix_image(np.array([[3, 1], [0, 4]]), ["A", "B"], path)
    assert path.is_file()
    assert path.stat().st_size > 0


# ----- saving and loading -----
def test_save_and_load_round_trip(tmp_path, make_dataset):
    data = make_dataset()
    split, model, metadata = train_small_model(data, tmp_path / "data.csv")
    model_path = tmp_path / "models" / "model.joblib"
    metadata_path = tmp_path / "models" / "metadata.json"

    save_model(model, metadata, model_path, metadata_path)
    loaded_model, loaded_metadata = load_model(model_path, metadata_path)

    assert loaded_metadata["labels"] == ["A", "B", "C"]
    assert loaded_metadata["split"]["test_sessions"] == split.test
    assert loaded_metadata["num_features"] == config.NUM_FEATURES

    features, _ = select_rows(data, split.test)
    original, _ = predict_with_confidence(model, features)
    reloaded, _ = predict_with_confidence(loaded_model, features)
    assert np.array_equal(original, reloaded)


def test_metadata_split_has_no_overlap(tmp_path, make_dataset):
    split, _, metadata = train_small_model(make_dataset(), tmp_path / "data.csv")
    saved = metadata["split"]
    assert not set(saved["train_sessions"]) & set(saved["test_sessions"])
    assert not set(saved["validation_sessions"]) & set(saved["test_sessions"])
    assert saved["test_samples"] == 30 * len(split.test)


def test_load_model_reports_missing_files(tmp_path):
    with pytest.raises(ModelError, match="train_model.py"):
        load_model(tmp_path / "missing.joblib", tmp_path / "missing.json")


def test_load_model_detects_label_mismatch(tmp_path, make_dataset):
    _, model, metadata = train_small_model(make_dataset(), tmp_path / "data.csv")
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    save_model(model, metadata, model_path, metadata_path)

    tampered = json.loads(metadata_path.read_text(encoding="utf-8"))
    tampered["labels"] = ["A", "B", "Z"]
    metadata_path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ModelError, match="label"):
        load_model(model_path, metadata_path)


# ----- rebuilding the saved test set -----
def test_select_saved_test_rows_uses_only_test_sessions(tmp_path, make_dataset):
    data = make_dataset()
    split, _, metadata = train_small_model(data, tmp_path / "data.csv")
    features, labels = select_saved_test_rows(data, metadata)
    assert features.shape == (30 * len(split.test), config.NUM_FEATURES)
    assert len(labels) == len(features)


def test_select_saved_test_rows_fails_if_a_test_session_was_removed(tmp_path, make_dataset):
    data = make_dataset()
    split, _, metadata = train_small_model(data, tmp_path / "data.csv")
    reduced = data[data["session_id"] != split.test[0]]
    with pytest.raises(ModelError, match="no longer"):
        select_saved_test_rows(reduced, metadata)


def test_find_unseen_sessions_reports_new_data(tmp_path, make_dataset):
    data = make_dataset()
    _, _, metadata = train_small_model(data, tmp_path / "data.csv")
    assert find_unseen_sessions(data, metadata) == []

    new_rows = data[data["session_id"] == "A-session0"].copy()
    new_rows["session_id"] = "A-collected-later"
    extended = pd.concat([data, new_rows], ignore_index=True)
    assert find_unseen_sessions(extended, metadata) == ["A-collected-later"]


# ----- file_sha256 -----
def test_file_sha256_changes_with_content(tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("first", encoding="utf-8")
    first = file_sha256(path)
    assert first == file_sha256(path)
    path.write_text("second", encoding="utf-8")
    assert file_sha256(path) != first
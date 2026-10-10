"""Tests for the live predictor.

The status logic is tested with a scripted fake model and a fake detector, so
every frame's outcome is known in advance. One test also trains a small real
classifier on synthetic landmarks to check the preprocessing path end to end.
All data here is made up and says nothing about real-world accuracy.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from asl_recognition import config
from asl_recognition.features import landmarks_to_features
from asl_recognition.hand_detector import HandDetector
from asl_recognition.model import ModelError
from asl_recognition.predictor import (
    PredictionStatus,
    SignPredictor,
    prepare_frame,
)

LABELS = ["A", "B", "C"]
BLANK_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
HAND = np.random.default_rng(0).random((config.NUM_LANDMARKS, config.COORDS_PER_LANDMARK))

# Probability rows in the order of LABELS.
A_SURE = [0.90, 0.05, 0.05]
B_SURE = [0.05, 0.90, 0.05]
C_SURE = [0.05, 0.05, 0.90]
UNSURE = [0.50, 0.30, 0.20]


class FakeDetector:
    """Returns the prepared landmarks (or None) one frame at a time."""

    def __init__(self, detections):
        self._detections = list(detections)
        self.closed = False

    def detect(self, frame_bgr):
        return self._detections.pop(0)

    def close(self):
        self.closed = True


class ScriptedModel:
    """Fake classifier that returns prepared probability rows, one per call."""

    classes_ = np.array(LABELS)

    def __init__(self, rows):
        self._rows = [np.array([row]) for row in rows]

    def predict_proba(self, features):
        return self._rows.pop(0)


def make_metadata(**overrides):
    metadata = {"labels": list(LABELS), "mirror_camera": config.MIRROR_CAMERA, "model_name": "scripted"}
    metadata.update(overrides)
    return metadata


def make_predictor(frames, window=5, threshold=0.7):
    """frames: one entry per video frame, either a probability row or None (no hand)."""
    detector = FakeDetector([None if frame is None else HAND for frame in frames])
    model = ScriptedModel([frame for frame in frames if frame is not None])
    predictor = SignPredictor(
        model, make_metadata(), detector, confidence_threshold=threshold, smoothing_window=window
    )
    return predictor, detector


def run(predictor, count):
    return [predictor.process_frame(BLANK_FRAME) for _ in range(count)]


# ----- status logic -----
def test_no_hand_gives_no_hand_status():
    predictor, _ = make_predictor([None] * 3)
    for result in run(predictor, 3):
        assert result.status is PredictionStatus.NO_HAND
        assert result.letter is None
        assert result.hand_detected is False
        assert result.confidence == 0.0
        assert result.landmarks is None
        assert result.message


def test_steady_confident_frames_become_recognized():
    predictor, _ = make_predictor([A_SURE] * 4, window=5)  # needs 3 votes
    results = run(predictor, 4)
    assert [r.status for r in results] == [
        PredictionStatus.UNSTABLE,
        PredictionStatus.UNSTABLE,
        PredictionStatus.RECOGNIZED,
        PredictionStatus.RECOGNIZED,
    ]
    assert results[-1].letter == "A"
    assert results[-1].confidence == pytest.approx(0.9)
    assert results[-1].raw_letter == "A"
    assert results[-1].hand_detected is True
    assert results[-1].landmarks is not None


def test_low_confidence_is_uncertain_and_shows_no_letter():
    predictor, _ = make_predictor([UNSURE] * 4)
    for result in run(predictor, 4):
        assert result.status is PredictionStatus.UNCERTAIN
        assert result.letter is None
        assert result.raw_letter == "A"
        assert result.raw_confidence == pytest.approx(0.5)


def test_frame_to_frame_disagreement_stays_unstable():
    predictor, _ = make_predictor([A_SURE, B_SURE, C_SURE, A_SURE, B_SURE, C_SURE], window=5)
    for result in run(predictor, 6):
        assert result.status is PredictionStatus.UNSTABLE
        assert result.letter is None


def test_letter_survives_short_hand_loss_then_disappears():
    predictor, _ = make_predictor([A_SURE] * 5 + [None] * 3, window=5)
    results = run(predictor, 8)
    assert results[4].status is PredictionStatus.RECOGNIZED
    # 1 and 2 lost frames: A still has 4 and 3 votes out of the last 5.
    assert [r.status for r in results[5:]] == [
        PredictionStatus.RECOGNIZED,
        PredictionStatus.RECOGNIZED,
        PredictionStatus.NO_HAND,
    ]
    assert all(r.hand_detected is False for r in results[5:])


def test_letter_switches_after_enough_new_votes():
    predictor, _ = make_predictor([A_SURE] * 5 + [B_SURE] * 3, window=5)
    results = run(predictor, 8)
    assert [r.letter for r in results[5:]] == ["A", "A", "B"]


def test_reset_forgets_previous_frames():
    predictor, _ = make_predictor([A_SURE] * 4, window=5)
    results = run(predictor, 3)
    assert results[-1].status is PredictionStatus.RECOGNIZED
    predictor.reset()
    assert predictor.process_frame(BLANK_FRAME).status is PredictionStatus.UNSTABLE


def test_predict_single_does_not_use_smoothing():
    predictor, _ = make_predictor([A_SURE, UNSURE, None])
    first = predictor.predict_single(BLANK_FRAME)
    second = predictor.predict_single(BLANK_FRAME)
    third = predictor.predict_single(BLANK_FRAME)
    assert (first.status, first.letter) == (PredictionStatus.RECOGNIZED, "A")
    assert (second.status, second.letter) == (PredictionStatus.UNCERTAIN, None)
    assert (third.status, third.letter) == (PredictionStatus.NO_HAND, None)


# ----- construction and housekeeping -----
def test_rejects_bad_threshold_and_window():
    detector = FakeDetector([])
    for bad_threshold in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            SignPredictor(ScriptedModel([]), make_metadata(), detector, confidence_threshold=bad_threshold)
    with pytest.raises(ValueError):
        SignPredictor(ScriptedModel([]), make_metadata(), detector, smoothing_window=0)


def test_rejects_label_mismatch():
    with pytest.raises(ModelError, match="label"):
        SignPredictor(ScriptedModel([]), make_metadata(labels=["A", "B", "Z"]), FakeDetector([]))


def test_rejects_mirror_setting_mismatch():
    with pytest.raises(ModelError, match="mirror"):
        SignPredictor(
            ScriptedModel([]),
            make_metadata(mirror_camera=not config.MIRROR_CAMERA),
            FakeDetector([]),
        )


def test_missing_mirror_key_only_warns():
    metadata = make_metadata()
    del metadata["mirror_camera"]
    predictor = SignPredictor(ScriptedModel([]), metadata, FakeDetector([]))
    assert predictor.labels == LABELS


def test_close_and_context_manager_close_the_detector():
    predictor, detector = make_predictor([])
    predictor.close()
    assert detector.closed

    predictor, detector = make_predictor([])
    with predictor:
        assert not detector.closed
    assert detector.closed


@pytest.mark.parametrize("mirror", [True, False])
def test_prepare_frame_mirrors_only_when_configured(monkeypatch, mirror):
    monkeypatch.setattr(config, "MIRROR_CAMERA", mirror)
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[:, 0, :] = 255  # bright left edge

    prepared = prepare_frame(frame)
    if mirror:
        assert prepared[:, -1, :].all() and not prepared[:, 0, :].any()
    else:
        assert np.array_equal(prepared, frame)


def test_blank_frame_with_the_real_detector_reports_no_hand():
    with SignPredictor(ScriptedModel([]), make_metadata(), HandDetector()) as predictor:
        assert predictor.process_frame(BLANK_FRAME).status is PredictionStatus.NO_HAND


# ----- preprocessing consistency with a real classifier -----
def make_shapes():
    rng = np.random.default_rng(1)
    return {letter: rng.random((config.NUM_LANDMARKS, config.COORDS_PER_LANDMARK)) for letter in LABELS}


def train_real_model(shapes):
    """Train on features built with landmarks_to_features, exactly like the real pipeline."""
    rng = np.random.default_rng(2)
    features, labels = [], []
    for letter, shape in shapes.items():
        for _ in range(60):
            noisy = shape + rng.normal(0.0, 0.003, shape.shape)
            features.append(landmarks_to_features(noisy))
            labels.append(letter)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return model.fit(np.array(features), np.array(labels))


@pytest.mark.parametrize("letter", LABELS)
def test_real_model_recognizes_moved_and_resized_hands(letter):
    shapes = make_shapes()
    model = train_real_model(shapes)

    # The same hand shape, but bigger and in a different place in the frame.
    # Normalization should remove both differences.
    rng = np.random.default_rng(3)
    moved = [shapes[letter] * 1.7 + np.array([0.2, -0.1, 0.05]) + rng.normal(0, 0.003, (21, 3)) for _ in range(5)]

    predictor = SignPredictor(
        model, make_metadata(), FakeDetector(moved), confidence_threshold=0.5, smoothing_window=5
    )
    results = run(predictor, 5)
    assert results[-1].status is PredictionStatus.RECOGNIZED
    assert results[-1].letter == letter
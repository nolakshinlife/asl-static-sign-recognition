"""Tests for landmark normalization and feature extraction."""

import numpy as np
import pytest

from asl_recognition import config
from asl_recognition.features import (
    feature_column_names,
    landmarks_to_features,
    normalize_landmarks,
)


def make_landmarks(seed: int = 0) -> np.ndarray:
    """Random landmarks of the correct shape, used only as test input."""
    rng = np.random.default_rng(seed)
    return rng.random((config.NUM_LANDMARKS, config.COORDS_PER_LANDMARK))


def test_features_have_expected_shape_and_type():
    features = landmarks_to_features(make_landmarks())
    assert features.shape == (config.NUM_FEATURES,)
    assert features.dtype == np.float32


def test_wrist_is_at_origin_after_normalization():
    normalized = normalize_landmarks(make_landmarks())
    assert np.allclose(normalized[config.WRIST_INDEX], 0.0)


def test_farthest_landmark_is_one_unit_from_wrist():
    normalized = normalize_landmarks(make_landmarks())
    farthest = np.max(np.linalg.norm(normalized, axis=1))
    assert farthest == pytest.approx(1.0)


def test_features_do_not_change_when_hand_moves():
    landmarks = make_landmarks()
    shifted = landmarks + np.array([0.3, -0.2, 0.1])
    assert np.allclose(
        landmarks_to_features(landmarks), landmarks_to_features(shifted), atol=1e-5
    )


def test_features_do_not_change_when_hand_is_scaled():
    landmarks = make_landmarks()
    scaled = landmarks * 2.5
    assert np.allclose(
        landmarks_to_features(landmarks), landmarks_to_features(scaled), atol=1e-5
    )


def test_wrong_shape_raises_value_error():
    with pytest.raises(ValueError):
        landmarks_to_features(np.zeros((20, 3)))


def test_nan_values_raise_value_error():
    landmarks = make_landmarks()
    landmarks[3, 1] = np.nan
    with pytest.raises(ValueError):
        landmarks_to_features(landmarks)


def test_identical_points_raise_value_error():
    # All landmarks at one point means the hand has no size to divide by.
    landmarks = np.full((config.NUM_LANDMARKS, config.COORDS_PER_LANDMARK), 0.5)
    with pytest.raises(ValueError):
        landmarks_to_features(landmarks)


def test_feature_column_names():
    names = feature_column_names()
    assert len(names) == config.NUM_FEATURES
    assert len(set(names)) == len(names)
    assert names[:3] == ["x0", "y0", "z0"]
    assert names[-3:] == ["x20", "y20", "z20"]
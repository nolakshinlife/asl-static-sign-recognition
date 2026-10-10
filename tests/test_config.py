"""Tests for project settings."""

from asl_recognition import config


def test_feature_count_matches_landmarks():
    assert config.NUM_FEATURES == 63
    assert config.NUM_FEATURES == config.NUM_LANDMARKS * config.COORDS_PER_LANDMARK


def test_labels_are_unique_uppercase_letters():
    labels = config.STATIC_SIGN_LABELS
    assert len(labels) == len(set(labels))
    assert all(len(label) == 1 and label.isupper() for label in labels)


def test_motion_letters_are_excluded():
    # J and Z need hand movement, so they are not static signs.
    assert "J" not in config.STATIC_SIGN_LABELS
    assert "Z" not in config.STATIC_SIGN_LABELS
    assert len(config.STATIC_SIGN_LABELS) == 24
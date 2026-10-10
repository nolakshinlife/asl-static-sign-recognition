"""Tests for the HandDetector class.

These tests do not need a webcam. They use blank or invalid arrays.
"""

import numpy as np
import pytest

from asl_recognition.hand_detector import HandDetector


def test_invalid_confidence_raises_value_error():
    with pytest.raises(ValueError):
        HandDetector(min_detection_confidence=1.5)
    with pytest.raises(ValueError):
        HandDetector(min_tracking_confidence=-0.1)


def test_invalid_max_hands_raises_value_error():
    with pytest.raises(ValueError):
        HandDetector(max_num_hands=0)


def test_detect_rejects_non_colour_image():
    with HandDetector() as detector:
        with pytest.raises(ValueError):
            detector.detect(np.zeros((10, 10), dtype=np.uint8))


def test_detect_rejects_wrong_dtype():
    with HandDetector() as detector:
        with pytest.raises(ValueError):
            detector.detect(np.zeros((10, 10, 3), dtype=np.float32))


def test_blank_frame_has_no_hand():
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with HandDetector() as detector:
        assert detector.detect(blank_frame) is None


def test_detect_after_close_raises_runtime_error():
    detector = HandDetector()
    detector.close()
    with pytest.raises(RuntimeError):
        detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
"""Turn raw hand landmarks into features for the classifier.

Raw MediaPipe coordinates depend on where the hand is in the frame and how
close it is to the camera. The same letter signed in two places would give
very different numbers. Normalizing removes those differences so the
classifier only learns the hand shape.
"""

from typing import List

import numpy as np

from asl_recognition import config

# If the hand "size" is smaller than this, the landmarks are effectively the
# same point and dividing by it would give meaningless values.
_MIN_HAND_SIZE = 1e-6


def _validate_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Return landmarks as a float64 array of shape (21, 3), or raise ValueError."""
    try:
        array = np.asarray(landmarks, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Landmarks must be numeric: {error}") from error

    expected_shape = (config.NUM_LANDMARKS, config.COORDS_PER_LANDMARK)
    if array.shape != expected_shape:
        raise ValueError(
            f"Landmarks must have shape {expected_shape}, got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Landmarks contain NaN or infinite values.")
    return array


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Make landmarks independent of hand position and hand size.

    Step 1: subtract the wrist position, so the wrist becomes (0, 0, 0).
            This removes the effect of where the hand is in the frame.
    Step 2: divide by the largest distance from the wrist to any landmark.
            This removes the effect of distance from the camera.

    Returns:
        Array of shape (21, 3). The farthest landmark is exactly 1 away from
        the wrist.
    """
    points = _validate_landmarks(landmarks)

    centered = points - points[config.WRIST_INDEX]
    hand_size = float(np.max(np.linalg.norm(centered, axis=1)))
    if hand_size < _MIN_HAND_SIZE:
        raise ValueError("Hand size is zero, so landmarks cannot be normalized.")

    return centered / hand_size


def landmarks_to_features(landmarks: np.ndarray) -> np.ndarray:
    """Convert (21, 3) landmarks into a flat float32 vector of 63 numbers.

    This is the exact input format used for training and for live prediction.
    Using one function for both keeps them consistent.
    """
    normalized = normalize_landmarks(landmarks)
    return normalized.flatten().astype(np.float32)


def feature_column_names() -> List[str]:
    """Column names for the 63 features: x0, y0, z0, x1, y1, z1, ... x20, y20, z20.

    The order matches landmarks_to_features(), so these names can be used as
    CSV headers when the dataset is saved.
    """
    return [
        f"{axis}{index}"
        for index in range(config.NUM_LANDMARKS)
        for axis in ("x", "y", "z")
    ]
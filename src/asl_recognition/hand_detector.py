"""Hand detection and landmark extraction using MediaPipe Hands."""

import logging
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from asl_recognition import config

logger = logging.getLogger(__name__)

_mp_hands = mp.solutions.hands
# Pairs of landmark indices that should be joined by a line when drawing.
HAND_CONNECTIONS = _mp_hands.HAND_CONNECTIONS


def _validate_frame(frame_bgr: np.ndarray) -> None:
    """Raise ValueError if the frame is not a normal 8-bit colour image."""
    if not isinstance(frame_bgr, np.ndarray):
        raise ValueError("Frame must be a numpy array.")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(
            f"Frame must have shape (height, width, 3), got {frame_bgr.shape}."
        )
    if frame_bgr.size == 0:
        raise ValueError("Frame is empty.")
    if frame_bgr.dtype != np.uint8:
        raise ValueError(f"Frame must be uint8, got {frame_bgr.dtype}.")


class HandDetector:
    """Finds one hand in a frame and returns its 21 landmarks.

    Use it as a context manager so the MediaPipe resources are released:

        with HandDetector() as detector:
            landmarks = detector.detect(frame)
    """

    def __init__(
        self,
        max_num_hands: int = config.MAX_NUM_HANDS,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
        static_image_mode: bool = False,
    ) -> None:
        """Create the detector.

        Args:
            max_num_hands: Maximum number of hands MediaPipe looks for.
            min_detection_confidence: Value from 0 to 1 for the first detection.
            min_tracking_confidence: Value from 0 to 1 for tracking between frames.
            static_image_mode: True for unrelated photos, False for video. With
                video, MediaPipe tracks the hand between frames, which is faster.
        """
        # Check the settings before creating MediaPipe objects, so a bad value
        # fails immediately with a clear message.
        if max_num_hands < 1:
            raise ValueError(f"max_num_hands must be at least 1, got {max_num_hands}.")
        for name, value in (
            ("min_detection_confidence", min_detection_confidence),
            ("min_tracking_confidence", min_tracking_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}.")

        self._hands = _mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._closed = False
        logger.debug("HandDetector created.")

    def detect(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Detect a hand in one BGR frame (the format OpenCV uses).

        Returns:
            A float32 array of shape (21, 3) holding x, y, z for each landmark,
            or None if no hand was found. x and y are fractions of the image
            width and height (0 to 1). z is relative depth, with the wrist as
            reference. If several hands are found, only the first is returned.
        """
        if self._closed:
            raise RuntimeError("HandDetector is closed and cannot process frames.")
        _validate_frame(frame_bgr)

        # OpenCV frames are BGR but MediaPipe expects RGB.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]
        return np.array(
            [[point.x, point.y, point.z] for point in hand.landmark],
            dtype=np.float32,
        )

    def close(self) -> None:
        """Release MediaPipe resources. Safe to call more than once."""
        if not self._closed:
            self._hands.close()
            self._closed = True
            logger.debug("HandDetector closed.")

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def draw_landmarks(
    frame_bgr: np.ndarray,
    landmarks: np.ndarray,
    line_color: Tuple[int, int, int] = (0, 255, 0),
    point_color: Tuple[int, int, int] = (0, 0, 255),
) -> None:
    """Draw the hand skeleton on the frame (modifies the frame in place).

    Landmarks hold values between 0 and 1, so they are multiplied by the frame
    width and height to get pixel positions.
    """
    height, width = frame_bgr.shape[:2]
    points = [(int(x * width), int(y * height)) for x, y, _ in landmarks]

    for start_index, end_index in HAND_CONNECTIONS:
        cv2.line(frame_bgr, points[start_index], points[end_index], line_color, 2)
    for point in points:
        cv2.circle(frame_bgr, point, 4, point_color, -1)
        
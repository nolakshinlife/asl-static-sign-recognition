"""Live sign prediction: camera frame -> hand landmarks -> features -> letter.

Everything here reuses the same pieces as training and evaluation:
  * hand_detector.HandDetector       finds the 21 landmarks
  * features.landmarks_to_features   builds the same 63 normalized features as the dataset
  * model.predict_with_confidence    picks the letter and its confidence
  * smoothing.PredictionSmoother     steadies the answer across frames

IMPORTANT: this recognizes only the static handshapes the model was trained
on (listed in model_metadata.json). It does not translate ASL, and the model
has no "none of these" class, so an untrained handshape can still be
classified as one of the known letters.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import cv2
import numpy as np
from sklearn.base import BaseEstimator

from asl_recognition import config
from asl_recognition.features import landmarks_to_features
from asl_recognition.hand_detector import HandDetector
from asl_recognition.model import ModelError, load_model, predict_with_confidence
from asl_recognition.smoothing import PredictionSmoother

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class PredictionStatus(Enum):
    """What the current frame (or recent frames) amount to."""

    NO_HAND = "no_hand"          # no hand in view
    UNCERTAIN = "uncertain"      # a hand is visible but the model is not confident
    UNSTABLE = "unstable"        # confident frames, but they do not agree yet
    RECOGNIZED = "recognized"    # a letter we are confident about


STATUS_MESSAGES = {
    PredictionStatus.NO_HAND: "No hand detected. Show one hand to the camera.",
    PredictionStatus.UNCERTAIN: "Not sure. Hold a clear handshape in view.",
    PredictionStatus.UNSTABLE: "Hold the sign steady.",
    PredictionStatus.RECOGNIZED: "Recognized.",
}


class LandmarkDetector(Protocol):
    """What the predictor needs from a hand detector (HandDetector fits this).

    Describing it this way lets the tests use a simple fake detector.
    """

    def detect(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return a (21, 3) landmark array, or None if no hand is found."""

    def close(self) -> None:
        """Release the detector's resources."""


@dataclass(frozen=True, eq=False)
class FramePrediction:
    """Raw result for one frame, before the threshold and smoothing."""

    hand_detected: bool
    letter: Optional[str]                 # the model's top letter (None if no hand)
    confidence: float                     # probability of that letter (0.0 if no hand)
    landmarks: Optional[np.ndarray]       # (21, 3) array for drawing, or None


@dataclass(frozen=True, eq=False)
class PredictionResult:
    """What the interface should show for this frame.

    Attributes:
        status: See PredictionStatus.
        letter: The letter to display. Only set when status is RECOGNIZED.
        confidence: For RECOGNIZED, the average confidence behind the letter;
            otherwise the raw confidence of the current frame.
        hand_detected: Whether the CURRENT frame contained a hand. A letter can
            still be shown for a few frames after the hand disappears, because
            of smoothing.
        landmarks: Landmarks of the current frame for drawing, or None.
        raw_letter: The model's top letter for the current frame, before the
            confidence threshold and smoothing (useful for debugging).
        raw_confidence: Confidence of raw_letter.
    """

    status: PredictionStatus
    letter: Optional[str]
    confidence: float
    hand_detected: bool
    landmarks: Optional[np.ndarray]
    raw_letter: Optional[str]
    raw_confidence: float

    @property
    def message(self) -> str:
        """A short sentence describing the status, ready to show to the user."""
        return STATUS_MESSAGES[self.status]


def prepare_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Mirror the frame if config.MIRROR_CAMERA is on.

    collect_data.py flips frames in exactly this way before detecting the hand.
    A mirrored hand has different coordinates, so live frames must be treated
    the same way as the training data. Call this on every raw camera frame
    BEFORE passing it to the predictor, and draw landmarks on the returned frame.
    """
    return cv2.flip(frame_bgr, 1) if config.MIRROR_CAMERA else frame_bgr


class SignPredictor:
    """Turns camera frames into recognized letters.

    The predictor owns the detector it is given and closes it in close().
    It can be used in a "with" block.
    """

    def __init__(
        self,
        model: BaseEstimator,
        metadata: Dict[str, Any],
        detector: LandmarkDetector,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        smoothing_window: int = config.SMOOTHING_WINDOW,
    ) -> None:
        """Create a predictor from an already loaded model.

        Most code should use SignPredictor.from_files() instead.

        Raises:
            ValueError: Bad confidence threshold or smoothing window.
            ModelError: The metadata does not match the model, or the model
                was trained with a different mirror setting.
        """
        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be greater than 0 and at most 1, got {confidence_threshold}."
            )

        labels = [str(label) for label in metadata.get("labels", [])]
        model_classes = [str(label) for label in getattr(model, "classes_", [])]
        if not labels or labels != model_classes:
            raise ModelError(
                "The label list in the metadata does not match the model's classes. "
                "Re-run scripts/train_model.py."
            )

        trained_mirror = metadata.get("mirror_camera")
        if trained_mirror is None:
            logger.warning("Model metadata does not say whether frames were mirrored during training.")
        elif bool(trained_mirror) != config.MIRROR_CAMERA:
            raise ModelError(
                f"The model was trained with mirror_camera={bool(trained_mirror)} but "
                f"config.MIRROR_CAMERA is {config.MIRROR_CAMERA}. Predictions would be wrong. "
                "Change the setting back or collect data and train again."
            )

        self._model = model
        self._detector = detector
        self._smoother = PredictionSmoother(window_size=smoothing_window)
        self.labels: List[str] = labels
        self.model_name = str(metadata.get("model_name", "unknown"))
        self.confidence_threshold = confidence_threshold

    @classmethod
    def from_files(
        cls,
        model_path: PathLike = config.MODEL_FILE,
        metadata_path: PathLike = config.MODEL_METADATA_FILE,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        smoothing_window: int = config.SMOOTHING_WINDOW,
        static_image_mode: bool = False,
    ) -> "SignPredictor":
        """Load the saved model and create a predictor with a real HandDetector.

        Args:
            static_image_mode: Use False for live video (the hand is tracked
                between frames) and True for unrelated single photos.

        Raises:
            ModelError: Model files are missing, unreadable or inconsistent.
            ValueError: Bad confidence threshold or smoothing window.
        """
        model, metadata = load_model(model_path, metadata_path)
        detector = HandDetector(static_image_mode=static_image_mode)
        try:
            return cls(model, metadata, detector, confidence_threshold, smoothing_window)
        except Exception:
            # Do not leave MediaPipe resources open if construction failed.
            detector.close()
            raise

    # ----- single frame -----
    def predict_frame(self, frame_bgr: np.ndarray) -> FramePrediction:
        """Detect the hand and classify it, with no threshold and no smoothing.

        The frame must already be mirrored like the training data
        (see prepare_frame).

        Raises:
            ValueError: If the frame is not a normal 8-bit colour image.
        """
        landmarks = self._detector.detect(frame_bgr)
        if landmarks is None:
            return FramePrediction(hand_detected=False, letter=None, confidence=0.0, landmarks=None)

        try:
            features = landmarks_to_features(landmarks)
        except ValueError as error:
            # Rare: the landmarks collapsed to one point, so treat it as no hand.
            logger.debug("Ignoring frame, feature extraction failed: %s", error)
            return FramePrediction(hand_detected=False, letter=None, confidence=0.0, landmarks=None)

        letters, confidences = predict_with_confidence(self._model, features)
        return FramePrediction(
            hand_detected=True,
            letter=str(letters[0]),
            confidence=float(confidences[0]),
            landmarks=landmarks,
        )

    # ----- single photo -----
    def predict_single(self, frame_bgr: np.ndarray) -> PredictionResult:
        """Classify one unrelated photo. No smoothing is used.

        The letter is reported only if the confidence reaches the threshold.
        """
        frame = self.predict_frame(frame_bgr)
        if not frame.hand_detected:
            status = PredictionStatus.NO_HAND
        elif frame.confidence < self.confidence_threshold:
            status = PredictionStatus.UNCERTAIN
        else:
            status = PredictionStatus.RECOGNIZED

        return PredictionResult(
            status=status,
            letter=frame.letter if status is PredictionStatus.RECOGNIZED else None,
            confidence=frame.confidence,
            hand_detected=frame.hand_detected,
            landmarks=frame.landmarks,
            raw_letter=frame.letter,
            raw_confidence=frame.confidence,
        )

    # ----- live video -----
    def process_frame(self, frame_bgr: np.ndarray) -> PredictionResult:
        """Classify one frame of a live video stream, with smoothing.

        Call this once per frame, in order. Frames without a hand or with low
        confidence count as "no vote", so the shown letter disappears after the
        hand is gone or the model stays unsure for a few frames.
        """
        frame = self.predict_frame(frame_bgr)
        is_usable = frame.hand_detected and frame.confidence >= self.confidence_threshold

        smoothed = self._smoother.update(
            frame.letter if is_usable else None,
            frame.confidence if is_usable else 0.0,
        )

        if smoothed.letter is not None:
            status = PredictionStatus.RECOGNIZED
        elif not frame.hand_detected:
            status = PredictionStatus.NO_HAND
        elif not is_usable:
            status = PredictionStatus.UNCERTAIN
        else:
            status = PredictionStatus.UNSTABLE

        return PredictionResult(
            status=status,
            letter=smoothed.letter,
            confidence=smoothed.confidence if smoothed.letter is not None else frame.confidence,
            hand_detected=frame.hand_detected,
            landmarks=frame.landmarks,
            raw_letter=frame.letter,
            raw_confidence=frame.confidence,
        )

    # ----- housekeeping -----
    def reset(self) -> None:
        """Forget the smoothing history (for example when the camera restarts)."""
        self._smoother.reset()

    def close(self) -> None:
        """Release the hand detector. Safe to call more than once."""
        self._detector.close()

    def __enter__(self) -> "SignPredictor":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
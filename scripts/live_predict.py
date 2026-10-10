"""Try live prediction in an OpenCV window (no Streamlit needed).

Run from the project root (Windows Command Prompt):
    python scripts\\live_predict.py

Press q in the window (or close it) to quit. The window shows the recognized
letter and confidence, or a short message explaining why there is no letter.

This only recognizes the static handshapes the model was trained on. It does
not translate ASL.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import cv2

# Lets this script run without installing the package first.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asl_recognition import config  # noqa: E402
from asl_recognition.camera import open_camera  # noqa: E402
from asl_recognition.hand_detector import draw_landmarks  # noqa: E402
from asl_recognition.logging_setup import setup_logging  # noqa: E402
from asl_recognition.model import ModelError  # noqa: E402
from asl_recognition.predictor import (  # noqa: E402
    PredictionResult,
    PredictionStatus,
    SignPredictor,
    prepare_frame,
)

logger = logging.getLogger("live_predict")

WINDOW_NAME = "ASL live prediction (press q to quit)"
GREEN = (0, 200, 0)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

STATUS_COLORS = {
    PredictionStatus.RECOGNIZED: GREEN,
    PredictionStatus.UNSTABLE: ORANGE,
    PredictionStatus.UNCERTAIN: ORANGE,
    PredictionStatus.NO_HAND: RED,
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(description="Live ASL static sign prediction.")
    parser.add_argument("--model-path", type=Path, default=config.MODEL_FILE, help="Trained model file.")
    parser.add_argument(
        "--metadata-path", type=Path, default=config.MODEL_METADATA_FILE, help="Model metadata file."
    )
    parser.add_argument(
        "--camera-index", type=int, default=config.DEFAULT_CAMERA_INDEX, help="Which camera to use."
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help="Minimum confidence (0 to 1) for a frame to count as a vote.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=config.SMOOTHING_WINDOW,
        help="Number of recent frames used for the majority vote.",
    )
    return parser.parse_args(argv)


def draw_overlay(frame, result: PredictionResult, labels: List[str]) -> None:
    """Draw the letter or status message and a reminder of what the model knows."""
    color = STATUS_COLORS[result.status]
    if result.status is PredictionStatus.RECOGNIZED:
        cv2.putText(
            frame, f"{result.letter}  {result.confidence:.0%}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4,
        )
    else:
        cv2.putText(frame, result.message, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Show the raw per-frame guess too, which helps when judging flicker.
    if result.raw_letter is not None:
        cv2.putText(
            frame, f"this frame: {result.raw_letter} {result.raw_confidence:.0%}",
            (10, 95 if result.status is PredictionStatus.RECOGNIZED else 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1,
        )

    height = frame.shape[0]
    cv2.putText(
        frame, f"Model knows only: {' '.join(labels)} (static handshapes)",
        (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1,
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Run the live prediction loop. Returns 0 on success and 1 on error."""
    args = parse_args(argv)
    setup_logging()

    try:
        predictor = SignPredictor.from_files(
            args.model_path,
            args.metadata_path,
            confidence_threshold=args.confidence_threshold,
            smoothing_window=args.smoothing_window,
        )
    except (ModelError, ValueError) as error:
        logger.error("%s", error)
        return 1

    try:
        capture = open_camera(args.camera_index)
    except (RuntimeError, ValueError) as error:
        predictor.close()
        logger.error("%s", error)
        return 1

    logger.info(
        "Live prediction started (model: %s, letters: %s).",
        predictor.model_name,
        " ".join(predictor.labels),
    )

    try:
        while True:
            success, frame = capture.read()
            if not success:
                logger.error("Failed to read a frame from the camera.")
                return 1

            frame = prepare_frame(frame)
            result = predictor.process_frame(frame)
            if result.landmarks is not None:
                draw_landmarks(frame, result.landmarks)
            draw_overlay(frame, result, predictor.labels)
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            # Closing the window with the X button should also quit.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        capture.release()
        cv2.destroyAllWindows()
        predictor.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
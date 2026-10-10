"""Manual check that the webcam and hand detection work.

Run from the project root:
    python scripts/check_webcam.py

A window opens with your webcam image. When a hand is detected, its landmarks
are drawn on it. Press "q" to quit.

Important for later milestones: by default the image is mirrored like a
selfie view. Data collection and live prediction must use the same setting,
because a mirrored hand has different coordinates.
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
from asl_recognition.features import landmarks_to_features  # noqa: E402
from asl_recognition.hand_detector import HandDetector, draw_landmarks  # noqa: E402
from asl_recognition.logging_setup import setup_logging  # noqa: E402

logger = logging.getLogger("check_webcam")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(description="Check webcam and hand detection.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=config.DEFAULT_CAMERA_INDEX,
        help="Which camera to use (0 is usually the built-in webcam).",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not flip the image horizontally.",
    )
    return parser.parse_args(argv)


def open_camera(camera_index: int) -> cv2.VideoCapture:
    """Open the camera or raise RuntimeError with a helpful message."""
    # On Windows the DirectShow backend usually opens faster and more
    # reliably than the default one.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    capture = cv2.VideoCapture(camera_index, backend)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open camera {camera_index}. Check that it is connected, "
            "not used by another app, and that camera access is allowed in "
            "Windows privacy settings."
        )
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return capture


def main(argv: Optional[List[str]] = None) -> int:
    """Run the webcam loop. Returns 0 on success and 1 on error."""
    args = parse_args(argv)
    setup_logging()

    if args.camera_index < 0:
        logger.error("Camera index must be 0 or higher, got %d.", args.camera_index)
        return 1

    try:
        capture = open_camera(args.camera_index)
    except RuntimeError as error:
        logger.error("%s", error)
        return 1

    window_name = "ASL hand check (press q to quit)"
    logger.info("Camera %d opened. Press q in the window to quit.", args.camera_index)

    try:
        with HandDetector() as detector:
            while True:
                success, frame = capture.read()
                if not success:
                    logger.error("Failed to read a frame from the camera.")
                    return 1

                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)

                landmarks = detector.detect(frame)
                if landmarks is None:
                    status = "No hand detected"
                else:
                    draw_landmarks(frame, landmarks)
                    try:
                        features = landmarks_to_features(landmarks)
                        status = f"Hand detected - {features.shape[0]} features"
                    except ValueError as error:
                        # Rare: landmarks collapsed to a single point.
                        status = "Hand detected - could not build features"
                        logger.warning("Feature extraction failed: %s", error)

                cv2.putText(
                    frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2,
                )
                cv2.imshow(window_name, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        # Always release the camera, even if something goes wrong above.
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
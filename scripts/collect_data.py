"""Collect labelled hand-landmark samples with the webcam.

Run from the project root:
    python scripts/collect_data.py                    (default letters)
    python scripts/collect_data.py --labels A B L     (choose your own)
    python scripts/collect_data.py --summary          (show counts and exit)

Keys in the camera window:
    SPACE  start recording the current letter / stop and save
    N / P  next / previous letter (not while recording)
    U      undo: delete the last session saved in this run
    Q      quit (a recording in progress is saved)

Tips for a dataset that works on new data:
  * Record each letter in several separate sessions (at least 3, more is
    better). Between sessions change the distance, angle, lighting or
    background. During a session, move your hand slightly.
  * Always use the same hand. The image is mirrored, and a left hand looks
    like a mirrored right hand to the classifier.
  * Check each handshape against an ASL reference chart before recording.
"""

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

# Lets this script run without installing the package first.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asl_recognition import config  # noqa: E402
from asl_recognition.camera import open_camera  # noqa: E402
from asl_recognition.dataset import (  # noqa: E402
    DatasetError,
    SessionRecorder,
    append_samples,
    dataset_exists,
    find_dataset_problems,
    load_dataset,
    load_summary,
    normalize_label_list,
    remove_session,
    summarize_dataset,
    validate_sample_count,
)
from asl_recognition.features import landmarks_to_features  # noqa: E402
from asl_recognition.hand_detector import HandDetector, draw_landmarks  # noqa: E402
from asl_recognition.logging_setup import setup_logging  # noqa: E402

logger = logging.getLogger("collect_data")

WINDOW_NAME = "ASL data collection"
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(description="Collect ASL hand-landmark samples.")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(config.DEFAULT_LABELS),
        help="Letters to collect, for example: --labels A B C",
    )
    parser.add_argument(
        "--samples-per-session",
        type=int,
        default=config.DEFAULT_SAMPLES_PER_SESSION,
        help="Samples recorded each time you press SPACE.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=config.DEFAULT_CAMERA_INDEX,
        help="Which camera to use (0 is usually the built-in webcam).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.RAW_DATASET_FILE,
        help="CSV file to append samples to.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print sample counts for the dataset and exit (no camera needed).",
    )
    return parser.parse_args(argv)


def print_summary(csv_path: Path, labels: List[str]) -> int:
    """Print how many samples and sessions each label has. Returns an exit code."""
    if not dataset_exists(csv_path):
        print(f"No dataset found at {csv_path} yet.")
        return 0

    try:
        data = load_dataset(csv_path, allow_empty=True)
    except DatasetError as error:
        logger.error("%s", error)
        return 1

    # Also show labels that are in the file but were not asked for.
    extra_labels = [label for label in sorted(data["label"].unique()) if label not in labels]
    summary = summarize_dataset(data, labels + extra_labels)
    print(summary.to_string(index=False))

    problems = find_dataset_problems(summary)
    if problems:
        print("\nNot ready for training yet:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("\nAll labels meet the minimum sample and session counts for training.")
    return 0


class DataCollector:
    """Holds the state of the collection window and handles keys and drawing."""

    def __init__(self, labels: List[str], samples_per_session: int, csv_path: Path) -> None:
        self.labels = labels
        self.samples_per_session = samples_per_session
        self.csv_path = csv_path
        self.label_index = 0
        self.recorder: Optional[SessionRecorder] = None
        # Sessions saved during this run, newest last. Undo uses this list.
        self.saved_sessions: List[str] = []
        self.message = "Press SPACE to start recording."
        self.stats: Dict[str, Tuple[int, int]] = {}
        self.refresh_stats()

    @property
    def current_label(self) -> str:
        return self.labels[self.label_index]

    def refresh_stats(self) -> None:
        """Reload (samples, sessions) for each label from the CSV file."""
        summary = load_summary(self.csv_path, self.labels)
        self.stats = {
            row.label: (row.samples, row.sessions) for row in summary.itertuples()
        }

    # ----- actions -----
    def start_recording(self, now: float) -> None:
        self.recorder = SessionRecorder(
            self.current_label, self.samples_per_session, start_time=now
        )
        self.message = ""

    def finish_recording(self) -> None:
        """Save the current session (if any) and go back to idle."""
        recorder = self.recorder
        self.recorder = None
        if recorder is None:
            return
        if not recorder.samples:
            self.message = "Nothing saved: no hand was captured."
            return

        try:
            saved = append_samples(
                self.csv_path, recorder.session_id, recorder.label, recorder.samples
            )
            self.saved_sessions.append(recorder.session_id)
            self.refresh_stats()
        except DatasetError as error:
            logger.error("Could not save the session: %s", error)
            self.message = "Could not save samples. See the log."
            return
        self.message = f"Saved {saved} samples for {recorder.label}."

    def undo_last(self) -> None:
        """Delete the most recent session saved in this run."""
        if self.recorder is not None:
            self.message = "Stop recording before undoing."
            return
        if not self.saved_sessions:
            self.message = "Nothing to undo in this run."
            return

        session_id = self.saved_sessions[-1]
        try:
            removed = remove_session(self.csv_path, session_id)
            self.refresh_stats()
        except DatasetError as error:
            logger.error("Could not undo the session: %s", error)
            self.message = "Could not undo. See the log."
            return
        self.saved_sessions.pop()
        self.message = f"Removed last session ({removed} samples)."

    def change_label(self, step: int) -> None:
        if self.recorder is not None:
            self.message = "Stop recording before changing the letter."
            return
        self.label_index = (self.label_index + step) % len(self.labels)
        self.message = f"Now recording letter {self.current_label}."

    def handle_key(self, key: int, now: float) -> bool:
        """React to a key press. Returns True when the user wants to quit."""
        character = chr(key).lower() if 0 <= key < 256 else ""
        if character == "q":
            return True
        if character == " ":
            if self.recorder is None:
                self.start_recording(now)
            else:
                self.finish_recording()
        elif character == "n":
            self.change_label(1)
        elif character == "p":
            self.change_label(-1)
        elif character == "u":
            self.undo_last()
        return False

    # ----- per-frame work -----
    def process_frame(self, frame, detector: HandDetector, now: float) -> None:
        """Detect the hand, store a sample if recording, and draw the overlay."""
        landmarks = detector.detect(frame)
        features = None
        if landmarks is not None:
            draw_landmarks(frame, landmarks)
            try:
                features = landmarks_to_features(landmarks)
            except ValueError as error:
                # Rare: all landmarks collapsed to one point.
                logger.warning("Skipping frame, feature extraction failed: %s", error)

        if self.recorder is not None:
            if features is not None:
                self.recorder.try_add(features, now)
            if self.recorder.is_complete:
                self.finish_recording()

        self.draw_overlay(frame, features is not None, now)

    def draw_overlay(self, frame, hand_found: bool, now: float) -> None:
        samples, sessions = self.stats.get(self.current_label, (0, 0))
        lines = [
            (
                f"Letter {self.current_label}  (saved: {samples} samples, {sessions} sessions)",
                GREEN,
            )
        ]

        if self.recorder is None:
            lines.append(("SPACE record | N/P letter | U undo | Q quit", GREEN))
        else:
            remaining = self.recorder.countdown_remaining(now)
            if remaining > 0:
                lines.append((f"Get ready... {math.ceil(remaining)}", GREEN))
            else:
                count = len(self.recorder.samples)
                lines.append(
                    (f"Recording {count}/{self.recorder.target_count}", GREEN)
                )

        lines.append(
            ("Hand detected", GREEN) if hand_found else ("No hand detected", RED)
        )
        if self.message:
            lines.append((self.message, GREEN))

        y = 28
        for text, color in lines:
            cv2.putText(
                frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
            y += 26

    def run(self, capture: cv2.VideoCapture, detector: HandDetector) -> int:
        """Main loop. Returns 0 on success and 1 on error."""
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    logger.error("Failed to read a frame from the camera.")
                    return 1

                if config.MIRROR_CAMERA:
                    frame = cv2.flip(frame, 1)

                now = time.monotonic()
                self.process_frame(frame, detector, now)
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if self.handle_key(key, now):
                    break
                # Closing the window with the X button should also quit.
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
        except KeyboardInterrupt:
            logger.info("Stopped by user.")
        finally:
            # Save whatever was captured, then always release the camera.
            self.finish_recording()
            capture.release()
            cv2.destroyAllWindows()
        return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point. Returns 0 on success and 1 on error."""
    args = parse_args(argv)
    setup_logging()

    try:
        labels = normalize_label_list(args.labels)
        samples_per_session = validate_sample_count(args.samples_per_session)
    except ValueError as error:
        logger.error("%s", error)
        return 1

    if args.summary:
        return print_summary(args.output, labels)

    try:
        collector = DataCollector(labels, samples_per_session, args.output)
    except DatasetError as error:
        logger.error("%s", error)
        return 1

    try:
        capture = open_camera(args.camera_index)
    except (RuntimeError, ValueError) as error:
        logger.error("%s", error)
        return 1

    logger.info(
        "Collecting letters %s into %s. Press SPACE to record.", " ".join(labels), args.output
    )
    with HandDetector() as detector:
        return collector.run(capture, detector)


if __name__ == "__main__":
    sys.exit(main())
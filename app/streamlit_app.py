"""Streamlit interface for the ASL static sign recognition project.

Run from the project root (Windows Command Prompt):
    python -m streamlit run app\\streamlit_app.py

This recognizes only the static handshapes the trained model knows (see
models\\model_metadata.json). It does NOT translate ASL.

Webcam notes (please read):
  * "Live camera" opens the webcam with OpenCV on the computer that runs
    Streamlit, so it works when you run the app on your own PC. It would not
    work on a hosted server, because that server has no webcam.
  * Streamlit is not a video player. Frames are pushed to the page one by one,
    so the frame rate is lower and less smooth than scripts\\live_predict.py.
  * Changing any setting restarts the live loop (and the camera) because
    Streamlit re-runs this script on every interaction.
  * Live mode and Photo mode both need the webcam, so only one is active at a
    time. Do not run live_predict.py or collect_data.py at the same time.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import streamlit as st

# Streamlit only adds the app/ folder to the import path, so add src/ ourselves.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = str(PROJECT_ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from asl_recognition import config  # noqa: E402
from asl_recognition.camera import open_camera  # noqa: E402
from asl_recognition.hand_detector import HandDetector, draw_landmarks  # noqa: E402
from asl_recognition.logging_setup import setup_logging  # noqa: E402
from asl_recognition.model import ModelError, load_model  # noqa: E402
from asl_recognition.predictor import (  # noqa: E402
    PredictionResult,
    PredictionStatus,
    SignPredictor,
    prepare_frame,
)

# Must be the first Streamlit command in the script.
st.set_page_config(page_title="ASL Static Sign Recognition")

MODE_LIVE = "Live camera"
MODE_PHOTO = "Photo"


@st.cache_resource(show_spinner=False)
def load_trained_model() -> Tuple[Any, Dict[str, Any]]:
    """Load the model and its metadata once and reuse them on every rerun.

    Streamlit re-runs the whole script after each interaction, so without the
    cache the model file would be read again every time.
    """
    return load_model(config.MODEL_FILE, config.MODEL_METADATA_FILE)


def build_predictor(
    model: Any,
    metadata: Dict[str, Any],
    threshold: float,
    window: int,
    static_image_mode: bool,
) -> SignPredictor:
    """Create a SignPredictor around the already loaded model.

    The predictor owns its HandDetector and closes it in close().
    """
    detector = HandDetector(static_image_mode=static_image_mode)
    try:
        return SignPredictor(
            model,
            metadata,
            detector,
            confidence_threshold=threshold,
            smoothing_window=window,
        )
    except Exception:
        # Do not leave MediaPipe resources open if construction failed.
        detector.close()
        raise


def result_message(result: PredictionResult) -> Tuple[str, str]:
    """Turn a prediction into (box style, text) for display."""
    detail = ""
    if result.raw_letter is not None:
        detail = f"\n\nThis frame alone: {result.raw_letter} ({result.raw_confidence:.0%})"

    if result.status is PredictionStatus.RECOGNIZED:
        return "success", f"### Letter: {result.letter}\n\nConfidence: {result.confidence:.0%}{detail}"
    if result.status is PredictionStatus.NO_HAND:
        return "info", result.message
    return "warning", f"{result.message}{detail}"


def render_message(target: Any, kind: str, text: str) -> None:
    """Show text in a coloured box. target can be st or a placeholder/column."""
    if kind == "success":
        target.success(text)
    elif kind == "warning":
        target.warning(text)
    else:
        target.info(text)


def run_live_camera(
    model: Any,
    metadata: Dict[str, Any],
    camera_index: int,
    threshold: float,
    window: int,
) -> None:
    """Show the webcam with live predictions until the user unticks 'Camera on'.

    When the user changes a widget, Streamlit stops this function at its next
    st call. The finally block then releases the camera and the detector.
    """
    try:
        capture = open_camera(camera_index)
    except (RuntimeError, ValueError) as error:
        st.error(str(error))
        st.info("Close other apps that use the webcam (Zoom, Teams, other scripts), then untick and tick the box again.")
        return

    try:
        predictor = build_predictor(model, metadata, threshold, window, static_image_mode=False)
    except (ModelError, ValueError) as error:
        capture.release()
        st.error(str(error))
        return

    video_column, result_column = st.columns([3, 2])
    frame_slot = video_column.empty()
    result_slot = result_column.empty()
    last_message = None

    try:
        while True:
            success, frame = capture.read()
            if not success:
                st.error("Could not read a frame from the camera. Untick and tick 'Camera on' to try again.")
                break

            # Same mirroring as in data collection, then predict with smoothing.
            frame = prepare_frame(frame)
            result = predictor.process_frame(frame)
            if result.landmarks is not None:
                draw_landmarks(frame, result.landmarks)

            # OpenCV frames are BGR, so tell Streamlit. JPEG keeps updates small.
            frame_slot.image(frame, channels="BGR", output_format="JPEG")

            # Only redraw the text box when its content changed.
            message = result_message(result)
            if message != last_message:
                render_message(result_slot, *message)
                last_message = message
    except ValueError as error:
        st.error(f"Prediction stopped: {error}")
    finally:
        capture.release()
        predictor.close()


def run_photo_mode(
    model: Any,
    metadata: Dict[str, Any],
    threshold: float,
    flip_photo: bool,
) -> None:
    """Classify one photo taken with the browser's camera. No smoothing is used."""
    photo = st.camera_input("Take a photo of one hand")
    if photo is None:
        st.info("Show one hand, take a photo, and the prediction appears here.")
        return

    frame = cv2.imdecode(np.frombuffer(photo.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        st.error("Could not read the photo. Please take it again.")
        return
    if flip_photo:
        frame = prepare_frame(frame)

    try:
        # static_image_mode=True because a photo is not part of a video. The
        # smoothing window is 1 because a single photo has nothing to smooth.
        with build_predictor(model, metadata, threshold, 1, static_image_mode=True) as predictor:
            result = predictor.predict_single(frame)
    except (ModelError, ValueError) as error:
        st.error(str(error))
        return

    if result.landmarks is not None:
        draw_landmarks(frame, result.landmarks)
    photo_column, result_column = st.columns([3, 2])
    photo_column.image(frame, channels="BGR", output_format="JPEG")
    render_message(result_column, *result_message(result))


def main() -> None:
    """Build the page."""
    setup_logging()
    st.title("ASL Static Sign Recognition")

    try:
        model, metadata = load_trained_model()
    except ModelError as error:
        st.error(str(error))
        st.info(
            "Train a model first: collect data with scripts\\collect_data.py, "
            "then run scripts\\train_model.py."
        )
        st.stop()

    labels = [str(label) for label in metadata["labels"]]
    st.info(
        f"This app recognizes only the static handshapes its model was trained on: "
        f"**{', '.join(labels)}**. It is not ASL translation: it does not understand words, "
        "sentences, facial expressions or movement. Any other handshape may be reported as "
        "uncertain or, wrongly, as one of these letters."
    )

    # ----- sidebar -----
    st.sidebar.header("Settings")
    mode = st.sidebar.radio("Mode", [MODE_LIVE, MODE_PHOTO])
    threshold = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=1.0,
        value=float(config.CONFIDENCE_THRESHOLD),
        step=0.05,
        help="A frame only counts when the model's confidence is at least this high.",
    )
    window = st.sidebar.slider(
        "Smoothing window (frames)",
        min_value=1,
        max_value=15,
        value=int(config.SMOOTHING_WINDOW),
        help="A letter is shown when it wins a majority of this many recent frames (live mode only).",
    )

    if st.sidebar.button("Reload model files"):
        st.cache_resource.clear()
        st.rerun()

    with st.sidebar.expander("Model details"):
        st.write(f"Model type: {metadata.get('model_name', 'unknown')}")
        st.write(f"Trained at (UTC): {metadata.get('trained_at', 'unknown')}")
        st.write(f"Letters: {', '.join(labels)}")
        training_samples = metadata.get("dataset", {}).get("num_samples")
        if training_samples is not None:
            st.write(f"Samples in the dataset at training time: {training_samples}")

    # ----- main area -----
    if mode == MODE_LIVE:
        camera_index = int(
            st.sidebar.number_input(
                "Camera index",
                min_value=0,
                max_value=10,
                value=int(config.DEFAULT_CAMERA_INDEX),
                step=1,
                help="0 is usually the built-in webcam.",
            )
        )
        camera_on = st.checkbox("Camera on")
        if camera_on:
            # This call keeps running until the user changes a widget.
            run_live_camera(model, metadata, camera_index, threshold, window)
        else:
            st.info(
                "The camera is off. Tick **Camera on** to start. Everything runs on this "
                "computer; nothing is saved or uploaded."
            )
    else:
        flip_photo = st.sidebar.checkbox(
            "Flip photo to match the training data",
            value=True,
            help=(
                "The training data came from a mirrored (selfie-view) camera image. If photo "
                "results look wrong while live mode works, try switching this off."
            ),
        )
        run_photo_mode(model, metadata, threshold, flip_photo)


main()
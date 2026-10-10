"""Webcam helper shared by the scripts and the Streamlit app."""

import sys

import cv2

from asl_recognition import config


def open_camera(camera_index: int = config.DEFAULT_CAMERA_INDEX) -> cv2.VideoCapture:
    """Open the camera or raise a RuntimeError with a helpful message.

    Raises:
        ValueError: If camera_index is negative.
        RuntimeError: If the camera cannot be opened.
    """
    if camera_index < 0:
        raise ValueError(f"Camera index must be 0 or higher, got {camera_index}.")

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
"""Shared test helpers."""

import numpy as np
import pandas as pd
import pytest

from asl_recognition import config
from asl_recognition.dataset import csv_columns


@pytest.fixture
def make_dataset():
    """Return a function that builds a synthetic landmark dataset.

    Each letter gets its own random "centre" in feature space and every sample
    is that centre plus a little noise, so the letters are clearly separable.
    This is made-up data used only to test the code. It says nothing about how
    the real model performs on real hands.
    """

    def _make(labels=("A", "B", "C"), sessions_per_label=5, samples_per_session=30, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for label in labels:
            centre = rng.normal(0.0, 0.5, size=config.NUM_FEATURES)
            centre[:3] = 0.0  # the wrist is the origin after normalization
            for session_number in range(sessions_per_label):
                session_id = f"{label}-session{session_number}"
                for _ in range(samples_per_session):
                    features = centre + rng.normal(0.0, 0.05, size=config.NUM_FEATURES)
                    features[:3] = 0.0
                    rows.append([session_id, label] + features.tolist())
        return pd.DataFrame(rows, columns=csv_columns())

    return _make
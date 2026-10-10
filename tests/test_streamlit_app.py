"""Smoke tests for the Streamlit app using Streamlit's headless test runner.

These tests never open a webcam. They check that the page builds without
errors, shows a helpful message when no model exists, and shows the model's
letters when one does.
"""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from asl_recognition import config
from asl_recognition.model import (
    build_metadata,
    save_model,
    select_rows,
    split_sessions,
    train_final_model,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

APP_FILE = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    """The app caches the loaded model, so start and finish every test clean."""
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def run_app():
    # The first run imports MediaPipe, which can take a few seconds.
    return AppTest.from_file(str(APP_FILE), default_timeout=60).run()


def test_shows_helpful_error_when_model_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODEL_FILE", tmp_path / "missing.joblib")
    monkeypatch.setattr(config, "MODEL_METADATA_FILE", tmp_path / "missing.json")

    app = run_app()

    assert not app.exception
    assert any("train_model.py" in error.value for error in app.error)


def test_page_builds_with_a_valid_model_and_camera_off(monkeypatch, tmp_path, make_dataset):
    # Train a tiny model on synthetic data and save it where the app will look.
    data = make_dataset()
    csv_path = tmp_path / "data.csv"
    data.to_csv(csv_path, index=False, float_format="%.6f")
    split = split_sessions(data)
    features, labels = select_rows(data, split.train + split.validation)
    model = train_final_model(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), features, labels
    )
    metadata = build_metadata(
        model_name="logistic_regression",
        model=model,
        data=data,
        data_path=csv_path,
        split=split,
        validation_scores={},
        seed=config.RANDOM_SEED,
        test_fraction=config.TEST_FRACTION,
        validation_fraction=config.VALIDATION_FRACTION,
    )
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    save_model(model, metadata, model_path, metadata_path)
    monkeypatch.setattr(config, "MODEL_FILE", model_path)
    monkeypatch.setattr(config, "MODEL_METADATA_FILE", metadata_path)

    app = run_app()

    assert not app.exception
    assert len(app.error) == 0
    # The scope notice must list the model's letters.
    assert any("A, B, C" in info.value for info in app.info)
    # The camera must start switched off.
    assert app.checkbox[0].label == "Camera on"
    assert app.checkbox[0].value is Falses
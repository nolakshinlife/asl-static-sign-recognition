"""Tests for the prediction smoother."""

import math

import pytest

from asl_recognition.smoothing import PredictionSmoother


@pytest.mark.parametrize("bad_size", [0, -1, 2.5, True, "7", None])
def test_rejects_invalid_window_size(bad_size):
    with pytest.raises(ValueError):
        PredictionSmoother(window_size=bad_size)


@pytest.mark.parametrize("bad_votes", [0, 4, -1, 1.5, True])
def test_rejects_invalid_min_votes(bad_votes):
    with pytest.raises(ValueError):
        PredictionSmoother(window_size=3, min_votes=bad_votes)


def test_default_min_votes_is_a_strict_majority():
    assert PredictionSmoother(window_size=7).min_votes == 4
    assert PredictionSmoother(window_size=5).min_votes == 3
    assert PredictionSmoother(window_size=4).min_votes == 3
    assert PredictionSmoother(window_size=1).min_votes == 1


def test_letter_becomes_stable_only_after_enough_votes():
    smoother = PredictionSmoother(window_size=7)
    results = [smoother.update("A", 0.9) for _ in range(4)]
    assert [r.letter for r in results] == [None, None, None, "A"]
    assert results[-1].votes == 4


def test_confidence_is_the_mean_of_the_voting_frames():
    smoother = PredictionSmoother(window_size=3)
    assert smoother.update("A", 0.8).letter is None  # 1 vote, needs 2
    second = smoother.update("A", 0.9)
    third = smoother.update("A", 1.0)
    assert second.letter == "A"
    assert second.confidence == pytest.approx(0.85)
    assert third.confidence == pytest.approx(0.9)


def test_none_frames_count_as_no_vote():
    smoother = PredictionSmoother(window_size=3)
    smoother.update("A", 0.9)
    assert smoother.update("A", 0.9).letter == "A"
    assert smoother.update(None).letter == "A"   # [A, A, None]: still 2 votes
    result = smoother.update(None)               # [A, None, None]: 1 vote
    assert result.letter is None
    assert result.confidence == 0.0


def test_old_frames_drop_out_of_the_window():
    smoother = PredictionSmoother(window_size=3)
    for _ in range(3):
        smoother.update("A", 0.9)
    assert smoother.update("B", 0.9).letter == "A"   # [A, A, B]
    assert smoother.update("B", 0.9).letter == "B"   # [A, B, B]


def test_alternating_letters_never_become_stable():
    smoother = PredictionSmoother(window_size=5)  # needs 3 votes
    letters = ["A", "B", "C", "A", "B", "C"]
    results = [smoother.update(letter, 0.9) for letter in letters]
    assert all(result.letter is None for result in results)


def test_tie_goes_to_the_most_recent_letter():
    smoother = PredictionSmoother(window_size=4, min_votes=2)
    results = [smoother.update(letter, 0.9) for letter in ["A", "B", "A", "B"]]
    assert results[-1].letter == "B"


def test_frames_in_window_grows_then_stays_at_window_size():
    smoother = PredictionSmoother(window_size=3)
    sizes = [smoother.update("A", 0.9).frames_in_window for _ in range(4)]
    assert sizes == [1, 2, 3, 3]


def test_reset_clears_history():
    smoother = PredictionSmoother(window_size=3)
    for _ in range(3):
        smoother.update("A", 0.9)
    smoother.reset()
    result = smoother.update("A", 0.9)
    assert result.letter is None
    assert result.frames_in_window == 1


@pytest.mark.parametrize("bad_letter", [5, "", ["A"]])
def test_update_rejects_invalid_letters(bad_letter):
    with pytest.raises(ValueError):
        PredictionSmoother(window_size=3).update(bad_letter, 0.9)


@pytest.mark.parametrize("bad_confidence", [-0.1, 1.5, math.nan, "high", None])
def test_update_rejects_invalid_confidence(bad_confidence):
    with pytest.raises(ValueError):
        PredictionSmoother(window_size=3).update("A", bad_confidence)
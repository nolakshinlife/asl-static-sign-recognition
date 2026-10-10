"""Smoothing of frame-by-frame predictions.

A classifier looking at single video frames flickers: a slightly different
finger position can flip the answer for one frame. Showing every raw answer
would make the letter jump around. Instead we remember the last few frames and
show a letter only when it wins a majority vote among them.

This module only deals with letters and numbers, so it is easy to test.
"""

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

from asl_recognition import config


@dataclass(frozen=True)
class SmoothedPrediction:
    """Result of one smoothing step.

    Attributes:
        letter: The stable letter, or None if no letter has enough votes yet.
        confidence: Average confidence of the frames that voted for the stable
            letter (0.0 when there is no stable letter).
        votes: How many recent frames voted for the most common letter.
        frames_in_window: How many frames are currently remembered.
    """

    letter: Optional[str]
    confidence: float
    votes: int
    frames_in_window: int


class PredictionSmoother:
    """Majority vote over the most recent frames.

    Each frame gives either a letter or None. None means "no usable
    prediction" (no hand, or the model was not confident enough). A None frame
    counts as a vote for nobody, so lost or doubtful frames make the output
    less stable instead of being ignored.

    A letter becomes the stable letter once it has at least min_votes of the
    last window_size frames. By default min_votes is a strict majority, which
    means two different letters can never be stable at the same time.
    """

    def __init__(
        self,
        window_size: int = config.SMOOTHING_WINDOW,
        min_votes: Optional[int] = None,
    ) -> None:
        """Create a smoother.

        Args:
            window_size: How many recent frames to remember (1 or more).
            min_votes: Votes needed for a letter to be stable. Defaults to a
                strict majority: window_size // 2 + 1.

        Raises:
            ValueError: If the numbers are not whole numbers in a sensible range.
        """
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1:
            raise ValueError(f"window_size must be a whole number of at least 1, got {window_size!r}.")

        if min_votes is None:
            min_votes = window_size // 2 + 1
        if (
            isinstance(min_votes, bool)
            or not isinstance(min_votes, int)
            or not 1 <= min_votes <= window_size
        ):
            raise ValueError(
                f"min_votes must be a whole number from 1 to {window_size}, got {min_votes!r}."
            )

        self._window_size = window_size
        self._min_votes = min_votes
        # deque with maxlen drops the oldest frame automatically.
        self._history: Deque[Tuple[Optional[str], float]] = deque(maxlen=window_size)

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def min_votes(self) -> int:
        return self._min_votes

    def reset(self) -> None:
        """Forget all remembered frames (for example when the camera restarts)."""
        self._history.clear()

    def update(self, letter: Optional[str], confidence: float = 0.0) -> SmoothedPrediction:
        """Add one frame and return the smoothed result.

        Args:
            letter: The frame's predicted letter, or None if there is no
                usable prediction for this frame.
            confidence: The model's confidence for that letter (0 to 1).

        Raises:
            ValueError: If letter is not a non-empty string or None, or the
                confidence is not a number from 0 to 1.
        """
        if letter is not None and (not isinstance(letter, str) or not letter):
            raise ValueError(f"letter must be a non-empty string or None, got {letter!r}.")
        try:
            confidence = float(confidence)
        except (TypeError, ValueError) as error:
            raise ValueError(f"confidence must be a number, got {confidence!r}.") from error
        # Written this way so NaN is rejected too (every comparison with NaN is False).
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be between 0 and 1, got {confidence}.")

        self._history.append((letter, confidence))
        return self._vote()

    def _vote(self) -> SmoothedPrediction:
        """Count the votes in the current window."""
        frames = len(self._history)
        counts = Counter(letter for letter, _ in self._history if letter is not None)
        if not counts:
            return SmoothedPrediction(letter=None, confidence=0.0, votes=0, frames_in_window=frames)

        # If two letters have the same number of votes, the one seen most
        # recently wins. This only matters when min_votes is set to half the
        # window or less.
        last_seen: Dict[str, int] = {}
        for index, (letter, _) in enumerate(self._history):
            if letter is not None:
                last_seen[letter] = index
        top_letter = max(counts, key=lambda candidate: (counts[candidate], last_seen[candidate]))
        votes = counts[top_letter]

        if votes < self._min_votes:
            return SmoothedPrediction(letter=None, confidence=0.0, votes=votes, frames_in_window=frames)

        confidences = [conf for letter, conf in self._history if letter == top_letter]
        return SmoothedPrediction(
            letter=top_letter,
            confidence=sum(confidences) / len(confidences),
            votes=votes,
            frames_in_window=frames,
        )
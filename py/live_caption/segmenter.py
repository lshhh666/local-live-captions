from __future__ import annotations

import math
from array import array
from collections import deque
from typing import Iterable

from .config import RuntimeConfig


class EnergyUtteranceSegmenter:
    """Bounded energy-based VAD for the first quality prototype."""

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config
        self._minimum_speech = self._samples(config.minimum_speech_ms / 1_000)
        self._end_silence = self._samples(config.end_silence_ms / 1_000)
        self._maximum_utterance = self._samples(config.maximum_utterance_seconds)
        self._pre_roll_capacity = self._samples(config.pre_roll_ms / 1_000)
        self._pre_roll: deque[float] = deque(maxlen=self._pre_roll_capacity)
        self._utterance = array("f")
        self._silence_samples = 0
        self._speaking = False
        self._last_chunk_is_final = True

    @property
    def buffered_sample_count(self) -> int:
        return len(self._pre_roll) + len(self._utterance)

    @property
    def maximum_buffered_samples(self) -> int:
        return self._maximum_utterance + self._pre_roll_capacity

    @property
    def last_chunk_is_final(self) -> bool:
        """False for a timed partial cut, true when silence ended the utterance."""
        return self._last_chunk_is_final

    def push(self, samples: Iterable[float]) -> array | None:
        values = array("f", samples)
        if not values:
            return None

        rms = math.sqrt(sum(sample * sample for sample in values) / len(values))
        if not self._speaking:
            self._pre_roll.extend(values)
            if rms < self._config.start_threshold:
                return None
            self._speaking = True
            self._utterance.extend(self._pre_roll)
            self._pre_roll.clear()
        else:
            self._utterance.extend(values)

        self._silence_samples = (
            self._silence_samples + len(values)
            if rms < self._config.continue_threshold
            else 0
        )

        if len(self._utterance) >= self._maximum_utterance:
            return self._finish(force=True)
        if self._silence_samples >= self._end_silence:
            return self._finish(force=False)
        return None

    def clear(self) -> None:
        self._pre_roll.clear()
        self._utterance = array("f")
        self._silence_samples = 0
        self._speaking = False

    def _finish(self, force: bool) -> array | None:
        self._last_chunk_is_final = not force
        speech_samples = max(0, len(self._utterance) - self._silence_samples)
        overlap = (
            array("f", self._utterance[-self._pre_roll_capacity :])
            if force and self._utterance
            else array("f")
        )
        result = (
            array("f", self._utterance)
            if force or speech_samples >= self._minimum_speech
            else None
        )
        self.clear()
        # Forced cuts can land in the middle of a word. Continue the active
        # utterance with a short overlap; TranscriptStabilizer de-dupes it.
        if overlap:
            self._speaking = True
            self._utterance.extend(overlap)
        return result

    def _samples(self, seconds: float) -> int:
        return math.ceil(self._config.sample_rate * seconds)

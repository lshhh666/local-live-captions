from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    sample_rate: int = 16_000
    audio_buffer_seconds: int = 30
    context_sentences: int = 5
    context_minutes: int = 10
    maximum_sentence_words: int = 45
    recognition_queue_size: int = 4
    translation_queue_size: int = 8
    minimum_speech_ms: int = 250
    end_silence_ms: int = 550
    maximum_utterance_seconds: float = 3.2
    pre_roll_ms: int = 400
    start_threshold: float = 0.006
    continue_threshold: float = 0.004


DEFAULT_CONFIG = RuntimeConfig()

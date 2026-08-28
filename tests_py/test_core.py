from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from live_caption.buffers import BoundedCaptionHistory, FixedAudioBuffer
from live_caption.config import RuntimeConfig
from live_caption.models import CaptionSegment
from live_caption.recognizer import DEFAULT_BEAM_SIZE, clean_transcript
from live_caption.segmenter import EnergyUtteranceSegmenter
from live_caption.stabilizer import TranscriptStabilizer


class FixedAudioBufferTests(unittest.TestCase):
    def test_overwrites_old_samples_without_growing(self) -> None:
        buffer = FixedAudioBuffer(sample_rate=10, duration_seconds=1)
        buffer.write(range(25))
        self.assertEqual(10, buffer.count)
        self.assertEqual(list(range(15, 25)), list(buffer.snapshot()))


class CaptionHistoryTests(unittest.TestCase):
    def test_limits_age_and_item_count(self) -> None:
        history = BoundedCaptionHistory(3, timedelta(minutes=1))
        now = datetime.now(UTC)
        history.add(CaptionSegment(now - timedelta(minutes=2), "old"))
        for word in ("one", "two", "three", "four"):
            history.add(CaptionSegment(now, word))
        self.assertEqual(["two", "three", "four"], [x.source_text for x in history.snapshot()])

    def test_attaches_translation_to_existing_sentence(self) -> None:
        history = BoundedCaptionHistory(3, timedelta(minutes=1))
        history.add(CaptionSegment(datetime.now(UTC), "Current sentence.", sentence_id=7))
        history.update_translation(7, "当前句子。")
        self.assertEqual("当前句子。", history.snapshot()[0].translated_text)


class StabilizerTests(unittest.TestCase):
    def test_removes_overlap_between_adjacent_windows(self) -> None:
        stabilizer = TranscriptStabilizer()
        self.assertEqual("I think this feature", stabilizer.commit("I think this feature"))
        self.assertEqual("is useful", stabilizer.commit("this feature is useful"))
        self.assertEqual("", stabilizer.commit("is useful"))

    def test_removes_punctuation_variant_overlap_between_windows(self) -> None:
        stabilizer = TranscriptStabilizer()
        self.assertEqual(
            "Like the final scene in Game of Thrones Red Wedding when",
            stabilizer.commit("Like the final scene in Game of Thrones Red Wedding when"),
        )
        self.assertEqual(
            "shockingly, everybody dies.",
            stabilizer.commit("wedding when, shockingly, everybody dies."),
        )

    def test_preserves_legitimate_repetition_inside_one_window(self) -> None:
        stabilizer = TranscriptStabilizer()
        self.assertEqual("very very important", stabilizer.commit("very very important"))
        self.assertEqual("go go go", stabilizer.commit("go go go"))

    def test_collapses_punctuated_single_word_overlap_artifact(self) -> None:
        stabilizer = TranscriptStabilizer()
        self.assertEqual("models become.", stabilizer.commit("models become. become."))

    def test_collapses_punctuated_multiword_asr_repetition(self) -> None:
        stabilizer = TranscriptStabilizer()
        self.assertEqual(
            "Red Wedding when, shockingly, everybody dies.",
            stabilizer.commit(
                "Red Wedding when, wedding when, shockingly, everybody dies."
            ),
        )


class TranscriptCleaningTests(unittest.TestCase):
    def test_balanced_recognition_beam_is_enabled(self) -> None:
        self.assertEqual(3, DEFAULT_BEAM_SIZE)

    def test_removes_music_markers_without_dropping_spoken_text(self) -> None:
        self.assertEqual("", clean_transcript("♪ ♪ ♪ ♪"))
        self.assertEqual("", clean_transcript("[MUSIC]"))
        self.assertEqual(
            "This is an iced latte with honey.",
            clean_transcript("♪ ♪ This is an iced latte with honey."),
        )

    def test_removes_chunk_boundary_ellipsis(self) -> None:
        self.assertEqual("combines into a database", clean_transcript("...combines into a database..."))
        self.assertEqual("a natural ... pause remains", clean_transcript("a natural ... pause remains"))


class SegmenterTests(unittest.TestCase):
    def test_buffer_is_bounded(self) -> None:
        config = RuntimeConfig(sample_rate=100)
        segmenter = EnergyUtteranceSegmenter(config)
        for index in range(1_000):
            samples = [0.1] * 25 if index % 2 == 0 else [0.0] * 25
            segmenter.push(samples)
            self.assertLessEqual(segmenter.buffered_sample_count, segmenter.maximum_buffered_samples)

    def test_forced_cut_keeps_overlap_for_next_window(self) -> None:
        config = RuntimeConfig(
            sample_rate=100,
            maximum_utterance_seconds=1,
            pre_roll_ms=200,
            minimum_speech_ms=100,
        )
        segmenter = EnergyUtteranceSegmenter(config)
        result = None
        for _ in range(10):
            result = segmenter.push([0.1] * 10)
        self.assertIsNotNone(result)
        self.assertEqual(20, segmenter.buffered_sample_count)
        self.assertFalse(segmenter.last_chunk_is_final)

        for _ in range(6):
            result = segmenter.push([0.0] * 10)
        self.assertIsNotNone(result)
        self.assertTrue(segmenter.last_chunk_is_final)


if __name__ == "__main__":
    unittest.main()

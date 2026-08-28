import unittest
from datetime import UTC, datetime

from live_caption.models import CaptionSegment
from live_caption.overlay import OverlayState, fitted_overlay_height


class OverlayStateTests(unittest.TestCase):
    def test_revisions_replace_the_same_sentence(self) -> None:
        state = OverlayState()
        now = datetime.now(UTC)
        state.update(CaptionSegment(now, "I do not", "我不", False, 3, 1))
        snapshot = state.update(
            CaptionSegment(now, "I do not like it.", "我不喜欢它。", True, 3, 2)
        )
        self.assertEqual("I do not like it.", snapshot.source_text)
        self.assertEqual("我不喜欢它。", snapshot.translated_text)
        self.assertTrue(snapshot.is_final)

    def test_old_translation_cannot_replace_new_sentence(self) -> None:
        state = OverlayState()
        now = datetime.now(UTC)
        state.update(CaptionSegment(now, "New sentence.", "", False, 5, 1))
        result = state.update(CaptionSegment(now, "Old sentence.", "旧句。", True, 4, 9))
        self.assertIsNone(result)
        self.assertEqual("New sentence.", state.snapshot().source_text)

    def test_new_source_revision_clears_old_translation(self) -> None:
        state = OverlayState()
        now = datetime.now(UTC)
        state.update(CaptionSegment(now, "I think", "我认为", False, 2, 1))
        snapshot = state.update(CaptionSegment(now, "I think it works", "", False, 2, 2))
        self.assertEqual("I think it works", snapshot.source_text)
        self.assertEqual("", snapshot.translated_text)

    def test_older_revision_of_same_sentence_is_ignored(self) -> None:
        state = OverlayState()
        now = datetime.now(UTC)
        state.update(CaptionSegment(now, "New version", "新版本", False, 2, 3))
        result = state.update(CaptionSegment(now, "Old version", "旧版本", False, 2, 2))
        self.assertIsNone(result)
        self.assertEqual("New version", state.snapshot().source_text)


class OverlaySizingTests(unittest.TestCase):
    def test_short_caption_keeps_compact_height(self) -> None:
        self.assertEqual(178, fitted_overlay_height(30, 28, 32))

    def test_long_caption_expands_to_fit_content(self) -> None:
        self.assertEqual(242, fitted_overlay_height(30, 80, 100))

    def test_extremely_long_caption_respects_screen_limit(self) -> None:
        self.assertEqual(360, fitted_overlay_height(30, 300, 300))


if __name__ == "__main__":
    unittest.main()

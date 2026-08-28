import unittest
import threading
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from live_caption.models import CaptionSegment
from live_caption.translator import (
    LlamaCppTranslator,
    clean_translation,
    has_untranslated_english,
    has_repeated_chinese_phrase,
    looks_like_context_repetition,
    looks_like_prompt_leak,
)


class TranslationGuardTests(unittest.TestCase):
    def test_repairs_sprawling_without_leaving_english(self):
        translator = object.__new__(LlamaCppTranslator)
        responses = iter(("伦敦很 sprawling。", "伦敦很 sprawling。"))
        translator._request = lambda prompt: next(responses)
        self.assertEqual("伦敦很辽阔。", translator.translate("London is sprawling.", ()))

    def test_removes_spaces_between_chinese_words(self):
        self.assertEqual("我推荐去看看。", clean_translation("我 推荐 去 看看 。"))
        self.assertEqual(
            "苹果、香蕉……“都可以”（很方便）",
            clean_translation("苹果 、 香蕉 …… “ 都可以 ” （ 很方便 ）"),
        )

    def test_detects_accidental_repeated_chinese_phrase(self):
        self.assertTrue(has_repeated_chinese_phrase("我爱轮船轮船，我推荐推荐。"))
        self.assertFalse(has_repeated_chinese_phrase("你可以去看看。"))
        self.assertFalse(has_repeated_chinese_phrase("我们再讨论讨论。"))

    def test_valid_reduplication_is_not_suppressed_after_retry(self):
        translator = object.__new__(LlamaCppTranslator)
        responses = iter(("我们讨论讨论，再研究研究。", "我们讨论讨论，再研究研究。"))
        translator._request = lambda prompt: next(responses)
        self.assertEqual(
            "我们讨论讨论，再研究研究。",
            translator.translate("Let's discuss it and study it some more.", ()),
        )

    def test_retries_repeated_chinese_output(self):
        translator = object.__new__(LlamaCppTranslator)
        responses = iter(("我 爱 轮船 轮船，我 推荐 推荐 去看看。", "我推荐去看看。"))
        translator._request = lambda prompt: next(responses)
        self.assertEqual("我推荐去看看。", translator.translate("I recommend checking it out.", ()))

    def test_repairs_broken_but_connector(self):
        self.assertEqual(
            "现在已经没问题了。但是，我更喜欢早上。",
            clean_translation("现在已经没问题了。但。我更喜欢早上。"),
        )

    def test_detects_implausibly_long_translation(self):
        source = "She decided to teach me herself on weekdays."
        translated = "她决定工作日亲自教我。" + "这是上一段被错误重复的很长内容。" * 4
        self.assertTrue(looks_like_context_repetition(source, translated, ()))

    def test_detects_complete_previous_translation_inside_current_output(self):
        previous = CaptionSegment(
            datetime.now(UTC),
            "I kept up with an American education.",
            "我一直跟上美国的教育进度。",
            sentence_id=1,
        )

    def test_allows_deliberate_source_repetition_and_continuation(self):
        previous = CaptionSegment(
            datetime.now(UTC),
            "I kept up with an American education.",
            "我一直跟上美国的教育进度。",
            sentence_id=1,
        )
        self.assertFalse(
            looks_like_context_repetition(
                "I kept up with an American education, and it helped me later.",
                "我一直跟上美国的教育进度，而且这后来对我帮助很大。",
                (previous,),
            )
        )
        self.assertTrue(
            looks_like_context_repetition(
                "So she decided to teach me herself.",
                "我一直跟上美国的教育进度。所以她决定亲自教我。",
                (previous,),
            )
        )

    def test_retries_context_repetition_without_showing_it(self):
        translator = object.__new__(LlamaCppTranslator)
        responses = iter(("上一句的完整翻译。所以她决定亲自教我。", "所以她决定亲自教我。"))
        translator._request = lambda prompt: next(responses)
        context = (
            CaptionSegment(datetime.now(UTC), "Previous.", "上一句的完整翻译。"),
        )
        self.assertEqual(
            "所以她决定亲自教我。",
            translator.translate("So she decided to teach me herself.", context),
        )

    def test_detects_internal_translation_instruction_leak(self):
        self.assertTrue(
            looks_like_prompt_leak("上一版存在漏译，这次必须翻译每个短语和句子，不能照抄英文。")
        )
        self.assertFalse(looks_like_prompt_leak("上一版电影的翻译很自然。"))

    def test_retries_instead_of_showing_internal_instruction(self):
        translator = object.__new__(LlamaCppTranslator)
        responses = iter(
            (
                "上一版存在漏译，这次必须翻译每个短语和句子，不能照抄英文。",
                "这是自然的中文字幕。",
            )
        )
        translator._request = lambda prompt: next(responses)
        self.assertEqual("这是自然的中文字幕。", translator.translate("Natural captions.", ()))

    def test_detects_copied_english_phrase(self):
        self.assertTrue(has_untranslated_english("信任我，everything gets easier."))

    def test_detects_lowercase_untranslated_word(self):
        self.assertTrue(has_untranslated_english("这是为你做的。 exactly."))
        self.assertTrue(has_untranslated_english("这是为你做的。 Exactly."))

    def test_allows_proper_name_and_acronym(self):
        self.assertFalse(has_untranslated_english("Monica。"))
        self.assertFalse(has_untranslated_english("我们可以用 AI。"))
        self.assertFalse(has_untranslated_english("分析 DNA，并比较 DNA。"))

    def test_accepts_chinese_translation(self):
        self.assertFalse(has_untranslated_english("相信我，一旦开始用英语思考，一切都会容易起来。"))

    def test_repairs_common_discourse_word_without_model_call(self):
        translator = object.__new__(LlamaCppTranslator)
        repaired = translator._repair_residual_english(
            "This episode is for you. Exactly.", "这一集是为你准备的。 exactly."
        )
        self.assertEqual("这一集是为你准备的。 没错。", repaired)

    def test_repairs_common_phrase_without_model_call(self):
        translator = object.__new__(LlamaCppTranslator)
        repaired = translator._repair_residual_english(
            "Red Wedding, shockingly. Everybody dies.",
            "红色婚礼，令人震惊。 everybody dies.",
        )
        self.assertEqual("红色婚礼，令人震惊。 所有人都会死。", repaired)

    def test_repairs_stubborn_conversational_english_instead_of_blank_caption(self):
        translator = object.__new__(LlamaCppTranslator)
        responses = iter(
            (
                "从附近超市，而且 yeah，just enjoy.",
                "从附近超市，而且 yeah，just enjoy.",
            )
        )
        translator._request = lambda prompt: next(responses)
        self.assertEqual(
            "从附近超市，然后享受就好。",
            translator.translate("from a nearby supermarket, and yeah, just enjoy.", ()),
        )

    def test_repairs_stubborn_probability_word_instead_of_blank_caption(self):
        translator = object.__new__(LlamaCppTranslator)
        responses = iter(
            (
                "如果我搬回伦敦，我 probably 会，我 probably 会搬家。",
                "如果我搬回伦敦，我 probably 会搬家。",
            )
        )
        translator._request = lambda prompt: next(responses)
        self.assertEqual(
            "如果我搬回伦敦，我可能会搬家。",
            translator.translate(
                "If I were moving back to London, I would probably, I'd probably move.",
                (),
            ),
        )

    def test_repairs_stubborn_emphasis_word_instead_of_blank_caption(self):
        translator = object.__new__(LlamaCppTranslator)
        responses = iter(
            (
                "我 definitely 推荐去看看。",
                "我 definitely 推荐去看看。",
            )
        )
        translator._request = lambda prompt: next(responses)
        self.assertEqual(
            "我非常推荐去看看。",
            translator.translate("I definitely recommend checking it out.", ()),
        )

    def test_translates_cheers_as_a_toast_when_context_mentions_drinks(self):
        translator = object.__new__(LlamaCppTranslator)
        context = (
            CaptionSegment(datetime.now(UTC), "They brought a drink.", "他们带了饮料。"),
        )
        self.assertEqual("干杯。", translator.translate("Cheers.", context))

    def test_translates_standalone_cheers_as_thanks_without_drinking_context(self):
        translator = object.__new__(LlamaCppTranslator)
        self.assertEqual("谢谢。", translator.translate("Cheers!", ()))

    def test_source_only_context_is_included_in_prompt(self):
        translator = object.__new__(LlamaCppTranslator)
        prompts = []

        def request(prompt):
            prompts.append(prompt)
            return "他随后离开了。"

        translator._request = request
        context = (
            CaptionSegment(datetime.now(UTC), "Alex entered the room.", ""),
        )
        self.assertEqual("他随后离开了。", translator.translate("He then left.", context))
        self.assertIn("原文：Alex entered the room.", prompts[0])
        self.assertIn("当前句：He then left.", prompts[0])


class TranslatorLifecycleTests(unittest.TestCase):
    def test_cancelled_startup_closes_spawned_server(self):
        class FakeProcess:
            def __init__(self):
                self.terminated = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                del timeout
                return 0

        process = FakeProcess()
        cancelled = threading.Event()
        cancelled.set()
        with tempfile.TemporaryDirectory() as directory:
            server = Path(directory) / "llama-server.exe"
            model = Path(directory) / "model.gguf"
            server.touch()
            model.touch()
            with patch("live_caption.translator.subprocess.Popen", return_value=process):
                with self.assertRaises(InterruptedError):
                    LlamaCppTranslator(server, model, cancel_event=cancelled)
        self.assertTrue(process.terminated)


if __name__ == "__main__":
    unittest.main()

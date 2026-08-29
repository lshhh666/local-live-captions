from __future__ import annotations

import threading
import unittest
from array import array

from live_caption.config import RuntimeConfig
from live_caption.pipeline import CaptionPipeline


class _Recognizer:
    def recognize(self, samples) -> str:
        del samples
        return "Hello world."


class _BlockingTranslator:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def translate(self, source, context) -> str:
        del source, context
        self.started.set()
        self.release.wait(timeout=2)
        return "你好，世界。"

    def close(self) -> None:
        self.release.set()


class _SequenceRecognizer:
    def __init__(self) -> None:
        self._results = iter(("I do not", "like it."))

    def recognize(self, samples) -> str:
        del samples
        return next(self._results)


class _ListRecognizer:
    def __init__(self, *results: str) -> None:
        self._results = iter(results)

    def recognize(self, samples) -> str:
        del samples
        return next(self._results)


class _StopBlockingRecognizer:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def recognize(self, samples) -> str:
        del samples
        self.started.set()
        self.release.wait(timeout=2)
        return "Must not appear."


class _RevisionTranslator(_BlockingTranslator):
    def translate(self, source, context) -> str:
        del context
        if not self.started.is_set():
            self.started.set()
            self.release.wait(timeout=2)
        return "旧的临时翻译" if source == "I do not" else "我不喜欢它。"


class _ContextRecordingTranslator:
    def __init__(self) -> None:
        self.contexts = []

    def translate(self, source, context) -> str:
        self.contexts.append(tuple(item.source_text for item in context))
        return f"译文：{source}"

    def close(self) -> None:
        pass


class _LatestOnlyTranslator:
    def __init__(self) -> None:
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.latest_done = threading.Event()

    def translate(self, source, context) -> str:
        del context
        self.calls.append(source)
        if len(self.calls) == 1:
            self.started.set()
            self.release.wait(timeout=2)
        if source.endswith("Three."):
            self.latest_done.set()
        return f"译文：{source}"

    def close(self) -> None:
        self.release.set()


class PipelineTests(unittest.TestCase):
    def test_merges_lowercase_prepositional_tail_across_forced_boundary(self) -> None:
        captions = []
        pipeline = CaptionPipeline(
            RuntimeConfig(), _ListRecognizer("unused"), _BlockingTranslator(), captions.append
        )
        pipeline._append_source(
            "If I moved back to London, I would probably move back.",
            chunk_is_final=False,
        )
        pipeline._append_source("into this area.", chunk_is_final=False)
        self.assertEqual(
            "If I moved back to London, I would probably move back into this area.",
            captions[-1].source_text,
        )
        self.assertFalse(captions[-1].is_final)

    def test_does_not_merge_capitalized_prepositional_sentence(self) -> None:
        captions = []
        pipeline = CaptionPipeline(
            RuntimeConfig(), _ListRecognizer("unused"), _BlockingTranslator(), captions.append
        )
        pipeline._append_source("The first sentence ends.", chunk_is_final=False)
        pipeline._append_source("Within minutes, everything changed.", chunk_is_final=False)
        self.assertEqual(("The first sentence ends.", True), (captions[1].source_text, captions[1].is_final))
        self.assertEqual("Within minutes, everything changed.", captions[-1].source_text)

    def test_queued_final_translation_is_selected_before_partial(self) -> None:
        pipeline = CaptionPipeline(
            RuntimeConfig(), _ListRecognizer("unused"), _BlockingTranslator(), lambda caption: None
        )
        partial = (1, 1, None, "partial", False, ())
        final = (2, 1, None, "final", True, ())
        with pipeline._revision_lock:
            pipeline._latest_revisions.update({1: 1, 2: 1})
        pipeline._translation_queue.put(partial)
        pipeline._translation_queue.put(final)

        selected = pipeline._get_next_translation(timeout=0.1)
        self.assertEqual(final, selected)
        pipeline._translation_queue.task_done()
        deferred = pipeline._translation_queue.get_nowait()
        self.assertEqual(partial, deferred)
        pipeline._translation_queue.task_done()

    def test_merges_gerund_complement_split_by_forced_audio_boundary(self) -> None:
        captions = []
        pipeline = CaptionPipeline(
            RuntimeConfig(), _ListRecognizer("unused"), _BlockingTranslator(), captions.append
        )
        pipeline._append_source("I definitely recommend.", chunk_is_final=False)
        pipeline._append_source("and checking it out.", chunk_is_final=False)
        self.assertEqual(
            "I definitely recommend checking it out.",
            captions[-1].source_text,
        )
        self.assertFalse(captions[-1].is_final)

    def test_does_not_merge_unrelated_gerund_sentence(self) -> None:
        captions = []
        pipeline = CaptionPipeline(
            RuntimeConfig(), _ListRecognizer("unused"), _BlockingTranslator(), captions.append
        )
        pipeline._append_source("He stopped.", chunk_is_final=False)
        pipeline._append_source("And checking the clock, he left.", chunk_is_final=False)
        self.assertEqual(("He stopped.", True), (captions[1].source_text, captions[1].is_final))
        self.assertEqual("And checking the clock, he left.", captions[-1].source_text)

    def test_next_chunk_confirms_previous_punctuated_sentence(self) -> None:
        captions = []
        pipeline = CaptionPipeline(
            RuntimeConfig(), _ListRecognizer("unused"), _BlockingTranslator(), captions.append
        )
        pipeline._append_source("First complete sentence.", chunk_is_final=False)
        pipeline._append_source("Second sentence continues", chunk_is_final=False)
        self.assertEqual(
            [
                ("First complete sentence.", False),
                ("First complete sentence.", True),
                ("Second sentence continues", False),
            ],
            [(item.source_text, item.is_final) for item in captions],
        )

    def test_internal_punctuation_splits_continuous_narration(self) -> None:
        captions = []
        pipeline = CaptionPipeline(
            RuntimeConfig(), _ListRecognizer("unused"), _BlockingTranslator(), captions.append
        )
        pipeline._append_source(
            "First complete sentence. Second sentence continues", chunk_is_final=False
        )
        self.assertEqual(
            [
                ("First complete sentence.", True),
                ("Second sentence continues", False),
            ],
            [(item.source_text, item.is_final) for item in captions],
        )

    def test_abbreviation_does_not_split_sentence(self) -> None:
        captions = []
        pipeline = CaptionPipeline(
            RuntimeConfig(), _ListRecognizer("unused"), _BlockingTranslator(), captions.append
        )
        pipeline._append_source("Dr. Smith kept speaking", chunk_is_final=False)
        self.assertEqual("Dr. Smith kept speaking", captions[-1].source_text)
        self.assertFalse(captions[-1].is_final)

    def test_dotted_and_title_abbreviations_do_not_split_sentence(self) -> None:
        for source in (
            "He moved from the U.S. to Canada today",
            "The lesson starts at 9 a.m. every weekday",
            "Prof. Smith kept speaking",
        ):
            captions = []
            pipeline = CaptionPipeline(
                RuntimeConfig(), _ListRecognizer("unused"), _BlockingTranslator(), captions.append
            )
            pipeline._append_source(source, chunk_is_final=False)
            self.assertEqual(source, captions[-1].source_text)
            self.assertFalse(captions[-1].is_final)

    def test_long_continuous_sentence_splits_at_clause_boundary(self) -> None:
        captions = []
        pipeline = CaptionPipeline(
            RuntimeConfig(), _ListRecognizer("unused"), _BlockingTranslator(), captions.append
        )
        pipeline._append_source(
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen, "
            "then the next clause continues naturally",
            chunk_is_final=False,
        )
        self.assertEqual(
            [
                ("one two three four five six seven eight nine ten eleven twelve thirteen fourteen,", True),
                ("then the next clause continues naturally", False),
            ],
            [(item.source_text, item.is_final) for item in captions],
        )

    def test_short_comma_does_not_split_sentence(self) -> None:
        captions = []
        pipeline = CaptionPipeline(
            RuntimeConfig(), _ListRecognizer("unused"), _BlockingTranslator(), captions.append
        )
        pipeline._append_source("However, this remains one short clause", chunk_is_final=False)
        self.assertEqual(1, len(captions))
        self.assertEqual("However, this remains one short clause", captions[0].source_text)
        self.assertFalse(captions[0].is_final)

    def test_long_continuous_sentence_splits_before_connector(self) -> None:
        captions = []
        pipeline = CaptionPipeline(
            RuntimeConfig(), _ListRecognizer("unused"), _BlockingTranslator(), captions.append
        )
        pipeline._append_source(
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
            "fifteen but the next clause continues naturally",
            chunk_is_final=False,
        )
        self.assertEqual(
            [
                ("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen", True),
                ("but the next clause continues naturally", False),
            ],
            [(item.source_text, item.is_final) for item in captions],
        )

    def test_connector_at_next_chunk_start_splits_existing_long_clause(self) -> None:
        captions = []
        pipeline = CaptionPipeline(
            RuntimeConfig(), _ListRecognizer("unused"), _BlockingTranslator(), captions.append
        )
        first = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
        pipeline._append_source(first, chunk_is_final=False)
        pipeline._append_source("but the next clause continues", chunk_is_final=False)
        self.assertEqual(
            [(first, False), (first, True), ("but the next clause continues", False)],
            [(item.source_text, item.is_final) for item in captions],
        )

    def test_final_history_exists_before_translation_is_enqueued(self) -> None:
        pipeline = CaptionPipeline(
            RuntimeConfig(), _ListRecognizer("unused"), _BlockingTranslator(), lambda item: None
        )
        pipeline._pending_source = "A final sentence."
        observed = []

        def inspect_queue(target, item, label):
            del target, item, label
            observed.extend(pipeline._history.snapshot())

        pipeline._put_latest = inspect_queue
        pipeline._emit_pending(is_final=True)
        self.assertEqual("A final sentence.", observed[0].source_text)

    def test_stop_waits_for_workers_then_clears_all_session_state(self) -> None:
        recognizer = _StopBlockingRecognizer()
        captions = []
        pipeline = CaptionPipeline(
            RuntimeConfig(), recognizer, _ContextRecordingTranslator(), captions.append
        )
        pipeline.start()
        pipeline._recognition_queue.put((array("f", [0.1]), True))
        self.assertTrue(recognizer.started.wait(timeout=1))

        stopped = threading.Event()
        stop_thread = threading.Thread(target=lambda: (pipeline.stop(), stopped.set()))
        stop_thread.start()
        self.assertFalse(stopped.wait(timeout=0.05))
        recognizer.release.set()
        self.assertTrue(stopped.wait(timeout=1))
        stop_thread.join(timeout=1)

        self.assertEqual([], captions)
        self.assertEqual((), pipeline._history.snapshot())
        self.assertEqual({}, pipeline._latest_revisions)
        self.assertTrue(pipeline._recognition_queue.empty())
        self.assertTrue(pipeline._translation_queue.empty())
        self.assertFalse(pipeline._recognition_worker.is_alive())
        self.assertFalse(pipeline._translation_worker.is_alive())

    def test_translation_overflow_does_not_leak_revision_entries(self) -> None:
        pipeline = CaptionPipeline(
            RuntimeConfig(translation_queue_size=2),
            _ListRecognizer(),
            _ContextRecordingTranslator(),
            lambda caption: None,
        )
        for source in ("One.", "Two.", "Three."):
            pipeline._pending_source = source
            pipeline._emit_pending(is_final=True)
            pipeline._reset_pending()
        self.assertEqual({1, 2}, set(pipeline._latest_revisions))
        pipeline.clear()
        self.assertEqual({}, pipeline._latest_revisions)
        self.assertEqual(0, pipeline._translation_queue.unfinished_tasks)

    def test_translation_overflow_prefers_dropping_provisional_item(self) -> None:
        pipeline = CaptionPipeline(
            RuntimeConfig(translation_queue_size=2),
            _ListRecognizer(),
            _ContextRecordingTranslator(),
            lambda caption: None,
        )
        for source, is_final in (("Final one.", True), ("Temporary", False), ("Final two.", True)):
            pipeline._pending_source = source
            pipeline._emit_pending(is_final=is_final)
            pipeline._reset_pending()
        queued = []
        while not pipeline._translation_queue.empty():
            item = pipeline._translation_queue.get_nowait()
            queued.append(item[3])
            pipeline._translation_queue.task_done()
        self.assertEqual(["Final one.", "Final two."], queued)
        self.assertEqual({0, 2}, set(pipeline._latest_revisions))
        pipeline.clear()

    def test_translation_context_keeps_only_recent_final_sentences(self) -> None:
        translator = _ContextRecordingTranslator()
        pipeline = CaptionPipeline(
            RuntimeConfig(context_sentences=2),
            _ListRecognizer(),
            translator,
            lambda caption: None,
        )
        contexts = []
        for source in ("One.", "Two.", "Three.", "Four."):
            pipeline._pending_source = source
            pipeline._emit_pending(is_final=True)
            item = pipeline._translation_queue.get_nowait()
            contexts.append(tuple(caption.source_text for caption in item[5]))
            pipeline._translation_queue.task_done()
            pipeline._reset_pending()
        self.assertEqual(
            [(), ("One.",), ("One.", "Two."), ("Two.", "Three.")], contexts
        )
        pipeline.clear()

    def test_busy_translation_skips_obsolete_revisions_of_same_sentence(self) -> None:
        translator = _LatestOnlyTranslator()
        source_count = 0
        all_sources_seen = threading.Event()
        latest_translation_seen = threading.Event()

        def receive(caption) -> None:
            nonlocal source_count
            if not caption.translated_text:
                source_count += 1
                if source_count == 3:
                    all_sources_seen.set()
            elif caption.source_text == "One Two Three.":
                latest_translation_seen.set()

        pipeline = CaptionPipeline(
            RuntimeConfig(),
            _ListRecognizer("One", "Two", "Three."),
            translator,
            receive,
        )
        pipeline.start()
        try:
            pipeline._recognition_queue.put((array("f", [0.1]), False))
            self.assertTrue(translator.started.wait(timeout=1))
            pipeline._recognition_queue.put((array("f", [0.1]), False))
            pipeline._recognition_queue.put((array("f", [0.1]), True))
            self.assertTrue(all_sources_seen.wait(timeout=1))
            translator.release.set()
            self.assertTrue(translator.latest_done.wait(timeout=1))
            self.assertTrue(latest_translation_seen.wait(timeout=1))
            self.assertEqual(["One", "One Two Three."], translator.calls)
            self.assertEqual({}, pipeline._latest_revisions)
        finally:
            pipeline.stop()

    def test_busy_translation_preserves_distinct_final_sentences(self) -> None:
        translator = _LatestOnlyTranslator()
        pipeline = CaptionPipeline(
            RuntimeConfig(),
            _ListRecognizer("One.", "Two.", "Three."),
            translator,
            lambda caption: None,
        )
        pipeline.start()
        try:
            pipeline._recognition_queue.put((array("f", [0.1]), True))
            self.assertTrue(translator.started.wait(timeout=1))
            pipeline._recognition_queue.put((array("f", [0.1]), True))
            pipeline._recognition_queue.put((array("f", [0.1]), True))
            translator.release.set()
            self.assertTrue(translator.latest_done.wait(timeout=1))
            self.assertEqual(["One.", "Two.", "Three."], translator.calls)
        finally:
            pipeline.stop()

    def test_single_large_recognition_result_is_split_at_word_limit(self) -> None:
        captions = []
        done = threading.Event()
        words = [f"word{index}" for index in range(50)]

        def receive(caption) -> None:
            if not caption.translated_text:
                captions.append(caption)
                if len(captions) == 2:
                    done.set()

        pipeline = CaptionPipeline(
            RuntimeConfig(maximum_sentence_words=45),
            _ListRecognizer(" ".join(words)),
            _ContextRecordingTranslator(),
            receive,
        )
        pipeline.start()
        try:
            pipeline._recognition_queue.put((array("f", [0.1]), True))
            self.assertTrue(done.wait(timeout=1))
            self.assertEqual([45, 5], [len(item.source_text.split()) for item in captions])
            self.assertTrue(all(item.is_final for item in captions))
        finally:
            pipeline.stop()

    def test_source_is_emitted_before_translation_finishes(self) -> None:
        translator = _BlockingTranslator()
        captions = []
        source_seen = threading.Event()
        translation_seen = threading.Event()

        def receive(caption) -> None:
            captions.append(caption)
            (translation_seen if caption.translated_text else source_seen).set()

        pipeline = CaptionPipeline(RuntimeConfig(), _Recognizer(), translator, receive)
        pipeline.start()
        try:
            pipeline._recognition_queue.put((array("f", [0.1]), True))
            self.assertTrue(source_seen.wait(timeout=1))
            self.assertTrue(translator.started.wait(timeout=1))
            self.assertFalse(translation_seen.is_set())
            self.assertEqual("Hello world.", captions[0].source_text)

            translator.release.set()
            self.assertTrue(translation_seen.wait(timeout=1))
            self.assertEqual("你好，世界。", captions[-1].translated_text)
        finally:
            pipeline.stop()

    def test_repeated_recognition_finalizes_instead_of_reprinting_partial(self) -> None:
        captions = []
        source_events = threading.Event()

        def receive(caption) -> None:
            if not caption.translated_text:
                captions.append(caption)
                if len(captions) >= 2:
                    source_events.set()

        pipeline = CaptionPipeline(
            RuntimeConfig(),
            _ListRecognizer("Thank you.", "Thank you."),
            _BlockingTranslator(),
            receive,
        )
        pipeline.start()
        try:
            pipeline._recognition_queue.put((array("f", [0.1]), False))
            pipeline._recognition_queue.put((array("f", [0.1]), True))
            self.assertTrue(source_events.wait(timeout=1))
            self.assertEqual([False, True], [item.is_final for item in captions])
            self.assertEqual(["Thank you.", "Thank you."], [item.source_text for item in captions])
        finally:
            pipeline.stop()

    def test_punctuated_forced_chunks_become_separate_sentences(self) -> None:
        captions = []
        source_events = threading.Event()

        def receive(caption) -> None:
            if not caption.translated_text:
                captions.append(caption)
                if len(captions) >= 3:
                    source_events.set()

        pipeline = CaptionPipeline(
            RuntimeConfig(),
            _ListRecognizer("First sentence.", "Second sentence."),
            _BlockingTranslator(),
            receive,
        )
        pipeline.start()
        try:
            pipeline._recognition_queue.put((array("f", [0.1]), False))
            pipeline._recognition_queue.put((array("f", [0.1]), True))
            self.assertTrue(source_events.wait(timeout=1))
            self.assertEqual(
                [
                    ("First sentence.", False),
                    ("First sentence.", True),
                    ("Second sentence.", True),
                ],
                [(item.source_text, item.is_final) for item in captions],
            )
        finally:
            pipeline.stop()

    def test_stale_partial_translation_cannot_override_final_revision(self) -> None:
        translator = _RevisionTranslator()
        captions = []
        final_translation_seen = threading.Event()

        def receive(caption) -> None:
            captions.append(caption)
            if caption.translated_text and caption.is_final:
                final_translation_seen.set()

        pipeline = CaptionPipeline(
            RuntimeConfig(), _SequenceRecognizer(), translator, receive
        )
        pipeline.start()
        try:
            pipeline._recognition_queue.put((array("f", [0.1]), False))
            self.assertTrue(translator.started.wait(timeout=1))
            pipeline._recognition_queue.put((array("f", [0.1]), True))

            for _ in range(100):
                if len([item for item in captions if not item.translated_text]) >= 2:
                    break
                threading.Event().wait(0.01)
            translator.release.set()
            self.assertTrue(final_translation_seen.wait(timeout=1))

            translations = [item.translated_text for item in captions if item.translated_text]
            self.assertEqual(["我不喜欢它。"], translations)
            self.assertEqual(
                "我不喜欢它。",
                pipeline._history.snapshot()[-1].translated_text,
            )
        finally:
            pipeline.stop()


if __name__ == "__main__":
    unittest.main()

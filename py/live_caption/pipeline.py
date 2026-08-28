from __future__ import annotations

import queue
import re
import threading
from datetime import UTC, datetime, timedelta

from .buffers import BoundedCaptionHistory, FixedAudioBuffer
from .config import RuntimeConfig
from .models import CaptionSegment
from .segmenter import EnergyUtteranceSegmenter
from .stabilizer import TranscriptStabilizer


class CaptionPipeline:
    _PREFERRED_CLAUSE_WORDS = 14
    _CLAUSE_CONNECTORS = {"although", "and", "because", "but", "however", "so", "though", "while"}
    _GERUND_COMPLEMENT_VERBS = {"recommend", "recommended"}
    _NON_TERMINAL_ABBREVIATIONS = {
        "dr.",
        "e.g.",
        "etc.",
        "i.e.",
        "mr.",
        "mrs.",
        "ms.",
        "prof.",
        "jr.",
        "sr.",
        "fig.",
        "inc.",
        "ltd.",
        "no.",
        "st.",
        "vs.",
    }
    _DOTTED_ABBREVIATION = re.compile(r"^(?:[a-z]\.){2,}$", re.IGNORECASE)

    def __init__(self, config: RuntimeConfig, recognizer, translator, on_caption, on_error=None) -> None:
        self._config = config
        self._recognizer = recognizer
        self._translator = translator
        self._on_caption = on_caption
        self._on_error = on_error
        self._audio = FixedAudioBuffer(config.sample_rate, config.audio_buffer_seconds)
        self._segmenter = EnergyUtteranceSegmenter(config)
        self._stabilizer = TranscriptStabilizer()
        self._history = BoundedCaptionHistory(
            config.context_sentences, timedelta(minutes=config.context_minutes)
        )
        self._recognition_queue: queue.Queue = queue.Queue(config.recognition_queue_size)
        self._translation_queue: queue.Queue = queue.Queue(config.translation_queue_size)
        self._stop = threading.Event()
        self._revision_lock = threading.Lock()
        self._latest_revisions: dict[int, int] = {}
        self._pending_source = ""
        self._pending_timestamp: datetime | None = None
        self._sentence_id = 0
        self._revision = 0
        self._empty_chunks = 0
        self._recognition_worker = threading.Thread(
            target=self._run_recognition, name="recognition", daemon=True
        )
        self._translation_worker = threading.Thread(
            target=self._run_translation, name="translation", daemon=True
        )

    def start(self) -> None:
        self._recognition_worker.start()
        self._translation_worker.start()

    def push_audio(self, samples) -> None:
        self._audio.write(samples)
        utterance = self._segmenter.push(samples)
        if utterance is None:
            return
        self._put_latest(
            self._recognition_queue,
            (utterance, self._segmenter.last_chunk_is_final),
            "识别",
        )

    def stop(self) -> None:
        self._stop.set()
        close = getattr(self._translator, "close", None)
        try:
            if close is not None:
                close()
        finally:
            self._recognition_worker.join()
            self._translation_worker.join()
            self.clear()

    def clear(self) -> None:
        self._audio.clear()
        self._segmenter.clear()
        self._stabilizer.clear()
        self._history.clear()
        self._pending_source = ""
        self._pending_timestamp = None
        self._sentence_id = 0
        self._revision = 0
        self._empty_chunks = 0
        with self._revision_lock:
            self._latest_revisions.clear()
        while True:
            try:
                self._recognition_queue.get_nowait()
                self._recognition_queue.task_done()
            except queue.Empty:
                break
        while True:
            try:
                self._translation_queue.get_nowait()
                self._translation_queue.task_done()
            except queue.Empty:
                break

    def _run_recognition(self) -> None:
        while not self._stop.is_set():
            try:
                utterance, chunk_is_final = self._recognition_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                recognized = self._recognizer.recognize(utterance)
                if self._stop.is_set():
                    continue
                source = self._stabilizer.commit(recognized)
                if not source:
                    if not self._pending_source:
                        self._empty_chunks = 0
                        continue
                    self._empty_chunks += 1
                    if (
                        chunk_is_final
                        or self._empty_chunks >= 2
                    ):
                        self._emit_pending(is_final=True)
                        self._reset_pending(clear_stabilizer=chunk_is_final)
                    continue

                self._empty_chunks = 0
                self._append_source(source, chunk_is_final)
            except Exception as error:
                if not self._stop.is_set() and self._on_error is not None:
                    self._on_error(error)
            finally:
                self._recognition_queue.task_done()

    def _run_translation(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._translation_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            sentence_id, revision, timestamp, source, is_final, context = item
            try:
                with self._revision_lock:
                    is_latest_revision = self._latest_revisions.get(sentence_id) == revision
                if not is_latest_revision:
                    continue
                translated = self._translator.translate(source, context)
                if self._stop.is_set():
                    continue
                with self._revision_lock:
                    is_current = self._latest_revisions.get(sentence_id) == revision
                if is_current:
                    caption = CaptionSegment(
                        timestamp,
                        source,
                        translated,
                        is_final,
                        sentence_id,
                        revision,
                    )
                    if is_final and translated:
                        self._history.update_translation(sentence_id, translated)
                    self._on_caption(caption)
            except Exception as error:
                if not self._stop.is_set() and self._on_error is not None:
                    self._on_error(error)
            finally:
                with self._revision_lock:
                    if self._latest_revisions.get(sentence_id) == revision:
                        self._latest_revisions.pop(sentence_id, None)
                self._translation_queue.task_done()

    def _emit_pending(self, is_final: bool) -> None:
        if not self._pending_source:
            return
        self._revision += 1
        timestamp = self._pending_timestamp or datetime.now(UTC)
        sentence_id = self._sentence_id
        revision = self._revision
        current_source = self._pending_source
        context = self._history.recent(self._config.context_sentences)
        with self._revision_lock:
            self._latest_revisions[sentence_id] = revision
        self._on_caption(
            CaptionSegment(
                timestamp,
                current_source,
                "",
                is_final,
                sentence_id,
                revision,
            )
        )
        if is_final:
            self._history.add(
                CaptionSegment(timestamp, current_source, "", True, sentence_id, revision)
            )
        self._put_latest(
            self._translation_queue,
            (sentence_id, revision, timestamp, current_source, is_final, context),
            "翻译",
        )

    def _reset_pending(self, clear_stabilizer: bool = False) -> None:
        self._pending_source = ""
        self._pending_timestamp = None
        self._sentence_id += 1
        self._revision = 0
        self._empty_chunks = 0
        if clear_stabilizer:
            self._stabilizer.clear()

    def _append_source(self, source: str, chunk_is_final: bool) -> None:
        # A forced audio cut may end with plausible sentence punctuation. Wait
        # until the next recognized text arrives before trusting that boundary.
        source = self._merge_gerund_complement_across_boundary(source)
        pending_words = len(self._pending_source.split())
        first_source_word = self._normalized_word(source.split()[0]) if source.split() else ""
        if self._pending_source and (
            (
                pending_words >= self._PREFERRED_CLAUSE_WORDS
                and first_source_word in self._CLAUSE_CONNECTORS
            )
            or self._ends_sentence(self._pending_source)
            or (
                pending_words >= self._PREFERRED_CLAUSE_WORDS
                and self._ends_clause(self._pending_source)
            )
        ):
            self._emit_pending(is_final=True)
            self._reset_pending()

        remaining = source.split()
        limit = self._config.maximum_sentence_words
        while remaining:
            current_count = len(self._pending_source.split())
            available = max(1, limit - current_count)
            addition = remaining[:available]
            boundary = 0
            for index, token in enumerate(addition):
                has_following_text = index + 1 < len(remaining) or chunk_is_final
                connector = self._normalized_word(token)
                if (
                    index > 0
                    and current_count + index >= self._PREFERRED_CLAUSE_WORDS
                    and connector in self._CLAUSE_CONNECTORS
                ):
                    boundary = index
                    break
                if has_following_text and (
                    self._ends_sentence(token)
                    or (
                        current_count + index + 1 >= self._PREFERRED_CLAUSE_WORDS
                        and self._ends_clause(token)
                    )
                ):
                    boundary = index + 1
                    break
            if boundary:
                addition = addition[:boundary]
            del remaining[: len(addition)]
            if not self._pending_source:
                self._pending_timestamp = datetime.now(UTC)
            self._pending_source = " ".join(
                part for part in (self._pending_source, " ".join(addition)) if part
            )
            reached_limit = len(self._pending_source.split()) >= limit
            is_final = reached_limit or bool(boundary) or (chunk_is_final and not remaining)
            self._emit_pending(is_final=is_final)
            if is_final:
                self._reset_pending(clear_stabilizer=chunk_is_final and not remaining)

    @classmethod
    def _ends_sentence(cls, text: str) -> bool:
        token = text.rstrip().split()[-1].rstrip('"\')]}') if text.strip() else ""
        if (
            token.casefold() in cls._NON_TERMINAL_ABBREVIATIONS
            or cls._DOTTED_ABBREVIATION.fullmatch(token)
        ):
            return False
        return token.endswith((".", "!", "?", "。", "！", "？"))

    @staticmethod
    def _ends_clause(text: str) -> bool:
        token = text.rstrip().split()[-1].rstrip('"\')]}') if text.strip() else ""
        return token.endswith((",", ";", ":", "，", "；", "："))

    @staticmethod
    def _normalized_word(token: str) -> str:
        return token.strip('"\'()[]{}.,;:!?。，；：！？').casefold()

    def _merge_gerund_complement_across_boundary(self, source: str) -> str:
        """Repair narrow forced-cut artifacts such as 'recommend.' + 'and checking'."""
        pending_tokens = self._pending_source.split()
        source_tokens = source.split()
        if len(source_tokens) < 2 or not pending_tokens:
            return source
        previous = self._normalized_word(pending_tokens[-1])
        if (
            self._normalized_word(source_tokens[0]) != "and"
            or self._normalized_word(source_tokens[1]) != "checking"
            or previous not in self._GERUND_COMPLEMENT_VERBS
        ):
            return source
        self._pending_source = re.sub(r"[.!?]+\Z", "", self._pending_source).rstrip()
        return " ".join(source_tokens[1:])

    def _put_latest(self, target: queue.Queue, item, label: str) -> None:
        try:
            target.put_nowait(item)
            return
        except queue.Full:
            pass
        if target is self._translation_queue:
            dropped = self._replace_full_translation_queue(item)
            if dropped is not None:
                self._forget_dropped_translation(dropped)
                if self._on_error is not None:
                    self._on_error(RuntimeError(f"{label}速度跟不上，已跳过一条最旧内容"))
            return
        try:
            target.get_nowait()
            target.task_done()
        except queue.Empty:
            pass
        target.put_nowait(item)
        if self._on_error is not None:
            self._on_error(RuntimeError(f"{label}速度跟不上，已跳过一条最旧内容"))

    def _replace_full_translation_queue(self, incoming):
        items = []
        while True:
            try:
                items.append(self._translation_queue.get_nowait())
                self._translation_queue.task_done()
            except queue.Empty:
                break
        items.append(incoming)
        if len(items) <= self._translation_queue.maxsize:
            for queued in items:
                self._translation_queue.put_nowait(queued)
            return None

        with self._revision_lock:
            drop_index = next(
                (
                    index
                    for index, queued in enumerate(items)
                    if self._latest_revisions.get(queued[0]) != queued[1]
                ),
                -1,
            )
        if drop_index < 0:
            drop_index = next(
                (index for index, queued in enumerate(items) if not queued[4]),
                0,
            )
        dropped = items.pop(drop_index)
        for queued in items:
            self._translation_queue.put_nowait(queued)
        return dropped

    def _forget_dropped_translation(self, dropped) -> None:
        sentence_id, revision, *_ = dropped
        with self._revision_lock:
            if self._latest_revisions.get(sentence_id) == revision:
                self._latest_revisions.pop(sentence_id, None)

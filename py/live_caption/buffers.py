from __future__ import annotations

from array import array
from collections import deque
from datetime import datetime, timedelta
from threading import Lock
from typing import Iterable

from .models import CaptionSegment


class FixedAudioBuffer:
    """Fixed-capacity float32 ring buffer; old samples are overwritten."""

    def __init__(self, sample_rate: int, duration_seconds: float) -> None:
        capacity = int(sample_rate * duration_seconds)
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._data = array("f", [0.0]) * capacity
        self._write_index = 0
        self._count = 0
        self._lock = Lock()

    @property
    def capacity(self) -> int:
        return len(self._data)

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def write(self, samples: Iterable[float]) -> None:
        values = array("f", samples)
        with self._lock:
            if len(values) >= self.capacity:
                self._data[:] = values[-self.capacity :]
                self._write_index = 0
                self._count = self.capacity
                return

            for sample in values:
                self._data[self._write_index] = sample
                self._write_index = (self._write_index + 1) % self.capacity
            self._count = min(self.capacity, self._count + len(values))

    def snapshot(self) -> array:
        with self._lock:
            start = (self._write_index - self._count) % self.capacity
            if start + self._count <= self.capacity:
                return array("f", self._data[start : start + self._count])
            first = self._data[start:]
            second_count = self._count - len(first)
            return array("f", first + self._data[:second_count])

    def clear(self) -> None:
        with self._lock:
            self._data[:] = array("f", [0.0]) * self.capacity
            self._write_index = 0
            self._count = 0


class BoundedCaptionHistory:
    def __init__(self, maximum_items: int, maximum_age: timedelta) -> None:
        if maximum_items <= 0 or maximum_age <= timedelta(0):
            raise ValueError("history bounds must be positive")
        self._maximum_items = maximum_items
        self._maximum_age = maximum_age
        self._items: deque[CaptionSegment] = deque()
        self._lock = Lock()

    def add(self, segment: CaptionSegment) -> None:
        with self._lock:
            self._items.append(segment)
            oldest_allowed = segment.timestamp - self._maximum_age
            while len(self._items) > self._maximum_items:
                self._items.popleft()
            while self._items and self._items[0].timestamp < oldest_allowed:
                self._items.popleft()

    def snapshot(self) -> tuple[CaptionSegment, ...]:
        with self._lock:
            return tuple(self._items)

    def recent(self, count: int) -> tuple[CaptionSegment, ...]:
        with self._lock:
            return tuple(list(self._items)[-count:])

    def update_translation(self, sentence_id: int, translated_text: str) -> None:
        """Attach an accepted final translation to its bounded history item."""
        with self._lock:
            for index, item in enumerate(self._items):
                if item.sentence_id != sentence_id:
                    continue
                self._items[index] = CaptionSegment(
                    item.timestamp,
                    item.source_text,
                    translated_text,
                    item.is_final,
                    item.sentence_id,
                    item.revision,
                )
                return

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

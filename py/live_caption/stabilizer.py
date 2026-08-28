from __future__ import annotations

from collections import deque
import re


_TOKEN_EDGE = re.compile(r"^[^\w']+|[^\w']+$")
_PUNCTUATED = re.compile(r"[.,;:!?…，；：！？。]$")
_KEEP_SINGLE_REPETITIONS = {"bye", "go", "ha", "no", "really", "so", "very"}


class TranscriptStabilizer:
    """Commits stable text and removes overlap between adjacent windows."""

    def __init__(self, tail_limit: int = 80) -> None:
        self._previous_partial: tuple[str, ...] = ()
        self._committed_tail: deque[str] = deque(maxlen=tail_limit)

    def update_partial(self, text: str) -> str:
        current = self._tokens(text)
        common_length = 0
        for left, right in zip(current, self._previous_partial, strict=False):
            if left.casefold() != right.casefold():
                break
            common_length += 1
        self._previous_partial = current
        return " ".join(current[:common_length])

    def commit(self, text: str) -> str:
        current = self._tokens(text)
        tail = tuple(self._committed_tail)
        overlap = 0
        for candidate in range(min(len(tail), len(current)), 0, -1):
            if tuple(self._normalized(token) for token in tail[-candidate:]) == tuple(
                self._normalized(token) for token in current[:candidate]
            ):
                overlap = candidate
                break

        new_tokens = self._collapse_punctuated_phrase_repetition(current[overlap:])
        self._committed_tail.extend(new_tokens)
        self._previous_partial = ()
        return " ".join(new_tokens)

    def clear(self) -> None:
        self._previous_partial = ()
        self._committed_tail.clear()

    @staticmethod
    def _tokens(text: str) -> tuple[str, ...]:
        return tuple(text.split())

    @staticmethod
    def _normalized(token: str) -> str:
        return _TOKEN_EDGE.sub("", token.casefold())

    @classmethod
    def _collapse_punctuated_phrase_repetition(
        cls, tokens: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Remove narrow ASR artifacts like 'wedding when, wedding when,'."""
        result = list(tokens)
        changed = True
        while changed:
            changed = False
            for width in range(min(4, len(result) // 2), 0, -1):
                for start in range(len(result) - width * 2 + 1):
                    left = result[start : start + width]
                    right = result[start + width : start + width * 2]
                    if not (_PUNCTUATED.search(left[-1]) and _PUNCTUATED.search(right[-1])):
                        continue
                    if tuple(cls._normalized(token) for token in left) != tuple(
                        cls._normalized(token) for token in right
                    ):
                        continue
                    if width == 1 and cls._normalized(left[0]) in _KEEP_SINGLE_REPETITIONS:
                        continue
                    del result[start + width : start + width * 2]
                    changed = True
                    break
                if changed:
                    break
        return tuple(result)

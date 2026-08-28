from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class CaptionSegment:
    timestamp: datetime
    source_text: str
    translated_text: str = ""
    is_final: bool = True
    sentence_id: int = 0
    revision: int = 0

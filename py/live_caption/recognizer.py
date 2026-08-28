from __future__ import annotations

import re
import os
import sys
from array import array
from pathlib import Path

import numpy as np


_DLL_DIRECTORY_HANDLES: list[object] = []


_NON_SPEECH_MARKERS = re.compile(
    r"\[(?:music|applause|laughter|silence)\]|"
    r"\((?:music|applause|laughter|silence)\)|"
    r"[♪♫♬♩]+|(?:\?\s*){3,}",
    re.IGNORECASE,
)
_EDGE_ELLIPSIS = re.compile(r"^(?:\.{2,}|…+)\s*|\s*(?:\.{2,}|…+)$")
DEFAULT_BEAM_SIZE = 3


def clean_transcript(text: str) -> str:
    """Remove common music/noise captions and punctuation-only hallucinations."""
    cleaned = " ".join(_NON_SPEECH_MARKERS.sub(" ", text).split()).strip()
    cleaned = _EDGE_ELLIPSIS.sub("", cleaned).strip()
    return cleaned if any(character.isalpha() for character in cleaned) else ""


def add_project_cuda_dll_directories() -> None:
    """Expose NVIDIA wheels installed inside this venv without changing Windows globally."""
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return
    frozen_root = getattr(sys, "_MEIPASS", None)
    package_root = (
        Path(frozen_root) / "nvidia"
        if frozen_root
        else Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    )
    directories = [
        package_root / "cublas" / "bin",
        package_root / "cudnn" / "bin",
        package_root / "cuda_nvrtc" / "bin",
    ]
    existing = [str(path) for path in directories if path.is_dir()]
    if not existing:
        return
    os.environ["PATH"] = os.pathsep.join(existing + [os.environ.get("PATH", "")])
    if not _DLL_DIRECTORY_HANDLES:
        _DLL_DIRECTORY_HANDLES.extend(os.add_dll_directory(path) for path in existing)


class FasterWhisperRecognizer:
    def __init__(
        self,
        model: str,
        language: str = "en",
        device: str = "cuda",
        compute_type: str = "float16",
        download_root: str | None = None,
    ) -> None:
        add_project_cuda_dll_directories()
        from faster_whisper import WhisperModel

        self._language = None if language == "auto" else language
        self._model = WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
        )
        if device == "cuda":
            # Loading the model alone does not load cuBLAS/cuDNN. A tiny warm-up
            # verifies the complete GPU runtime before audio capture begins.
            segments, _ = self._model.transcribe(
                np.zeros(8_000, dtype=np.float32),
                language=self._language,
                beam_size=1,
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.5,
                    "min_speech_duration_ms": 200,
                    "min_silence_duration_ms": 250,
                    "speech_pad_ms": 100,
                },
                condition_on_previous_text=False,
            )
            list(segments)

    def recognize(self, samples: array) -> str:
        audio = np.asarray(samples, dtype=np.float32)
        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=DEFAULT_BEAM_SIZE,
            vad_filter=True,
            vad_parameters={
                "threshold": 0.5,
                "min_speech_duration_ms": 200,
                "min_silence_duration_ms": 250,
                "speech_pad_ms": 100,
            },
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return clean_transcript(text)

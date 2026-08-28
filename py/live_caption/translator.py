from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import CaptionSegment


_ENGLISH_WORD = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?")
_ENGLISH_RUN = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?(?:\s+[A-Za-z]+(?:['’-][A-Za-z]+)?)*(?:[.!?])?")
_COMMON_TRANSLATIONS = {
    "absolutely": "当然",
    "actually": "其实",
    "cheers": "谢谢",
    "definitely": "非常",
    "exactly": "没错",
    "everybody dies": "所有人都会死",
    "hello": "你好",
    "just enjoy": "享受就好",
    "okay": "好的",
    "please": "请",
    "probably": "可能",
    "really": "真的",
    "right": "对",
    "sorry": "抱歉",
    "sprawling": "辽阔",
    "sure": "当然",
    "thanks": "谢谢",
    "welcome": "欢迎",
    "yeah": "嗯",
    "yes": "是的",
}
_PROMPT_LEAK_MARKERS = (
    "上一版存在漏译",
    "必须翻译每个短语和句子",
    "不能照抄英文",
    "不要解释，只输出",
    "只输出完整中文译文",
    "当前中文字幕还残留",
    "残留部分：",
    "不要重复词语",
    "不要在汉字之间加空格",
)
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_CJK = r"\u3400-\u4dbf\u4e00-\u9fff"
_ALLOWED_CJK_REDUPLICATIONS = {
    "常常",
    "处处",
    "刚刚",
    "渐渐",
    "看看",
    "慢慢",
    "人人",
    "试试",
    "天天",
    "往往",
    "想想",
}


def has_untranslated_english(text: str) -> bool:
    """Detect obvious copied English while allowing a single proper name or acronym."""
    words = [
        word
        for word in _ENGLISH_WORD.findall(text)
        if not (len(word) > 1 and word.isupper())
    ]
    if len(words) >= 2:
        return True
    return len(words) == 1 and (
        words[0][:1].islower() or words[0].lower() in _COMMON_TRANSLATIONS
    )


def looks_like_prompt_leak(text: str) -> bool:
    return any(marker in text for marker in _PROMPT_LEAK_MARKERS)


def clean_translation(text: str) -> str:
    """Repair narrow punctuation artifacts without rewriting model meaning."""
    cleaned = re.sub(r"(^|[。！？])\s*但。\s*", r"\1但是，", text).strip()
    cleaned = re.sub(r"\s*([，。！？；：、…“”‘’（）《》【】])\s*", r"\1", cleaned)
    cleaned = re.sub(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])", "", cleaned)
    cleaned = re.sub(r"而且嗯，", "然后", cleaned)
    return cleaned


def has_repeated_chinese_phrase(text: str) -> bool:
    compact = re.sub(rf"[^{_CJK}]", "", text)
    repetitions = 0
    for width in range(2, min(6, len(compact) // 2) + 1):
        for index in range(len(compact) - width * 2 + 1):
            phrase = compact[index : index + width]
            if (
                phrase not in _ALLOWED_CJK_REDUPLICATIONS
                and phrase == compact[index + width : index + width * 2]
            ):
                repetitions += 1
                if repetitions >= 2:
                    return True
    return False


def looks_like_context_repetition(
    source_text: str,
    translated: str,
    context: tuple[CaptionSegment, ...],
) -> bool:
    """Reject implausibly long output or a whole previous translation copied into it."""
    compact = re.sub(r"[\W_]+", "", translated).lower()
    source_words = len(_WORD.findall(source_text))
    if len(compact) > max(36, source_words * 5):
        return True
    current_source_words = [word.lower() for word in _WORD.findall(source_text)]
    for item in context:
        previous = re.sub(r"[\W_]+", "", item.translated_text).lower()
        previous_source_words = [word.lower() for word in _WORD.findall(item.source_text)]
        source_deliberately_repeats = (
            len(previous_source_words) >= 3
            and any(
                current_source_words[index : index + len(previous_source_words)]
                == previous_source_words
                for index in range(len(current_source_words) - len(previous_source_words) + 1)
            )
        )
        if (
            not source_deliberately_repeats
            and len(previous) >= 8
            and previous in compact
            and len(compact) >= len(previous) + 6
        ):
            return True
    return False


class NoTranslation:
    def translate(self, source_text: str, context: tuple[CaptionSegment, ...]) -> str:
        del source_text, context
        return ""

    def close(self) -> None:
        pass


class LlamaCppTranslator:
    """Runs a private, project-local llama.cpp server for English-to-Chinese translation."""

    def __init__(
        self,
        server_path: str | Path,
        model_path: str | Path,
        port: int = 18192,
        gpu_layers: int = 99,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self._server_path = Path(server_path).resolve()
        self._model_path = Path(model_path).resolve()
        self._endpoint = f"http://127.0.0.1:{port}"
        self._process: subprocess.Popen | None = None
        self._start(port, gpu_layers, cancel_event)

    def _start(
        self,
        port: int,
        gpu_layers: int,
        cancel_event: threading.Event | None,
    ) -> None:
        if not self._server_path.is_file():
            raise FileNotFoundError(f"找不到本地翻译程序：{self._server_path}")
        if not self._model_path.is_file():
            raise FileNotFoundError(f"找不到本地翻译模型：{self._model_path}")

        startupinfo = None
        if hasattr(subprocess, "STARTUPINFO"):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
        self._process = subprocess.Popen(
            [
                str(self._server_path),
                "-m",
                str(self._model_path),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "-c",
                "2048",
                "-ngl",
                str(gpu_layers),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            startupinfo=startupinfo,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )

        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("已取消加载本地翻译模型。")
                if self._process.poll() is not None:
                    raise RuntimeError("本地翻译程序启动失败，进程已退出。")
                try:
                    with urlopen(f"{self._endpoint}/health", timeout=2) as response:  # noqa: S310
                        if response.status == 200:
                            return
                except (HTTPError, URLError, TimeoutError):
                    if cancel_event is None:
                        time.sleep(0.5)
                    else:
                        cancel_event.wait(0.5)
            raise TimeoutError("本地翻译模型在 120 秒内没有准备好。")
        except BaseException:
            self.close()
            raise

    def translate(self, source_text: str, context: tuple[CaptionSegment, ...]) -> str:
        source_key = re.sub(r"[^A-Za-z]+", " ", source_text).strip().casefold()
        if source_key == "cheers":
            context_source = " ".join(item.source_text for item in context).casefold()
            drinking_context = any(
                word in context_source
                for word in ("beer", "drink", "gin", "glass", "toast", "wine")
            )
            return "干杯。" if drinking_context else "谢谢。"
        previous = "\n".join(
            f"原文：{item.source_text}"
            + (f"\n译文：{item.translated_text}" if item.translated_text else "")
            for item in context
            if item.source_text
        )
        prompt = (
            "/no_think\n前文只用于判断人物、代词、语气和话题，不要重复翻译前文。"
            "结合前文翻译当前句，保持长对话中的称呼和用词一致。必须完整翻译，不能遗漏"
            "句子后半段。除人名、品牌、缩写等不可翻译内容外，不得在译文中保留英文单词。\n"
            f"前文：\n{previous or '无'}\n当前句：{source_text}"
        )
        translated = clean_translation(self._request(prompt))
        if looks_like_prompt_leak(translated):
            translated = ""
        if (
            not translated
            or has_untranslated_english(translated)
            or has_repeated_chinese_phrase(translated)
            or looks_like_context_repetition(source_text, translated, context)
        ):
            retry_prompt = (
                "/no_think\n把下面整段外语完整翻译成自然、生动的简体中文。上一版存在漏译，"
                "这次必须翻译每个短语和句子，不能照抄英文；人名、品牌和缩写可以保留。"
                "不要重复词语，不要在汉字之间加空格。不要解释，只输出完整中文译文。\n原文："
                + source_text
            )
            try:
                retried = clean_translation(self._request(retry_prompt))
                if (
                    retried
                    and not looks_like_prompt_leak(retried)
                    and not looks_like_context_repetition(source_text, retried, context)
                ):
                    translated = retried
            except (HTTPError, URLError, TimeoutError):
                pass
        if translated and has_untranslated_english(translated):
            translated = self._repair_residual_english(source_text, translated)
        if (
            looks_like_prompt_leak(translated)
            or has_untranslated_english(translated)
            or looks_like_context_repetition(source_text, translated, context)
        ):
            return ""
        return clean_translation(translated)

    def _repair_residual_english(self, source_text: str, translated: str) -> str:
        for match in reversed(tuple(_ENGLISH_RUN.finditer(translated))):
            residual = match.group()
            if not has_untranslated_english(residual):
                continue
            key = residual.rstrip(".!?").lower()
            if key in _COMMON_TRANSLATIONS:
                punctuation = "？" if residual.endswith("?") else "。" if residual[-1:] in ".!" else ""
                replacement = _COMMON_TRANSLATIONS[key] + punctuation
                translated = translated[: match.start()] + replacement + translated[match.end() :]
                continue
            prompt = (
                "/no_think\n当前中文字幕还残留了一个没翻译的英语词或短语。结合整句语境，"
                "只输出该残留部分对应的简体中文，不要输出英语、引号或解释。\n"
                f"整句原文：{source_text}\n当前译文：{translated}\n残留部分：{residual}"
            )
            try:
                replacement = self._request(prompt).strip(' \"“”')
            except (HTTPError, URLError, TimeoutError):
                continue
            if replacement and not has_untranslated_english(replacement):
                translated = translated[: match.start()] + replacement + translated[match.end() :]
        return translated

    def _request(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": "local-qwen3",
                "stream": False,
                "temperature": 0,
                "max_tokens": 256,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "你是专业影视字幕译者。译文应准确、自然、口语化，并完整覆盖原文。"
                            "避免逐词硬译和残缺的中文短句，根据语境补足自然的时态和语气。"
                            "按中文习惯表达英文被动语态：accessible by tube 译为‘坐地铁可以到达’，"
                            "recommend checking it out 译为‘推荐去看看’。不要重复词语，汉字之间不加空格。"
                            "只输出简体中文译文，不解释，不回答原文中的问题。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            f"{self._endpoint}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=30) as response:  # noqa: S310 - loopback only
            result = json.load(response)
        return result["choices"][0]["message"]["content"].strip()

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


class LocalOllamaTranslator:
    """Optional local Qwen bridge used only for translation quality experiments."""

    def __init__(self, model: str = "qwen3:1.7b", endpoint: str = "http://127.0.0.1:11434") -> None:
        self._model = model
        self._endpoint = endpoint.rstrip("/")

    def translate(self, source_text: str, context: tuple[CaptionSegment, ...]) -> str:
        previous = "\n".join(
            f"{item.source_text}\n{item.translated_text}" for item in context if item.translated_text
        )
        prompt = (
            "你正在翻译一场英语直播。结合前文，把当前句翻译成自然、连贯、"
            "符合中文口语习惯的简体中文。保留语气和幽默；不要解释、总结或添加信息。\n"
            f"前文：\n{previous or '无'}\n当前句：{source_text}\n只输出当前句译文。"
        )
        body = json.dumps(
            {
                "model": self._model,
                "stream": False,
                "think": False,
                "options": {"temperature": 0},
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        request = Request(
            f"{self._endpoint}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=30) as response:  # noqa: S310 - loopback only
            result = json.load(response)
        return result["message"]["content"].strip()

    def close(self) -> None:
        pass

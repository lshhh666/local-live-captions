from __future__ import annotations

import queue
from dataclasses import dataclass

from .models import CaptionSegment


_OVERLAY_MINIMUM_HEIGHT = 178
_OVERLAY_MAXIMUM_HEIGHT = 360
_CONTENT_VERTICAL_PADDING = 32


def fitted_overlay_height(
    header_height: int,
    source_height: int,
    translation_height: int,
    minimum_height: int = _OVERLAY_MINIMUM_HEIGHT,
    maximum_height: int = _OVERLAY_MAXIMUM_HEIGHT,
) -> int:
    required = header_height + source_height + translation_height + _CONTENT_VERTICAL_PADDING
    return min(maximum_height, max(minimum_height, required))


@dataclass(frozen=True, slots=True)
class OverlaySnapshot:
    sentence_id: int
    source_text: str
    translated_text: str
    is_final: bool


class OverlayState:
    """Keeps only the newest sentence so the window replaces instead of appending."""

    def __init__(self) -> None:
        self._sentence_id = -1
        self._revision = -1
        self._source_text = ""
        self._translated_text = ""
        self._is_final = False

    def update(self, caption: CaptionSegment) -> OverlaySnapshot | None:
        if caption.sentence_id < self._sentence_id:
            return None
        if caption.sentence_id == self._sentence_id and caption.revision < self._revision:
            return None
        if caption.sentence_id > self._sentence_id:
            self._sentence_id = caption.sentence_id
            self._revision = -1
            self._source_text = ""
            self._translated_text = ""
        if caption.revision > self._revision:
            self._revision = caption.revision
            if not caption.translated_text:
                self._translated_text = ""
        if caption.source_text:
            self._source_text = caption.source_text
        if caption.translated_text:
            self._translated_text = caption.translated_text
        self._is_final = caption.is_final
        return self.snapshot()

    def snapshot(self) -> OverlaySnapshot:
        return OverlaySnapshot(
            self._sentence_id,
            self._source_text,
            self._translated_text,
            self._is_final,
        )


class CaptionOverlay:
    """Borderless always-on-top Tk window that runs on Python's main thread."""

    def __init__(self, on_close, font_size: int = 16, opacity: float = 0.90) -> None:
        self._on_close = on_close
        self._font_size = font_size
        self._opacity = opacity
        self._queue: queue.Queue = queue.Queue(maxsize=128)
        self._root = None
        self._header = None
        self._source = None
        self._translation = None
        self._status = None
        self._width = 0
        self._bottom_edge = 0
        self._maximum_height = _OVERLAY_MAXIMUM_HEIGHT
        self._state = OverlayState()

    def start(self) -> None:
        import tkinter as tk

        root = tk.Tk()
        self._root = root
        root.title("Local Live Captions")
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", self._opacity)
        root.configure(bg="#111318")

        width = min(1_000, root.winfo_screenwidth() - 80)
        height = _OVERLAY_MINIMUM_HEIGHT
        x = max(20, (root.winfo_screenwidth() - width) // 2)
        y = max(20, root.winfo_screenheight() - height - 90)
        root.geometry(f"{width}x{height}+{x}+{y}")
        self._width = width
        self._bottom_edge = y + height
        self._maximum_height = min(
            _OVERLAY_MAXIMUM_HEIGHT,
            max(_OVERLAY_MINIMUM_HEIGHT, root.winfo_screenheight() - 80),
        )

        header = tk.Frame(root, bg="#20242c", height=30)
        self._header = header
        header.pack(fill="x")
        title = tk.Label(
            header,
            text="本地实时字幕",
            bg="#20242c",
            fg="#aeb7c6",
            font=("Microsoft YaHei UI", 9),
        )
        title.pack(side="left", padx=12)
        self._status = tk.Label(
            header,
            text="等待语音",
            bg="#20242c",
            fg="#7dd3fc",
            font=("Microsoft YaHei UI", 9),
        )
        self._status.pack(side="right", padx=(0, 12))
        close_button = tk.Button(
            header,
            text="×",
            command=self._request_close,
            bg="#20242c",
            fg="#cbd5e1",
            activebackground="#ef4444",
            activeforeground="white",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 12),
            width=3,
        )
        close_button.pack(side="right")

        self._source = tk.Label(
            root,
            text="播放英文视频后，字幕会显示在这里",
            bg="#111318",
            fg="#f8fafc",
            font=("Segoe UI", self._font_size),
            justify="center",
            wraplength=width - 48,
        )
        self._source.pack(fill="x", padx=24, pady=(14, 4))
        self._translation = tk.Label(
            root,
            text="",
            bg="#111318",
            fg="#facc15",
            font=("Microsoft YaHei UI", self._font_size + 2, "bold"),
            justify="center",
            wraplength=width - 48,
        )
        self._translation.pack(fill="x", padx=24, pady=(2, 12))

        drag = {"x": 0, "y": 0}

        def drag_start(event) -> None:
            drag["x"], drag["y"] = event.x_root, event.y_root

        def drag_move(event) -> None:
            dx, dy = event.x_root - drag["x"], event.y_root - drag["y"]
            new_x, new_y = root.winfo_x() + dx, root.winfo_y() + dy
            root.geometry(f"+{new_x}+{new_y}")
            self._bottom_edge = new_y + root.winfo_height()
            drag["x"], drag["y"] = event.x_root, event.y_root

        for widget in (header, title, self._status):
            widget.bind("<ButtonPress-1>", drag_start)
            widget.bind("<B1-Motion>", drag_move)

        root.protocol("WM_DELETE_WINDOW", self._request_close)
        root.after(50, self._drain)

    def publish(self, caption: CaptionSegment) -> None:
        if self._root is None:
            return
        try:
            self._queue.put_nowait(caption)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            self._queue.put_nowait(caption)

    def stop(self) -> None:
        root, self._root = self._root, None
        if root is None:
            return
        try:
            root.destroy()
        except Exception:
            pass

    def run_until_stopped(self, finished) -> None:
        root = self._root
        if root is None:
            return

        def check_finished() -> None:
            if finished.is_set():
                self.stop()
                return
            if self._root is not None:
                root.after(50, check_finished)

        root.after(50, check_finished)
        root.mainloop()

    def _drain(self) -> None:
        root = self._root
        if root is None:
            return
        try:
            while True:
                item = self._queue.get_nowait()
                self._queue.task_done()
                snapshot = self._state.update(item)
                if snapshot is None:
                    continue
                self._source.config(text=snapshot.source_text or "正在聆听……")
                self._translation.config(text=snapshot.translated_text)
                self._status.config(
                    text="句意已确认 ✓" if snapshot.is_final else "实时修订中 ~",
                    fg="#86efac" if snapshot.is_final else "#7dd3fc",
                )
                self._resize_to_content()
        except queue.Empty:
            pass
        root.after(50, self._drain)

    def _resize_to_content(self) -> None:
        root = self._root
        if root is None or self._header is None:
            return
        root.update_idletasks()
        height = fitted_overlay_height(
            self._header.winfo_reqheight(),
            self._source.winfo_reqheight(),
            self._translation.winfo_reqheight(),
            maximum_height=self._maximum_height,
        )
        x = root.winfo_x()
        y = max(20, self._bottom_edge - height)
        root.geometry(f"{self._width}x{height}+{x}+{y}")

    def _request_close(self) -> None:
        self._on_close()
        self.stop()

from __future__ import annotations

import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from .windows_job import WindowsProcessJob


LANGUAGES = {
    "英语": "en",
    "俄语": "ru",
    "自动识别": "auto",
}

CAPTION_PREFIXES = ("EN~ ", "EN✓ ", "RU~ ", "RU✓ ", "SRC~ ", "SRC✓ ", "ZH~ ", "ZH✓ ")
WHISPER_FILES = (
    "config.json",
    "model.bin",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocabulary.json",
)
LLAMA_RUNTIME_FILES = (
    "ggml-base.dll",
    "ggml-cpu-x64.dll",
    "ggml-vulkan.dll",
    "ggml.dll",
    "libomp.dll",
    "llama-common.dll",
    "llama-server-impl.dll",
    "llama-server.exe",
    "llama.dll",
)
EXPECTED_LARGE_FILE_SIZES = {
    "models/large-v3-turbo/model.bin": 1_617_884_929,
    "models/qwen3-1.7b/Qwen3-1.7B-Q4_K_M.gguf": 1_282_439_264,
}


def is_caption_output(text: str) -> bool:
    return text.startswith(CAPTION_PREFIXES)


def launcher_status_from_line(text: str) -> tuple[str, str] | None:
    """Translate worker diagnostics into short, user-facing progress states."""
    if text.startswith("正在加载本地语音识别模型"):
        return "●  正在加载识别模型  1/3", "#fbbf24"
    if text.startswith("正在加载本地中文翻译模型"):
        return "●  正在加载翻译模型  2/3", "#fbbf24"
    if text.startswith("中文翻译模型已就绪"):
        return "●  正在连接系统声音  3/3", "#fbbf24"
    if text.startswith("监听设备："):
        return "●  字幕运行中", "#4ade80"
    return None


@dataclass(frozen=True, slots=True)
class LauncherOptions:
    language: str = "en"
    font_size: int = 16
    opacity: float = 0.90
    cpu: bool = False


def validate_options(options: LauncherOptions) -> None:
    if options.language not in {"en", "ru", "auto"}:
        raise ValueError("语言选项无效")
    if not 12 <= options.font_size <= 32:
        raise ValueError("字幕字号必须在 12 到 32 之间")
    if not 0.50 <= options.opacity <= 1.00:
        raise ValueError("透明度必须在 50% 到 100% 之间")


def application_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def build_caption_command(
    project_root: Path, options: LauncherOptions, portable: bool = False
) -> list[str]:
    validate_options(options)
    command = (
        [str(project_root / "caption-worker.exe")]
        if portable
        else [
            str(project_root / ".venv" / "Scripts" / "python.exe"),
            "-m",
            "live_caption.cli",
        ]
    )
    command.extend([
        "--model",
        "large-v3-turbo",
        "--model-dir",
        str(project_root / "models"),
        "--language",
        options.language,
        "--translate",
        "llamacpp",
        "--llama-server",
        str(project_root / "runtime" / "llama.cpp" / "llama-server.exe"),
        "--translation-model",
        str(project_root / "models" / "qwen3-1.7b" / "Qwen3-1.7B-Q4_K_M.gguf"),
        "--overlay",
        "--font-size",
        str(options.font_size),
        "--overlay-opacity",
        f"{options.opacity:.2f}",
    ])
    if options.cpu:
        command.extend(("--cpu", "--compute", "int8"))
    return command


def required_paths(project_root: Path, portable: bool = False) -> tuple[Path, ...]:
    paths = [
        project_root
        / ("caption-worker.exe" if portable else ".venv/Scripts/python.exe"),
        project_root / "models" / "qwen3-1.7b" / "Qwen3-1.7B-Q4_K_M.gguf",
    ]
    paths.extend(project_root / "models" / "large-v3-turbo" / name for name in WHISPER_FILES)
    paths.extend(project_root / "runtime" / "llama.cpp" / name for name in LLAMA_RUNTIME_FILES)
    return tuple(paths)


def invalid_large_files(project_root: Path) -> tuple[Path, ...]:
    invalid = []
    for relative, expected_size in EXPECTED_LARGE_FILE_SIZES.items():
        path = project_root / relative
        if path.is_file() and path.stat().st_size != expected_size:
            invalid.append(path)
    return tuple(invalid)


class CaptionLauncher:
    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        self._tk = tk
        self._ttk = ttk
        self._project_root = application_root()
        self._portable = bool(getattr(sys, "frozen", False))
        self._events: queue.Queue = queue.Queue()
        self._process: subprocess.Popen | None = None
        self._process_job: WindowsProcessJob | None = None
        self._close_requested = False
        self._force_stop_job = None
        self._stop_requested = False

        root = tk.Tk()
        self._root = root
        root.title("本地实时字幕")
        root.geometry("600x540")
        root.minsize(560, 540)
        root.configure(bg="#0b1220")
        root.protocol("WM_DELETE_WINDOW", self._request_close)

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("App.TFrame", background="#0b1220")
        style.configure("Card.TFrame", background="#172033")
        style.configure(
            "Title.TLabel",
            background="#0b1220",
            foreground="#f8fafc",
            font=("Microsoft YaHei UI", 21, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background="#0b1220",
            foreground="#94a3b8",
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "Section.TLabel",
            background="#172033",
            foreground="#f1f5f9",
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        style.configure(
            "Card.TLabel",
            background="#172033",
            foreground="#dbe4f0",
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "Muted.TLabel",
            background="#172033",
            foreground="#8290a6",
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Primary.TButton",
            background="#2563eb",
            foreground="#ffffff",
            borderwidth=0,
            focusthickness=0,
            font=("Microsoft YaHei UI", 12, "bold"),
            padding=(18, 12),
        )
        style.map(
            "Primary.TButton",
            background=[("pressed", "#1d4ed8"), ("active", "#3b82f6")],
        )
        style.configure(
            "Danger.TButton",
            background="#dc2626",
            foreground="#ffffff",
            borderwidth=0,
            focusthickness=0,
            font=("Microsoft YaHei UI", 12, "bold"),
            padding=(18, 12),
        )
        style.map(
            "Danger.TButton",
            background=[("pressed", "#b91c1c"), ("active", "#ef4444")],
        )
        style.configure(
            "Link.TButton",
            background="#0b1220",
            foreground="#94a3b8",
            borderwidth=0,
            focusthickness=0,
            font=("Microsoft YaHei UI", 9),
            padding=(0, 5),
        )
        style.map("Link.TButton", foreground=[("active", "#dbeafe")])
        style.configure(
            "Compat.TCheckbutton",
            background="#172033",
            foreground="#e2e8f0",
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.map(
            "Compat.TCheckbutton",
            background=[("active", "#172033")],
            foreground=[("disabled", "#64748b")],
        )
        style.configure(
            "App.Horizontal.TScale",
            background="#172033",
            troughcolor="#334155",
        )
        style.configure(
            "App.TCombobox",
            fieldbackground="#eef2f7",
            background="#eef2f7",
            foreground="#111827",
            arrowcolor="#475569",
            padding=6,
        )

        outer = ttk.Frame(root, padding=(28, 22, 28, 18), style="App.TFrame")
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer, style="App.TFrame")
        header.pack(fill="x")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="本地实时字幕", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self._status = tk.Label(
            header,
            text="●  未运行",
            bg="#172033",
            fg="#94a3b8",
            padx=12,
            pady=6,
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        self._status.grid(row=0, column=1, sticky="e")
        ttk.Label(
            outer,
            text="系统声音实时识别并翻译，全程只在这台电脑上运行",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(1, 16))

        settings = ttk.Frame(outer, padding=(18, 14), style="Card.TFrame")
        settings.pack(fill="x")
        settings.columnconfigure(1, weight=1)
        ttk.Label(settings, text="基本设置", style="Section.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        self._language = tk.StringVar(value="英语")
        self._font_size = tk.IntVar(value=16)
        self._opacity = tk.IntVar(value=90)
        self._cpu = tk.BooleanVar(value=False)

        ttk.Label(settings, text="视频语言", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", pady=7
        )
        self._language_box = ttk.Combobox(
            settings,
            textvariable=self._language,
            values=tuple(LANGUAGES),
            state="readonly",
            style="App.TCombobox",
            width=16,
        )
        self._language_box.grid(row=1, column=1, sticky="ew", padx=(28, 0), pady=7)

        ttk.Label(settings, text="字幕大小", style="Card.TLabel").grid(
            row=2, column=0, sticky="w", pady=7
        )
        font_row = ttk.Frame(settings, style="Card.TFrame")
        font_row.grid(row=2, column=1, sticky="ew", padx=(28, 0), pady=7)
        font_row.columnconfigure(0, weight=1)
        self._font_scale = ttk.Scale(
            font_row,
            from_=12,
            to=32,
            variable=self._font_size,
            orient="horizontal",
            style="App.Horizontal.TScale",
        )
        self._font_scale.grid(row=0, column=0, sticky="ew")
        self._font_value = ttk.Label(font_row, text="16", width=3, style="Card.TLabel")
        self._font_value.grid(row=0, column=1, padx=(12, 0))
        self._font_size.trace_add("write", self._update_values)

        ttk.Label(settings, text="字幕透明度", style="Card.TLabel").grid(
            row=3, column=0, sticky="w", pady=7
        )
        opacity_row = ttk.Frame(settings, style="Card.TFrame")
        opacity_row.grid(row=3, column=1, sticky="ew", padx=(28, 0), pady=7)
        opacity_row.columnconfigure(0, weight=1)
        self._opacity_scale = ttk.Scale(
            opacity_row,
            from_=50,
            to=100,
            variable=self._opacity,
            orient="horizontal",
            style="App.Horizontal.TScale",
        )
        self._opacity_scale.grid(row=0, column=0, sticky="ew")
        self._opacity_value = ttk.Label(
            opacity_row, text="90%", width=5, style="Card.TLabel"
        )
        self._opacity_value.grid(row=0, column=1, padx=(12, 0))
        self._opacity.trace_add("write", self._update_values)

        compatibility = ttk.Frame(outer, padding=(18, 12), style="Card.TFrame")
        compatibility.pack(fill="x", pady=(10, 0))
        self._cpu_check = ttk.Checkbutton(
            compatibility,
            text="兼容模式",
            variable=self._cpu,
            style="Compat.TCheckbutton",
        )
        self._cpu_check.pack(anchor="w")
        ttk.Label(
            compatibility,
            text="没有可用的 NVIDIA 显卡时开启，字幕速度会明显变慢",
            style="Muted.TLabel",
        ).pack(anchor="w", padx=(25, 0), pady=(2, 0))

        self._privacy_label = tk.Label(
            outer,
            text="🔒  默认不保存音频或字幕，停止后自动清空会话内存",
            bg="#0f2a27",
            fg="#9fe3cf",
            anchor="w",
            padx=14,
            pady=9,
            font=("Microsoft YaHei UI", 9),
        )
        self._privacy_label.pack(fill="x", pady=(10, 0))

        buttons = ttk.Frame(outer, style="App.TFrame")
        buttons.pack(fill="x", pady=(14, 4))
        buttons.columnconfigure(0, weight=1)
        self._start_button = ttk.Button(
            buttons,
            text="▶  开始实时字幕",
            command=self.start_caption,
            style="Primary.TButton",
        )
        self._start_button.grid(row=0, column=0, sticky="ew")
        self._stop_button = ttk.Button(
            buttons,
            text="■  停止并清空",
            command=self.stop_caption,
            style="Danger.TButton",
            state="disabled",
        )
        self._stop_button.grid(row=0, column=0, sticky="ew")
        self._stop_button.grid_remove()

        self._details_visible = False
        self._collapsed_geometry: tuple[int, int] | None = None
        self._details_button = ttk.Button(
            outer,
            text="运行详情  ▼",
            command=self._toggle_details,
            style="Link.TButton",
        )
        self._details_button.pack(anchor="center")
        self._details_panel = ttk.Frame(outer, padding=10, style="Card.TFrame")
        self._log = tk.Text(
            self._details_panel,
            width=1,
            height=7,
            bg="#0f172a",
            fg="#cbd5e1",
            insertbackground="white",
            relief="flat",
            wrap="word",
            font=("Consolas", 9),
            state="disabled",
        )
        self._log.pack(fill="both", expand=True)
        self._append_log("准备就绪。选择语言后点击“开始字幕”。")
        root.after(100, self._poll_events)

    def run(self) -> None:
        self._root.mainloop()

    def _toggle_details(self) -> None:
        if self._details_visible:
            self._hide_details()
        else:
            self._show_details()

    def _show_details(self) -> None:
        if self._details_visible:
            return
        self._details_visible = True
        self._details_panel.pack(fill="both", expand=True, pady=(4, 0))
        self._details_button.configure(text="运行详情  ▲")
        if self._root.state() == "normal":
            width = max(self._root.winfo_width(), 560)
            height = max(self._root.winfo_height(), 540)
            self._collapsed_geometry = (width, height)
            self._root.geometry(f"{width}x{max(height + 160, 700)}")

    def _hide_details(self) -> None:
        if not self._details_visible:
            return
        self._details_visible = False
        self._details_panel.pack_forget()
        self._details_button.configure(text="运行详情  ▼")
        if self._root.state() == "normal" and self._collapsed_geometry is not None:
            width, height = self._collapsed_geometry
            self._root.geometry(f"{width}x{height}")
        self._collapsed_geometry = None

    def start_caption(self) -> None:
        from tkinter import messagebox

        if self._process is not None:
            return
        missing = [
            path
            for path in required_paths(self._project_root, self._portable)
            if not path.is_file()
        ]
        if missing:
            messagebox.showerror(
                "缺少运行文件",
                "以下文件不存在：\n\n" + "\n".join(str(path) for path in missing),
                parent=self._root,
            )
            return
        invalid = invalid_large_files(self._project_root)
        if invalid:
            messagebox.showerror(
                "运行文件不完整",
                "以下大模型文件大小不正确，可能复制不完整：\n\n"
                + "\n".join(str(path) for path in invalid),
                parent=self._root,
            )
            return
        options = LauncherOptions(
            LANGUAGES[self._language.get()],
            int(round(self._font_size.get())),
            self._opacity.get() / 100,
            self._cpu.get(),
        )
        command = build_caption_command(self._project_root, options, self._portable)
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "CREATE_NO_WINDOW", 0
        )
        try:
            process = subprocess.Popen(
                command,
                cwd=self._project_root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
            process_job = WindowsProcessJob(process._handle)
        except OSError as error:
            if "process" in locals():
                process.kill()
            messagebox.showerror("无法启动字幕", str(error), parent=self._root)
            return
        self._process = process
        self._process_job = process_job
        self._stop_requested = False
        self._clear_log()
        self._set_running_controls(True)
        self._set_status("●  正在加载本地模型……", "#fbbf24")
        self._append_log("正在启动；首次加载通常需要几秒钟……")
        threading.Thread(target=self._read_output, args=(process,), daemon=True).start()

    def stop_caption(self) -> None:
        process = self._process
        if process is None:
            return
        if self._stop_requested:
            return
        self._stop_requested = True
        self._set_status("●  正在停止并清空……", "#fbbf24")
        self._stop_button.configure(state="disabled")
        try:
            process.stdin.write("\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError, AttributeError):
            pass
        self._force_stop_job = self._root.after(15_000, self._force_stop)

    def _read_output(self, process: subprocess.Popen) -> None:
        if process.stdout is not None:
            for line in process.stdout:
                self._events.put(("line", line.rstrip()))
        self._events.put(("exit", process.wait()))

    def _poll_events(self) -> None:
        try:
            while True:
                event, value = self._events.get_nowait()
                if event == "line":
                    if not is_caption_output(value):
                        self._append_log(value)
                    status = launcher_status_from_line(value)
                    if status is not None:
                        self._set_status(*status)
                elif event == "exit":
                    self._process = None
                    if self._process_job is not None:
                        self._process_job.close()
                        self._process_job = None
                    if self._force_stop_job is not None:
                        self._root.after_cancel(self._force_stop_job)
                        self._force_stop_job = None
                    self._set_running_controls(False)
                    if self._stop_requested or value == 0:
                        self._clear_log()
                        self._set_status("●  已停止并清空", "#9ca3af")
                        self._append_log("已停止。本次会话内容已清空。")
                    else:
                        self._set_status(f"●  启动失败（代码 {value}）", "#f87171")
                        self._append_log(f"字幕进程已退出（代码 {value}）。")
                        self._show_details()
                    self._stop_requested = False
                    if self._close_requested:
                        self._root.destroy()
                        return
                elif event == "force_error":
                    self._append_log(f"强制停止遇到问题：{value}")
        except queue.Empty:
            pass
        self._root.after(100, self._poll_events)

    def _set_running_controls(self, running: bool) -> None:
        if running:
            self._start_button.grid_remove()
            self._stop_button.grid()
            self._stop_button.configure(state="normal")
        else:
            self._stop_button.grid_remove()
            self._start_button.grid()
            self._start_button.configure(state="normal")
        state = "disabled" if running else "readonly"
        self._language_box.configure(state=state)
        self._font_scale.configure(state="disabled" if running else "normal")
        self._opacity_scale.configure(state="disabled" if running else "normal")
        self._cpu_check.configure(state="disabled" if running else "normal")

    def _set_status(self, text: str, color: str) -> None:
        status_backgrounds = {
            "#4ade80": "#113328",
            "#fbbf24": "#3a2d12",
            "#f87171": "#3b1d24",
            "#9ca3af": "#172033",
        }
        self._status.configure(
            text=text,
            foreground=color,
            background=status_backgrounds.get(color, "#172033"),
        )

    def _append_log(self, text: str) -> None:
        if not text:
            return
        self._log.configure(state="normal")
        self._log.insert("end", text + "\n")
        lines = int(self._log.index("end-1c").split(".")[0])
        if lines > 80:
            self._log.delete("1.0", f"{lines - 80}.0")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _clear_log(self) -> None:
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")

    def _update_values(self, *args) -> None:
        del args
        self._font_value.configure(text=str(int(round(self._font_size.get()))))
        self._opacity_value.configure(text=f"{int(round(self._opacity.get()))}%")

    def _request_close(self) -> None:
        if self._process is None:
            self._root.destroy()
            return
        self._close_requested = True
        self.stop_caption()

    def _force_stop(self) -> None:
        self._force_stop_job = None
        process = self._process
        process_job = self._process_job
        if process is None or process.poll() is not None:
            return
        self._append_log("正常停止超时，正在结束本次字幕进程……")
        threading.Thread(
            target=self._terminate_process_tree,
            args=(process, process_job),
            name="caption-force-stop",
            daemon=True,
        ).start()

    def _terminate_process_tree(
        self,
        process: subprocess.Popen,
        process_job: WindowsProcessJob | None,
    ) -> None:
        try:
            result = subprocess.run(
                ("taskkill", "/PID", str(process.pid), "/T", "/F"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                timeout=10,
            )
            if result.returncode != 0 and process.poll() is None:
                raise OSError(f"taskkill 退出代码 {result.returncode}")
        except (OSError, subprocess.TimeoutExpired) as error:
            self._events.put(("force_error", str(error)))
            try:
                if process_job is None:
                    raise OSError("本次会话的 Windows 作业对象不存在")
                process_job.terminate()
            except OSError as job_error:
                self._events.put(("force_error", f"进程树清理失败：{job_error}"))


def main() -> int:
    if "--smoke-test" in sys.argv:
        launcher = CaptionLauncher()
        launcher._root.after(300, launcher._root.destroy)
        launcher.run()
        print("launcher-smoke-test-ok")
        return 0
    CaptionLauncher().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

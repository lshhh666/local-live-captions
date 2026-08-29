from __future__ import annotations

import math
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


def blend_hex_colors(foreground: str, background: str, opacity: float) -> str:
    """Blend two #RRGGBB colors for the launcher's opacity preview."""
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be between 0 and 1")
    foreground_channels = tuple(
        int(foreground[index : index + 2], 16) for index in (1, 3, 5)
    )
    background_channels = tuple(
        int(background[index : index + 2], 16) for index in (1, 3, 5)
    )
    blended = tuple(
        round(front * opacity + back * (1.0 - opacity))
        for front, back in zip(foreground_channels, background_channels)
    )
    return "#" + "".join(f"{channel:02x}" for channel in blended)


def audio_meter_level(rms: float) -> float:
    """Map RMS audio energy to a stable 0..1 logarithmic meter level."""
    if rms <= 0:
        return 0.0
    decibels = 20.0 * math.log10(max(rms, 1e-9))
    return min(1.0, max(0.0, (decibels + 60.0) / 60.0))


@dataclass(frozen=True, slots=True)
class LauncherOptions:
    language: str = "en"
    font_size: int = 16
    opacity: float = 0.90
    cpu: bool = False
    device_index: int | None = None


def validate_options(options: LauncherOptions) -> None:
    if options.language not in {"en", "ru", "auto"}:
        raise ValueError("语言选项无效")
    if not 12 <= options.font_size <= 32:
        raise ValueError("字幕字号必须在 12 到 32 之间")
    if not 0.50 <= options.opacity <= 1.00:
        raise ValueError("透明度必须在 50% 到 100% 之间")
    if options.device_index is not None and options.device_index < 0:
        raise ValueError("音频设备编号无效")


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
    if options.device_index is not None:
        command.extend(("--device-index", str(options.device_index)))
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


def check_installation_files(
    project_root: Path, portable: bool = False
) -> tuple[bool, str]:
    try:
        missing = tuple(
            path
            for path in required_paths(project_root, portable)
            if not path.is_file()
        )
        if missing:
            return False, f"缺少 {len(missing)} 个运行文件"
        invalid = invalid_large_files(project_root)
        if invalid:
            return False, f"有 {len(invalid)} 个模型文件不完整"
        return True, "模型与运行文件完整"
    except OSError as error:
        return False, f"无法检查运行文件：{error}"


def probe_cuda_environment() -> tuple[bool, bool, str]:
    """Check the CTranslate2 CPU runtime and optional CUDA acceleration."""
    try:
        import ctranslate2

        cpu_compute_types = ctranslate2.get_supported_compute_types("cpu")
        if "int8" not in cpu_compute_types:
            return False, False, "CTranslate2 CPU 运行时不支持 int8"
    except Exception as error:
        return False, False, f"CTranslate2 运行库加载失败：{error}"

    try:
        device_count = ctranslate2.get_cuda_device_count()
        if device_count < 1:
            return True, False, "未检测到可用的 NVIDIA CUDA 设备"
        compute_types = ctranslate2.get_supported_compute_types("cuda")
        if "float16" not in compute_types:
            return True, False, "显卡不支持当前字幕模型需要的 float16"
        suffix = "" if device_count == 1 else f"（{device_count} 张）"
        return True, True, f"NVIDIA CUDA 可用{suffix}"
    except Exception as error:
        return True, False, f"CUDA 不可用，CPU 运行时正常：{error}"


class CaptionLauncher:
    def __init__(self) -> None:
        import tkinter as tk

        self._tk = tk
        self._project_root = application_root()
        self._portable = bool(getattr(sys, "frozen", False))
        self._events: queue.Queue = queue.Queue()
        self._process: subprocess.Popen | None = None
        self._process_job: WindowsProcessJob | None = None
        self._close_requested = False
        self._force_stop_job = None
        self._stop_requested = False
        self._audio_devices: dict[int, dict] = {}
        self._selected_device_index: int | None = None
        self._device_scan_complete = False
        self._device_scan_running = False
        self._device_scan_generation = 0
        self._start_validation_running = False
        self._start_attempt = 0
        self._audio_test_running = False
        self._audio_test_cancel = threading.Event()
        self._last_audio_level = 0.0
        self._cuda_available: bool | None = None
        self._runtime_available: bool | None = None
        self._installation_ready: bool | None = None
        self._installation_message = "正在检查运行文件"
        self._environment_check_complete = False
        self._environment_check_running = False
        self._close_deadline_job = None

        root = tk.Tk()
        self._root = root
        root.title("本地实时字幕")
        default_width = min(840, max(480, root.winfo_screenwidth() - 40))
        default_height = min(740, max(480, root.winfo_screenheight() - 80))
        root.geometry(f"{default_width}x{default_height}")
        root.minsize(min(700, default_width), min(650, default_height))
        root.configure(bg="#090e1a")
        root.protocol("WM_DELETE_WINDOW", self._request_close)
        self._colors = {
            "background": "#090e1a",
            "sidebar": "#11182a",
            "surface": "#151f35",
            "surface_alt": "#1b2842",
            "accent": "#6d63f6",
            "accent_hover": "#7c74ff",
            "text": "#f8fafc",
            "muted": "#93a1b7",
            "dim": "#7c8aa0",
        }

        self._language = tk.StringVar(value="英语")
        self._font_size = tk.IntVar(value=16)
        self._opacity = tk.IntVar(value=90)
        self._cpu = tk.BooleanVar(value=False)
        self._controls_running = False

        outer = tk.Frame(root, bg=self._colors["background"])
        outer.pack(fill="both", expand=True)
        outer.grid_rowconfigure(0, weight=1)
        outer.grid_columnconfigure(1, weight=1)

        sidebar = tk.Frame(outer, width=220, bg=self._colors["sidebar"])
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        brand = tk.Frame(sidebar, bg=self._colors["sidebar"])
        brand.pack(fill="x", padx=24, pady=(28, 8))
        tk.Label(
            brand,
            text="译",
            width=3,
            height=1,
            bg=self._colors["accent"],
            fg="#ffffff",
            font=("Microsoft YaHei UI", 18, "bold"),
        ).pack(anchor="w")
        tk.Label(
            brand,
            text="本地实时字幕",
            bg=self._colors["sidebar"],
            fg=self._colors["text"],
            font=("Microsoft YaHei UI", 17, "bold"),
        ).pack(anchor="w", pady=(16, 2))
        tk.Label(
            brand,
            text="LOCAL CAPTIONS",
            bg=self._colors["sidebar"],
            fg="#8278ff",
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w")
        tk.Label(
            sidebar,
            text="让没有字幕的英文、俄语视频\n也能直接看懂。",
            justify="left",
            bg=self._colors["sidebar"],
            fg=self._colors["muted"],
            font=("Microsoft YaHei UI", 10),
        ).pack(anchor="w", padx=24, pady=(12, 24))

        for icon, title, detail in (
            ("◉", "实时修订", "句子说完后自动校正"),
            ("◎", "双语字幕", "原文与中文同时显示"),
            ("◇", "完全本地", "声音不会上传到云端"),
        ):
            feature = tk.Frame(sidebar, bg=self._colors["sidebar"])
            feature.pack(fill="x", padx=24, pady=7)
            tk.Label(
                feature,
                text=icon,
                bg=self._colors["sidebar"],
                fg="#8278ff",
                font=("Segoe UI Symbol", 12, "bold"),
            ).grid(row=0, column=0, rowspan=2, sticky="n", padx=(0, 10))
            tk.Label(
                feature,
                text=title,
                bg=self._colors["sidebar"],
                fg="#e5eaf3",
                font=("Microsoft YaHei UI", 10, "bold"),
            ).grid(row=0, column=1, sticky="w")
            tk.Label(
                feature,
                text=detail,
                bg=self._colors["sidebar"],
                fg=self._colors["dim"],
                font=("Microsoft YaHei UI", 9),
            ).grid(row=1, column=1, sticky="w", pady=(2, 0))

        privacy = tk.Frame(sidebar, bg="#102a28", padx=13, pady=12)
        privacy.pack(side="bottom", fill="x", padx=18, pady=20)
        self._privacy_label = tk.Label(
            privacy,
            text="🔒  隐私模式已开启",
            bg="#102a28",
            fg="#92e4cb",
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        self._privacy_label.pack(anchor="w")
        tk.Label(
            privacy,
            text="默认不保存音频或字幕\n停止后自动清空会话内存",
            justify="left",
            bg="#102a28",
            fg="#6fae9d",
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w", padx=(20, 0), pady=(3, 0))

        main_host = tk.Frame(outer, bg=self._colors["background"])
        main_host.grid(row=0, column=1, sticky="nsew")
        main_host.grid_rowconfigure(0, weight=1)
        main_host.grid_columnconfigure(0, weight=1)
        self._main_canvas = tk.Canvas(
            main_host,
            bg=self._colors["background"],
            highlightthickness=0,
            bd=0,
        )
        self._main_canvas.grid(row=0, column=0, sticky="nsew")
        self._main_scrollbar = tk.Scrollbar(
            main_host,
            orient="vertical",
            command=self._main_canvas.yview,
            width=10,
            troughcolor=self._colors["background"],
            bg="#263550",
            activebackground="#344664",
            relief="flat",
            bd=0,
        )
        self._main_scrollbar.grid(row=0, column=1, sticky="ns")
        self._main_hscrollbar = tk.Scrollbar(
            main_host,
            orient="horizontal",
            command=self._main_canvas.xview,
            width=10,
            troughcolor=self._colors["background"],
            bg="#263550",
            activebackground="#344664",
            relief="flat",
            bd=0,
        )
        self._main_hscrollbar.grid(row=1, column=0, sticky="ew")
        self._main_canvas.configure(
            yscrollcommand=self._main_scrollbar.set,
            xscrollcommand=self._main_hscrollbar.set,
        )
        main = tk.Frame(
            self._main_canvas,
            bg=self._colors["background"],
            padx=28,
            pady=18,
        )
        self._main_content = main
        self._main_canvas_window = self._main_canvas.create_window(
            (0, 0), window=main, anchor="nw"
        )
        main.bind("<Configure>", self._sync_main_scroll_region)
        self._main_canvas.bind("<Configure>", self._resize_main_content)
        root.bind("<MouseWheel>", self._scroll_main_content)

        header = tk.Frame(main, bg=self._colors["background"])
        header.pack(fill="x")
        header.grid_columnconfigure(0, weight=1)
        tk.Label(
            header,
            text="字幕控制台",
            bg=self._colors["background"],
            fg=self._colors["text"],
            font=("Microsoft YaHei UI", 19, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self._status = tk.Label(
            header,
            text="●  未运行",
            bg=self._colors["surface"],
            fg=self._colors["muted"],
            padx=12,
            pady=6,
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        self._status.grid(row=0, column=1, sticky="e")
        tk.Label(
            main,
            text="选择视频语言并调整字幕外观，然后开始监听系统声音",
            bg=self._colors["background"],
            fg=self._colors["muted"],
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w", pady=(2, 14))

        audio_card = tk.Frame(main, bg=self._colors["surface"], padx=18, pady=13)
        audio_card.pack(fill="x")
        audio_header = tk.Frame(audio_card, bg=self._colors["surface"])
        audio_header.pack(fill="x")
        tk.Label(
            audio_header,
            text="系统声音",
            bg=self._colors["surface"],
            fg=self._colors["text"],
            font=("Microsoft YaHei UI", 11, "bold"),
        ).pack(side="left")
        self._environment_status = tk.Label(
            audio_header,
            text="正在进行启动前自检……",
            bg=self._colors["surface"],
            fg="#fbbf24",
            font=("Microsoft YaHei UI", 8, "bold"),
        )
        self._environment_status.pack(side="right")

        device_row = tk.Frame(audio_card, bg=self._colors["surface"])
        device_row.pack(fill="x", pady=(9, 7))
        device_row.grid_columnconfigure(0, weight=1)
        self._device_menu_button = tk.Menubutton(
            device_row,
            text="正在查找音频设备……",
            width=34,
            anchor="w",
            bg=self._colors["surface_alt"],
            activebackground="#243451",
            fg="#dce3ee",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=12,
            pady=7,
            font=("Microsoft YaHei UI", 9),
        )
        self._device_menu_button.grid(row=0, column=0, sticky="ew")
        self._device_menu = tk.Menu(
            self._device_menu_button,
            tearoff=False,
            bg=self._colors["surface_alt"],
            fg="#e5eaf3",
            activebackground=self._colors["accent"],
            activeforeground="#ffffff",
            bd=0,
            font=("Microsoft YaHei UI", 9),
        )
        self._device_menu_button.configure(menu=self._device_menu)
        self._refresh_devices_button = tk.Button(
            device_row,
            text="↻  刷新",
            command=self._refresh_devices,
            bg="#25324a",
            activebackground="#30415f",
            fg=self._colors["muted"],
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=10,
            pady=7,
            font=("Microsoft YaHei UI", 9),
        )
        self._refresh_devices_button.grid(row=0, column=1, padx=(8, 0))
        self._audio_test_button = tk.Button(
            device_row,
            text="检测声音",
            command=self._start_audio_test,
            bg="#3b356f",
            activebackground="#4a4385",
            fg="#d8d5ff",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=10,
            pady=7,
            font=("Microsoft YaHei UI", 9, "bold"),
            state="disabled",
        )
        self._audio_test_button.grid(row=0, column=2, padx=(8, 0))

        meter_row = tk.Frame(audio_card, bg=self._colors["surface"])
        meter_row.pack(fill="x")
        meter_row.grid_columnconfigure(0, weight=1)
        self._audio_meter = tk.Canvas(
            meter_row,
            height=8,
            bg="#263550",
            highlightthickness=0,
            bd=0,
        )
        self._audio_meter.grid(row=0, column=0, sticky="ew", pady=4)
        self._audio_level_bar = self._audio_meter.create_rectangle(
            0, 0, 0, 8, fill="#4ade80", outline=""
        )
        self._audio_meter.bind(
            "<Configure>", lambda event: self._set_audio_level(self._last_audio_level)
        )
        self._audio_test_status = tk.Label(
            meter_row,
            text="选择设备后可检测声音",
            bg=self._colors["surface"],
            fg=self._colors["dim"],
            font=("Microsoft YaHei UI", 8),
        )
        self._audio_test_status.grid(row=0, column=1, sticky="e", padx=(12, 0))

        self._preview = tk.Frame(
            main, bg=self._colors["surface"], padx=18, pady=14
        )
        self._preview.pack(fill="x", pady=(10, 0))
        preview_header = tk.Frame(self._preview, bg=self._colors["surface"])
        preview_header.pack(fill="x")
        preview_title = tk.Label(
            preview_header,
            text="字幕预览",
            bg=self._colors["surface"],
            fg=self._colors["muted"],
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        preview_title.pack(side="left")
        self._preview_hint = tk.Label(
            preview_header,
            text="16 px  ·  90%",
            bg=self._colors["surface"],
            fg=self._colors["dim"],
            font=("Segoe UI", 8),
        )
        self._preview_hint.pack(side="right")
        self._preview_source = tk.Label(
            self._preview,
            text="It's completely fine, but I prefer morning.",
            anchor="w",
            bg=self._colors["surface"],
            fg="#d5dbea",
            font=("Segoe UI", 13),
        )
        self._preview_source.pack(fill="x", pady=(10, 3))
        self._preview_translation = tk.Label(
            self._preview,
            text="这完全没问题，不过我更喜欢早上。",
            anchor="w",
            bg=self._colors["surface"],
            fg="#ffe14f",
            font=("Microsoft YaHei UI", 14, "bold"),
        )
        self._preview_translation.pack(fill="x")
        self._preview_background_widgets = (
            self._preview,
            preview_header,
            preview_title,
            self._preview_hint,
            self._preview_source,
            self._preview_translation,
        )

        settings = tk.Frame(main, bg=self._colors["surface"], padx=18, pady=14)
        settings.pack(fill="x", pady=(10, 0))
        settings.grid_columnconfigure((0, 1), weight=1, uniform="setting")
        tk.Label(
            settings,
            text="识别设置",
            bg=self._colors["surface"],
            fg=self._colors["text"],
            font=("Microsoft YaHei UI", 11, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        tk.Label(
            settings,
            text="视频语言",
            bg=self._colors["surface"],
            fg=self._colors["muted"],
            font=("Microsoft YaHei UI", 9),
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 6))

        languages = tk.Frame(settings, bg=self._colors["surface_alt"])
        languages.grid(row=2, column=0, columnspan=2, sticky="ew")
        languages.grid_columnconfigure((0, 1, 2), weight=1, uniform="language")
        self._language_buttons: dict[str, tk.Radiobutton] = {}
        for column, label in enumerate(LANGUAGES):
            button = tk.Radiobutton(
                languages,
                text=label,
                variable=self._language,
                value=label,
                command=self._language_changed,
                indicatoron=False,
                relief="flat",
                bd=0,
                cursor="hand2",
                takefocus=True,
                highlightthickness=2,
                highlightbackground=self._colors["surface_alt"],
                highlightcolor="#aaa5ff",
                padx=10,
                pady=7,
                font=("Microsoft YaHei UI", 9, "bold"),
            )
            button.grid(row=0, column=column, sticky="ew", padx=2, pady=2)
            self._language_buttons[label] = button

        font_box = tk.Frame(settings, bg=self._colors["surface_alt"], padx=12, pady=10)
        font_box.grid(row=3, column=0, sticky="ew", padx=(0, 5), pady=(10, 0))
        opacity_box = tk.Frame(
            settings, bg=self._colors["surface_alt"], padx=12, pady=10
        )
        opacity_box.grid(row=3, column=1, sticky="ew", padx=(5, 0), pady=(10, 0))
        for box in (font_box, opacity_box):
            box.grid_columnconfigure(0, weight=1)

        tk.Label(
            font_box,
            text="字幕大小",
            bg=self._colors["surface_alt"],
            fg=self._colors["muted"],
            font=("Microsoft YaHei UI", 9),
        ).grid(row=0, column=0, sticky="w")
        self._font_value = tk.Label(
            font_box,
            text="16",
            bg=self._colors["surface_alt"],
            fg="#b8b2ff",
            font=("Segoe UI", 10, "bold"),
        )
        self._font_value.grid(row=0, column=1, sticky="e")
        self._font_scale = tk.Scale(
            font_box,
            from_=12,
            to=32,
            variable=self._font_size,
            orient="horizontal",
            showvalue=False,
            resolution=1,
            bd=0,
            highlightthickness=0,
            sliderrelief="flat",
            sliderlength=16,
            width=8,
            bg=self._colors["surface_alt"],
            troughcolor="#2e3d5b",
            activebackground=self._colors["accent_hover"],
        )
        self._font_scale.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))

        tk.Label(
            opacity_box,
            text="透明度",
            bg=self._colors["surface_alt"],
            fg=self._colors["muted"],
            font=("Microsoft YaHei UI", 9),
        ).grid(row=0, column=0, sticky="w")
        self._opacity_value = tk.Label(
            opacity_box,
            text="90%",
            bg=self._colors["surface_alt"],
            fg="#b8b2ff",
            font=("Segoe UI", 10, "bold"),
        )
        self._opacity_value.grid(row=0, column=1, sticky="e")
        self._opacity_scale = tk.Scale(
            opacity_box,
            from_=50,
            to=100,
            variable=self._opacity,
            orient="horizontal",
            showvalue=False,
            resolution=1,
            bd=0,
            highlightthickness=0,
            sliderrelief="flat",
            sliderlength=16,
            width=8,
            bg=self._colors["surface_alt"],
            troughcolor="#2e3d5b",
            activebackground=self._colors["accent_hover"],
        )
        self._opacity_scale.grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )

        compatibility = tk.Frame(settings, bg=self._colors["surface"], pady=2)
        compatibility.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        compatibility.grid_columnconfigure(0, weight=1)
        compatibility_text = tk.Frame(compatibility, bg=self._colors["surface"])
        compatibility_text.grid(row=0, column=0, sticky="w")
        tk.Label(
            compatibility_text,
            text="CPU 兼容模式",
            bg=self._colors["surface"],
            fg="#e5eaf3",
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(anchor="w")
        tk.Label(
            compatibility_text,
            text="没有 NVIDIA 显卡时开启，字幕速度会变慢",
            bg=self._colors["surface"],
            fg=self._colors["dim"],
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w", pady=(2, 0))
        self._cpu_check = tk.Button(
            compatibility,
            command=self._toggle_cpu_mode,
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=13,
            pady=5,
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        self._cpu_check.grid(row=0, column=1, sticky="e")

        buttons = tk.Frame(main, bg=self._colors["background"])
        buttons.pack(fill="x", pady=(14, 0))
        buttons.columnconfigure(0, weight=1)
        self._start_button = tk.Button(
            buttons,
            text="▶  开始实时字幕",
            command=self.start_caption,
            bg=self._colors["accent"],
            activebackground=self._colors["accent_hover"],
            fg="#ffffff",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            cursor="hand2",
            pady=11,
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        self._start_button.grid(row=0, column=0, sticky="ew")
        self._stop_button = tk.Button(
            buttons,
            text="■  停止并清空",
            command=self.stop_caption,
            bg="#dc3f55",
            activebackground="#ef5267",
            fg="#ffffff",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            cursor="hand2",
            pady=11,
            font=("Microsoft YaHei UI", 11, "bold"),
            state="disabled",
        )
        self._stop_button.grid(row=0, column=0, sticky="ew")
        self._stop_button.grid_remove()

        self._details_visible = False
        self._collapsed_geometry: tuple[int, int] | None = None
        self._details_button = tk.Button(
            main,
            text="运行详情  ▼",
            command=self._toggle_details,
            bg=self._colors["background"],
            activebackground=self._colors["background"],
            fg=self._colors["muted"],
            activeforeground="#d6d3ff",
            relief="flat",
            bd=0,
            cursor="hand2",
            font=("Microsoft YaHei UI", 9),
        )
        self._details_button.pack(anchor="center")
        self._details_panel = tk.Frame(main, bg=self._colors["surface"], padx=10, pady=10)
        self._log = tk.Text(
            self._details_panel,
            width=1,
            height=5,
            bg="#0f172a",
            fg="#cbd5e1",
            insertbackground="white",
            relief="flat",
            wrap="word",
            font=("Consolas", 9),
            state="disabled",
        )
        self._log.pack(fill="both", expand=True)
        self._refresh_language_buttons()
        self._update_preview_language()
        self._refresh_cpu_toggle()
        self._font_size.trace_add("write", self._update_values)
        self._opacity.trace_add("write", self._update_values)
        self._update_values()
        self._append_log("准备就绪。选择语言后点击“开始字幕”。")
        self._refresh_devices()
        self._start_environment_check()
        self._refresh_start_availability()
        root.after(100, self._poll_events)

    def run(self) -> None:
        self._root.mainloop()

    def _resize_main_content(self, event) -> None:
        requested_width = self._main_content.winfo_reqwidth()
        self._main_canvas.itemconfigure(
            self._main_canvas_window,
            width=max(event.width, requested_width),
        )
        self._root.after_idle(self._sync_main_scroll_region)

    def _sync_main_scroll_region(self, event=None) -> None:
        del event
        target_width = max(
            self._main_canvas.winfo_width(),
            self._main_content.winfo_reqwidth(),
        )
        self._main_canvas.itemconfigure(self._main_canvas_window, width=target_width)
        bounds = self._main_canvas.bbox("all")
        if bounds is None:
            return
        self._main_canvas.configure(scrollregion=bounds)
        content_width = bounds[2] - bounds[0]
        content_height = bounds[3] - bounds[1]
        if content_width > self._main_canvas.winfo_width() + 2:
            self._main_hscrollbar.grid()
        else:
            self._main_canvas.xview_moveto(0)
            self._main_hscrollbar.grid_remove()
        if content_height > self._main_canvas.winfo_height() + 2:
            self._main_scrollbar.grid()
        else:
            self._main_canvas.yview_moveto(0)
            self._main_scrollbar.grid_remove()

    def _scroll_main_content(self, event) -> str | None:
        if not self._main_scrollbar.winfo_ismapped() or not event.delta:
            return None
        self._main_canvas.yview_scroll(int(-event.delta / 120), "units")
        return "break"

    def _refresh_devices(self) -> None:
        if (
            self._controls_running
            or self._audio_test_running
            or self._start_validation_running
            or self._close_requested
        ):
            return
        self._device_menu_button.configure(
            text="正在查找音频设备……", state="disabled"
        )
        self._device_scan_complete = False
        self._device_scan_running = True
        self._device_scan_generation += 1
        generation = self._device_scan_generation
        self._refresh_start_availability()
        self._refresh_devices_button.configure(state="disabled")
        self._audio_test_button.configure(state="disabled")
        threading.Thread(
            target=self._load_audio_devices,
            args=(generation,),
            name="launcher-audio-devices",
            daemon=True,
        ).start()

    def _load_audio_devices(self, generation: int) -> None:
        try:
            from .audio_capture import SystemAudioCapture

            devices = SystemAudioCapture.list_loopback_devices()
            self._events.put(("devices", (generation, devices)))
        except Exception as error:
            self._events.put(("device_error", (generation, str(error))))

    def _apply_audio_devices(self, devices: list[dict]) -> None:
        self._device_scan_complete = True
        self._device_scan_running = False
        previous_index = self._selected_device_index
        self._audio_devices = {
            int(device["index"]): dict(device)
            for device in devices
            if "index" in device and device.get("isLoopbackDevice", True)
        }
        self._device_menu.delete(0, "end")
        self._device_menu.add_command(
            label="自动选择默认输出设备",
            command=lambda: self._select_device(None),
        )
        if self._audio_devices:
            self._device_menu.add_separator()
        for index, device in self._audio_devices.items():
            rate = int(device.get("defaultSampleRate", 0))
            label = f"{device.get('name', f'设备 {index}')}  ·  {rate} Hz"
            self._device_menu.add_command(
                label=label,
                command=lambda value=index: self._select_device(value),
            )
        if previous_index not in self._audio_devices:
            previous_index = None
        self._select_device(previous_index)
        self._refresh_devices_button.configure(
            state="disabled" if self._controls_running else "normal"
        )
        self._refresh_start_availability()

    def _select_device(self, device_index: int | None) -> None:
        if (
            self._controls_running
            or self._audio_test_running
            or self._start_validation_running
            or self._close_requested
        ):
            return
        self._selected_device_index = device_index
        if device_index is None:
            text = "自动选择默认输出设备  ▾"
        else:
            device = self._audio_devices.get(device_index)
            text = (
                f"{device.get('name', f'设备 {device_index}')}  ▾"
                if device is not None
                else "自动选择默认输出设备  ▾"
            )
        self._device_menu_button.configure(text=text, state="normal")
        test_state = "normal" if self._audio_devices else "disabled"
        self._audio_test_button.configure(state=test_state)
        self._audio_test_status.configure(
            text="点击“检测声音”并播放视频", fg=self._colors["dim"]
        )
        self._set_audio_level(0.0)

    def _start_audio_test(self) -> None:
        if (
            self._controls_running
            or self._audio_test_running
            or self._start_validation_running
            or not self._audio_devices
            or self._close_requested
        ):
            return
        self._audio_test_running = True
        self._audio_test_cancel.clear()
        self._device_menu_button.configure(state="disabled")
        self._refresh_devices_button.configure(state="disabled")
        self._audio_test_button.configure(state="disabled", text="检测中……")
        self._start_button.configure(state="disabled")
        self._audio_test_status.configure(
            text="正在监听 2 秒，请播放视频", fg="#fbbf24"
        )
        self._set_audio_level(0.0)
        self._refresh_start_availability()
        threading.Thread(
            target=self._run_audio_test,
            name="launcher-audio-test",
            daemon=True,
        ).start()

    def _run_audio_test(self) -> None:
        capture = None
        peak_rms = 0.0
        test_error = None
        cancelled = self._audio_test_cancel.is_set()
        try:
            if not cancelled:
                from .audio_capture import SystemAudioCapture

                def measure(samples) -> None:
                    nonlocal peak_rms
                    rms = (
                        float((samples * samples).mean() ** 0.5)
                        if len(samples)
                        else 0.0
                    )
                    peak_rms = max(peak_rms, rms)
                    self._events.put(("audio_level", audio_meter_level(rms)))

                capture = SystemAudioCapture(measure)
                cancelled = self._audio_test_cancel.is_set()
                if not cancelled:
                    device = capture.start(self._selected_device_index)
                    self._events.put(
                        ("audio_test_device", device.get("name", "系统声音"))
                    )
                    cancelled = self._audio_test_cancel.wait(2.0)
        except Exception as error:
            test_error = str(error)
            cancelled = self._audio_test_cancel.is_set()
        finally:
            if capture is not None:
                try:
                    capture.stop()
                except Exception as error:
                    test_error = test_error or str(error)
        if test_error is not None:
            self._events.put(("audio_test_error", test_error))
        elif cancelled:
            self._events.put(("audio_test_cancelled", None))
        else:
            self._events.put(("audio_test_complete", peak_rms))

    def _finish_audio_test(
        self,
        peak_rms: float | None,
        error: str | None = None,
        cancelled: bool = False,
    ) -> None:
        self._audio_test_running = False
        self._audio_test_button.configure(text="检测声音")
        if cancelled:
            self._audio_test_status.configure(text="声音检测已取消", fg=self._colors["dim"])
            self._set_audio_level(0.0)
        elif error is not None:
            self._audio_test_status.configure(text="声音检测失败", fg="#f87171")
            self._set_audio_level(0.0)
            self._append_log(f"声音检测失败：{error}")
        elif peak_rms is not None and peak_rms >= 0.00003:
            self._audio_test_status.configure(text="检测到系统声音 ✓", fg="#4ade80")
            self._set_audio_level(audio_meter_level(peak_rms))
        else:
            self._audio_test_status.configure(
                text="未检测到声音，请播放视频后重试", fg="#fbbf24"
            )
            self._set_audio_level(0.0)
        self._device_menu_button.configure(
            state="disabled" if self._controls_running else "normal"
        )
        self._refresh_devices_button.configure(
            state="disabled" if self._controls_running else "normal"
        )
        self._audio_test_button.configure(
            state=(
                "normal"
                if self._audio_devices and not self._controls_running
                else "disabled"
            )
        )
        self._refresh_start_availability()

    def _set_audio_level(self, level: float) -> None:
        self._last_audio_level = min(1.0, max(0.0, level))
        width = max(1, self._audio_meter.winfo_width())
        height = max(1, self._audio_meter.winfo_height())
        self._audio_meter.coords(
            self._audio_level_bar,
            0,
            0,
            int(width * self._last_audio_level),
            height,
        )

    def _start_environment_check(self) -> None:
        self._environment_check_complete = False
        self._environment_check_running = True
        threading.Thread(
            target=self._run_environment_check,
            name="launcher-environment-check",
            daemon=True,
        ).start()

    def _run_environment_check(self) -> None:
        try:
            files_ready, files_message = check_installation_files(
                self._project_root, self._portable
            )
            runtime_ready, cuda_ready, cuda_message = probe_cuda_environment()
        except Exception as error:
            files_ready = False
            files_message = f"启动前自检失败：{error}"
            runtime_ready = False
            cuda_ready = False
            cuda_message = "GPU 检测未完成"
        self._events.put(
            (
                "environment",
                (
                    files_ready,
                    files_message,
                    runtime_ready,
                    cuda_ready,
                    cuda_message,
                ),
            )
        )

    def _apply_environment_check(
        self,
        files_ready: bool,
        files_message: str,
        runtime_ready: bool,
        cuda_ready: bool,
        cuda_message: str,
    ) -> None:
        self._installation_ready = files_ready
        self._installation_message = files_message
        self._runtime_available = runtime_ready
        self._cuda_available = cuda_ready
        self._environment_check_complete = True
        self._environment_check_running = False
        self._append_log(f"启动前自检：{files_message}；{cuda_message}。")
        self._refresh_environment_status()
        self._refresh_start_availability()

    def _refresh_start_availability(self) -> None:
        ready = (
            not self._controls_running
            and not self._audio_test_running
            and not self._start_validation_running
            and not self._close_requested
            and self._device_scan_complete
            and bool(self._audio_devices)
            and self._environment_check_complete
            and self._installation_ready is True
            and self._runtime_available is True
        )
        self._start_button.configure(state="normal" if ready else "disabled")

    def _refresh_environment_status(self) -> None:
        if self._installation_ready is False:
            text, color = "运行文件异常", "#f87171"
        elif self._installation_ready is None:
            text, color = "正在进行启动前自检……", "#fbbf24"
        elif self._runtime_available is False:
            text, color = "识别运行库异常", "#f87171"
        elif self._cpu.get():
            text, color = "文件 ✓  ·  CPU 模式", "#fbbf24"
        elif self._cuda_available:
            text, color = "文件 ✓  ·  GPU ✓", "#4ade80"
        elif self._cuda_available is False:
            text, color = "文件 ✓  ·  GPU 不可用", "#f87171"
        else:
            text, color = "文件 ✓  ·  正在检测 GPU", "#fbbf24"
        self._environment_status.configure(text=text, fg=color)

    def _select_language(self, language: str) -> None:
        if self._controls_running or self._start_validation_running:
            return
        self._language.set(language)
        self._refresh_language_buttons()
        self._update_preview_language()

    def _language_changed(self) -> None:
        if self._controls_running or self._start_validation_running:
            return
        self._refresh_language_buttons()
        self._update_preview_language()

    def _update_preview_language(self) -> None:
        examples = {
            "英语": (
                "It's completely fine, but I prefer morning.",
                "这完全没问题，不过我更喜欢早上。",
            ),
            "俄语": (
                "Всё в порядке, но я предпочитаю утро.",
                "一切都没问题，不过我更喜欢早上。",
            ),
            "自动识别": (
                "Language will be detected automatically.",
                "系统将自动识别视频中的语言。",
            ),
        }
        source, translation = examples[self._language.get()]
        self._preview_source.configure(text=source)
        self._preview_translation.configure(text=translation)

    def _refresh_language_buttons(self) -> None:
        selected = self._language.get()
        for language, button in self._language_buttons.items():
            is_selected = language == selected
            if self._controls_running:
                background = "#292c51" if is_selected else "#202b40"
                foreground = "#8e8aa9" if is_selected else self._colors["dim"]
                state = "disabled"
            else:
                background = self._colors["accent"] if is_selected else self._colors["surface_alt"]
                foreground = "#ffffff" if is_selected else self._colors["muted"]
                state = "normal"
            button.configure(
                state=state,
                text=f"✓  {language}" if is_selected else language,
                bg=background,
                selectcolor=background,
                activebackground=(
                    self._colors["accent_hover"]
                    if is_selected
                    else "#243451"
                ),
                fg=foreground,
                activeforeground="#ffffff",
                disabledforeground=foreground,
            )

    def _toggle_cpu_mode(self) -> None:
        if self._controls_running or self._start_validation_running:
            return
        self._cpu.set(not self._cpu.get())
        self._refresh_cpu_toggle()
        self._refresh_environment_status()
        self._refresh_start_availability()

    def _refresh_cpu_toggle(self) -> None:
        enabled = self._cpu.get()
        if self._controls_running:
            self._cpu_check.configure(
                state="disabled",
                text="已开启" if enabled else "关闭",
                bg="#252e42",
                fg=self._colors["dim"],
                disabledforeground=self._colors["dim"],
            )
            return
        self._cpu_check.configure(
            state="normal",
            text="已开启" if enabled else "关闭",
            bg="#3b356f" if enabled else "#25324a",
            activebackground="#4a4385" if enabled else "#30415f",
            fg="#d8d5ff" if enabled else self._colors["muted"],
            activeforeground="#ffffff",
        )

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
            width = max(self._root.winfo_width(), 700)
            height = max(self._root.winfo_height(), 480)
            self._collapsed_geometry = (width, height)
            screen_limit = max(480, self._root.winfo_screenheight() - 60)
            expanded_height = min(max(height + 110, 620), screen_limit)
            self._root.geometry(f"{width}x{expanded_height}")

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
        if self._start_validation_running:
            messagebox.showinfo(
                "正在复验音频设备",
                "请等待设备复验完成。",
                parent=self._root,
            )
            return
        if self._audio_test_running:
            messagebox.showinfo(
                "正在检测声音",
                "请等待声音检测完成后再开始字幕。",
                parent=self._root,
            )
            return
        if not self._environment_check_complete:
            messagebox.showinfo(
                "正在进行启动前自检",
                "运行文件和 GPU 仍在检查，请稍候再试。",
                parent=self._root,
            )
            return
        if not self._device_scan_complete:
            messagebox.showinfo(
                "正在检查音频设备",
                "音频设备仍在刷新，请稍候再试。",
                parent=self._root,
            )
            return
        if not self._audio_devices:
            messagebox.showerror(
                "没有可用的系统声音设备",
                "没有找到 WASAPI 回环设备。请确认扬声器或耳机已连接，然后点击刷新。",
                parent=self._root,
            )
            return
        if self._installation_ready is not True:
            messagebox.showerror(
                "运行文件检查未通过",
                self._installation_message,
                parent=self._root,
            )
            return
        if self._runtime_available is not True:
            messagebox.showerror(
                "识别运行库不可用",
                "CTranslate2 或其本地依赖加载失败，CPU 模式也无法解决。\n\n"
                "请重新解压完整便携版，或重新安装项目依赖。",
                parent=self._root,
            )
            return
        files_ready, files_message = check_installation_files(
            self._project_root, self._portable
        )
        if not files_ready:
            self._installation_ready = False
            self._installation_message = files_message
            self._refresh_environment_status()
            self._refresh_start_availability()
            messagebox.showerror(
                "运行文件复验未通过",
                files_message,
                parent=self._root,
            )
            return
        if self._cuda_available is False and not self._cpu.get():
            use_cpu = messagebox.askyesno(
                "未检测到可用的 NVIDIA 显卡",
                "启动前自检没有发现可用的 CUDA 环境。\n\n"
                "是否改用速度较慢的 CPU 兼容模式？",
                parent=self._root,
            )
            if not use_cpu:
                return
            self._cpu.set(True)
            self._refresh_cpu_toggle()
            self._refresh_environment_status()
        if self._selected_device_index is not None:
            expected = self._audio_devices.get(self._selected_device_index, {})
            self._start_device_validation(
                self._selected_device_index,
                str(expected.get("name", "")),
            )
            return
        self._launch_caption_process()

    def _start_device_validation(self, device_index: int, expected_name: str) -> None:
        self._start_attempt += 1
        attempt = self._start_attempt
        self._start_validation_running = True
        self._refresh_start_availability()
        self._device_menu_button.configure(state="disabled")
        self._refresh_devices_button.configure(state="disabled")
        self._audio_test_button.configure(state="disabled")
        self._set_status("●  正在复验音频设备……", "#fbbf24")
        threading.Thread(
            target=self._run_start_device_validation,
            args=(attempt, device_index, expected_name),
            name="launcher-start-device-validation",
            daemon=True,
        ).start()

    def _run_start_device_validation(
        self,
        attempt: int,
        device_index: int,
        expected_name: str,
    ) -> None:
        try:
            from .audio_capture import SystemAudioCapture

            current_devices = SystemAudioCapture.list_loopback_devices()
            matching = next(
                (
                    device
                    for device in current_devices
                    if int(device.get("index", -1)) == device_index
                    and device.get("isLoopbackDevice", True)
                ),
                None,
            )
            valid = matching is not None and matching.get("name") == expected_name
            self._events.put(("start_device_validation", (attempt, valid, None)))
        except Exception as error:
            self._events.put(
                ("start_device_validation", (attempt, False, str(error)))
            )

    def _finish_start_device_validation(
        self,
        attempt: int,
        valid: bool,
        error: str | None,
    ) -> None:
        from tkinter import messagebox

        self._start_validation_running = False
        if attempt != self._start_attempt or self._close_requested:
            self._maybe_finish_close()
            return
        if error is not None:
            messagebox.showerror(
                "无法复验音频设备",
                f"音频设备状态已变化或读取失败：{error}\n\n请点击刷新后重试。",
                parent=self._root,
            )
            self._set_status("●  未运行", "#9ca3af")
            self._refresh_devices()
            return
        if not valid:
            messagebox.showerror(
                "所选音频设备已变化",
                "原来选择的扬声器或耳机已拔出或编号发生变化。\n\n"
                "请在刷新后重新选择设备。",
                parent=self._root,
            )
            self._set_status("●  未运行", "#9ca3af")
            self._refresh_devices()
            return
        self._launch_caption_process()

    def _launch_caption_process(self) -> None:
        from tkinter import messagebox

        if self._close_requested or self._process is not None:
            return
        options = LauncherOptions(
            language=LANGUAGES[self._language.get()],
            font_size=int(round(self._font_size.get())),
            opacity=self._opacity.get() / 100,
            cpu=self._cpu.get(),
            device_index=self._selected_device_index,
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
            self._restore_idle_after_start_failure()
            return
        self._process = process
        self._process_job = process_job
        self._stop_requested = False
        self._clear_log()
        self._set_running_controls(True)
        self._set_status("●  正在加载本地模型……", "#fbbf24")
        self._append_log("正在启动；首次加载通常需要几秒钟……")
        threading.Thread(target=self._read_output, args=(process,), daemon=True).start()

    def _restore_idle_after_start_failure(self) -> None:
        self._set_status("●  未运行", "#9ca3af")
        self._device_menu_button.configure(
            state="normal" if self._audio_devices else "disabled"
        )
        self._refresh_devices_button.configure(state="normal")
        self._audio_test_button.configure(
            state="normal" if self._audio_devices else "disabled"
        )
        self._refresh_start_availability()

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
                elif event == "devices":
                    generation, devices = value
                    if generation != self._device_scan_generation:
                        continue
                    self._apply_audio_devices(devices)
                    if self._audio_devices:
                        self._append_log(
                            f"检测到 {len(self._audio_devices)} 个系统输出设备。"
                        )
                    else:
                        self._audio_test_status.configure(
                            text="没有找到系统输出设备", fg="#f87171"
                        )
                    if self._maybe_finish_close():
                        return
                elif event == "device_error":
                    generation, error = value
                    if generation != self._device_scan_generation:
                        continue
                    self._device_scan_complete = True
                    self._device_scan_running = False
                    self._audio_devices = {}
                    self._device_menu_button.configure(
                        text="音频设备读取失败", state="disabled"
                    )
                    self._refresh_devices_button.configure(state="normal")
                    self._audio_test_status.configure(
                        text="请点击刷新重试", fg="#f87171"
                    )
                    self._append_log(f"音频设备读取失败：{error}")
                    self._refresh_start_availability()
                    if self._maybe_finish_close():
                        return
                elif event == "audio_level":
                    self._set_audio_level(value)
                elif event == "audio_test_device":
                    self._audio_test_status.configure(
                        text=f"正在检测：{value}", fg="#fbbf24"
                    )
                elif event == "audio_test_complete":
                    self._finish_audio_test(value)
                    if self._maybe_finish_close():
                        return
                elif event == "audio_test_error":
                    self._finish_audio_test(None, value)
                    if self._maybe_finish_close():
                        return
                elif event == "audio_test_cancelled":
                    self._finish_audio_test(None, cancelled=True)
                    if self._maybe_finish_close():
                        return
                elif event == "environment":
                    self._apply_environment_check(*value)
                    if self._maybe_finish_close():
                        return
                elif event == "start_device_validation":
                    self._finish_start_device_validation(*value)
                    if self._maybe_finish_close():
                        return
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
                    if self._maybe_finish_close():
                        return
                elif event == "force_error":
                    self._append_log(f"强制停止遇到问题：{value}")
        except queue.Empty:
            pass
        self._root.after(100, self._poll_events)

    def _set_running_controls(self, running: bool) -> None:
        self._controls_running = running
        if running:
            self._start_button.grid_remove()
            self._stop_button.grid()
            self._stop_button.configure(state="normal")
        else:
            self._stop_button.grid_remove()
            self._start_button.grid()
        self._refresh_language_buttons()
        self._font_scale.configure(state="disabled" if running else "normal")
        self._opacity_scale.configure(state="disabled" if running else "normal")
        self._refresh_cpu_toggle()
        self._device_menu_button.configure(
            state="disabled" if running or not self._audio_devices else "normal"
        )
        self._refresh_devices_button.configure(
            state="disabled" if running else "normal"
        )
        self._audio_test_button.configure(
            state="disabled" if running or not self._audio_devices else "normal"
        )
        self._refresh_start_availability()

    def _set_status(self, text: str, color: str) -> None:
        status_backgrounds = {
            "#4ade80": "#113328",
            "#fbbf24": "#3a2d12",
            "#f87171": "#3b1d24",
            "#9ca3af": self._colors["surface"],
        }
        self._status.configure(
            text=text,
            foreground=color,
            background=status_backgrounds.get(color, self._colors["surface"]),
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
        font_size = int(round(self._font_size.get()))
        opacity = int(round(self._opacity.get()))
        self._font_value.configure(text=str(font_size))
        self._opacity_value.configure(text=f"{opacity}%")
        self._preview_hint.configure(text=f"{font_size} px  ·  {opacity}%")
        preview_background = blend_hex_colors(
            self._colors["surface"],
            self._colors["background"],
            opacity / 100,
        )
        for widget in self._preview_background_widgets:
            widget.configure(bg=preview_background)
        preview_size = min(17, max(11, font_size - 2))
        self._preview_source.configure(font=("Segoe UI", preview_size))
        self._preview_translation.configure(
            font=("Microsoft YaHei UI", preview_size + 1, "bold")
        )

    def _request_close(self) -> None:
        if self._process is None:
            self._close_requested = True
            self._start_attempt += 1
            if self._audio_test_running:
                self._audio_test_cancel.set()
            if (
                self._audio_test_running
                or self._device_scan_running
                or self._environment_check_running
                or self._start_validation_running
            ):
                self._start_button.configure(state="disabled")
                self._audio_test_button.configure(
                    state="disabled", text="正在关闭……"
                )
                self._audio_test_status.configure(
                    text="正在等待后台检查安全结束……", fg="#fbbf24"
                )
                self._schedule_close_deadline()
                return
            self._root.destroy()
            return
        self._close_requested = True
        self.stop_caption()

    def _schedule_close_deadline(self) -> None:
        if self._close_deadline_job is None:
            self._close_deadline_job = self._root.after(
                4_000, self._force_close_background
            )

    def _force_close_background(self) -> None:
        self._close_deadline_job = None
        self._start_attempt += 1
        self._device_scan_generation += 1
        self._root.destroy()

    def _maybe_finish_close(self) -> bool:
        if not self._close_requested:
            return False
        if (
            self._process is not None
            or self._audio_test_running
            or self._device_scan_running
            or self._environment_check_running
            or self._start_validation_running
        ):
            return False
        if self._close_deadline_job is not None:
            self._root.after_cancel(self._close_deadline_job)
            self._close_deadline_job = None
        self._root.destroy()
        return True

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

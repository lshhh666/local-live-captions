from __future__ import annotations

import argparse
import math
import signal
import sys
import threading
import time
from pathlib import Path

from .audio_capture import SystemAudioCapture
from .config import DEFAULT_CONFIG
from .overlay import CaptionOverlay
from .pipeline import CaptionPipeline
from .recognizer import FasterWhisperRecognizer
from .translator import LlamaCppTranslator, LocalOllamaTranslator, NoTranslation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地实时字幕质量验证版")
    parser.add_argument("--list-devices", action="store_true", help="列出回环音频设备")
    parser.add_argument(
        "--capture-test",
        type=float,
        metavar="SECONDS",
        help="只测试系统音频捕获，不加载模型、不保存内容",
    )
    parser.add_argument("--device-index", type=int, help="指定回环设备编号")
    parser.add_argument("--model", default="large-v3-turbo", help="faster-whisper 模型名或路径")
    parser.add_argument("--download-model", action="store_true", help="下载模型后退出")
    parser.add_argument(
        "--model-dir",
        default=str(Path.cwd() / "models"),
        help="模型下载目录，默认是当前项目的 models 文件夹",
    )
    parser.add_argument("--language", default="en", choices=["en", "ru", "auto"])
    parser.add_argument("--compute", default="float16", help="如 float16、int8_float16、int8")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    parser.add_argument("--translate", choices=["none", "llamacpp", "ollama"], default="none")
    parser.add_argument("--ollama-model", default="qwen3:1.7b")
    parser.add_argument("--llama-server", help="本地 llama-server.exe 路径")
    parser.add_argument("--translation-model", help="本地 GGUF 翻译模型路径")
    parser.add_argument("--translation-port", type=int, default=18192)
    parser.add_argument("--overlay", action="store_true", help="显示置顶悬浮字幕窗口")
    parser.add_argument("--font-size", type=int, default=16, help="悬浮字幕英文字号（12-32）")
    parser.add_argument(
        "--overlay-opacity", type=float, default=0.90, help="悬浮字幕透明度（0.50-1.00）"
    )
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    args = build_parser().parse_args()
    if not 12 <= args.font_size <= 32:
        parser_error("--font-size 必须在 12 到 32 之间")
    if not 0.50 <= args.overlay_opacity <= 1.00:
        parser_error("--overlay-opacity 必须在 0.50 到 1.00 之间")
    if args.list_devices:
        for device in SystemAudioCapture.list_loopback_devices():
            print(f"{device['index']:>3}  {device['name']}  {int(device['defaultSampleRate'])}Hz")
        return 0

    if args.capture_test is not None:
        if args.capture_test <= 0 or args.capture_test > 60:
            parser_error("--capture-test 必须在 0 到 60 秒之间")
        sample_count = 0
        square_sum = 0.0

        def measure(samples) -> None:
            nonlocal sample_count, square_sum
            sample_count += len(samples)
            square_sum += float((samples * samples).sum())

        capture = SystemAudioCapture(measure, DEFAULT_CONFIG.sample_rate)
        try:
            device = capture.start(args.device_index)
            print(f"监听设备：{device['name']}")
            print(f"正在测试 {args.capture_test:g} 秒；不会保存音频……")
            time.sleep(args.capture_test)
        finally:
            capture.stop()
        rms = math.sqrt(square_sum / sample_count) if sample_count else 0.0
        print(f"收到样本：{sample_count}，RMS：{rms:.6f}")
        if sample_count == 0:
            print("失败：没有收到系统音频数据。", file=sys.stderr)
            return 4
        if rms < 0.00001:
            print("捕获正常，但当前声音接近静音；测试时请播放视频。")
        else:
            print("系统音频捕获正常。")
        return 0

    model_directory = Path(args.model_dir) / args.model.replace("/", "--")
    incomplete_marker = model_directory / ".download-incomplete"
    if args.download_model:
        from faster_whisper.utils import download_model

        model_directory.mkdir(parents=True, exist_ok=True)
        incomplete_marker.write_text("模型下载尚未完成。\n", encoding="utf-8")
        print(f"正在下载模型 {args.model} 到 {model_directory}")
        try:
            path = download_model(args.model, output_dir=str(model_directory))
        except BaseException:
            print("下载未完成；断点文件已保留，可稍后重试。", file=sys.stderr)
            raise
        incomplete_marker.unlink(missing_ok=True)
        print(f"模型下载完成：{path}")
        return 0

    if incomplete_marker.exists():
        print(
            f"模型尚未下载完成：{model_directory}\n"
            "请运行 scripts\\resume-large-v3-turbo.ps1 后再启动字幕。",
            file=sys.stderr,
        )
        return 5

    selected_model = str(model_directory) if (model_directory / "config.json").exists() else args.model

    finished = threading.Event()

    def request_stop(signum, frame) -> None:
        del signum, frame
        if not finished.is_set():
            print("\n正在停止并清空会话，请稍候……")
        finished.set()

    def stop_on_enter() -> None:
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            return
        if not finished.is_set():
            print("\n收到 Enter，正在停止并清空会话，请稍候……")
        finished.set()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_stop)
    # Start listening before heavyweight model initialization so the launcher's
    # Stop button can cancel a slow startup through stdin.
    threading.Thread(target=stop_on_enter, name="console-stop", daemon=True).start()

    translator = None
    pipeline = None
    capture = None
    overlay = None
    pipeline_started = False
    try:
        device_label = "CPU" if args.cpu else "RTX/CUDA"
        print(f"正在加载本地语音识别模型（{device_label}）；首次使用可能需要下载。")
        recognizer = FasterWhisperRecognizer(
            selected_model,
            language=args.language,
            device="cpu" if args.cpu else "cuda",
            compute_type="int8" if args.cpu and args.compute == "float16" else args.compute,
            download_root=args.model_dir,
        )
        if finished.is_set():
            return 0

        if args.translate == "llamacpp":
            if not args.llama_server or not args.translation_model:
                parser_error("使用 llamacpp 翻译时必须指定 --llama-server 和 --translation-model")
            print("正在加载本地中文翻译模型（只监听本机，不上传内容）……")
            translator = LlamaCppTranslator(
                args.llama_server,
                args.translation_model,
                port=args.translation_port,
                cancel_event=finished,
            )
            print("中文翻译模型已就绪。")
        elif args.translate == "ollama":
            translator = LocalOllamaTranslator(args.ollama_model)
        else:
            translator = NoTranslation()
        if finished.is_set():
            return 0

        source_label = args.language.upper() if args.language != "auto" else "SRC"
        shown_source: dict[int, str] = {}
        shown_translation: dict[int, str] = {}
        overlay = (
            CaptionOverlay(finished.set, args.font_size, args.overlay_opacity)
            if args.overlay
            else None
        )
        if overlay is not None:
            overlay.start()

        def show_caption(caption) -> None:
            if overlay is not None:
                overlay.publish(caption)
            state = "✓" if caption.is_final else "~"
            if (
                caption.source_text
                and not caption.translated_text
                and shown_source.get(caption.sentence_id) != caption.source_text
            ):
                print(f"\n{source_label}{state} {caption.source_text}")
                shown_source[caption.sentence_id] = caption.source_text
            if (
                caption.translated_text
                and shown_translation.get(caption.sentence_id) != caption.translated_text
            ):
                print(f"ZH{state} {caption.translated_text}")
                shown_translation[caption.sentence_id] = caption.translated_text
            if caption.is_final:
                shown_source.pop(caption.sentence_id, None)
                shown_translation.pop(caption.sentence_id, None)

        def show_error(error: Exception) -> None:
            if finished.is_set():
                return
            message = str(error)
            prefix = "性能提示" if "速度跟不上" in message else "字幕处理暂时失败"
            print(f"\n{prefix}（后续字幕仍会继续）：{message}", file=sys.stderr)

        pipeline = CaptionPipeline(DEFAULT_CONFIG, recognizer, translator, show_caption, show_error)
        capture = SystemAudioCapture(pipeline.push_audio, DEFAULT_CONFIG.sample_rate)
        pipeline.start()
        pipeline_started = True
        device = capture.start(args.device_index)
        print(f"监听设备：{device['name']}")
        print("实时修订模式：~ 表示临时字幕，✓ 表示句意已基本确定。")
        print("默认不保存任何内容。按 Ctrl+C 或 Enter 停止并清空会话。")
        if overlay is not None:
            overlay.run_until_stopped(finished)
        else:
            while not finished.wait(0.2):
                pass
    except InterruptedError:
        return 0
    finally:
        if capture is not None:
            try:
                capture.stop()
            except Exception as error:
                print(f"音频设备关闭失败：{error}", file=sys.stderr)
        if pipeline is not None and pipeline_started:
            try:
                pipeline.stop()
            except Exception as error:
                print(f"字幕状态清理失败：{error}", file=sys.stderr)
        elif translator is not None:
            try:
                translator.close()
            except Exception as error:
                print(f"翻译服务关闭失败：{error}", file=sys.stderr)
        if overlay is not None:
            try:
                overlay.stop()
            except Exception as error:
                print(f"字幕窗口关闭失败：{error}", file=sys.stderr)
        print("已停止并清空会话内存。")
    return 0


def parser_error(message: str) -> None:
    raise SystemExit(message)


if __name__ == "__main__":
    sys.exit(main())

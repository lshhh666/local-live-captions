from __future__ import annotations

import queue
import subprocess
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from live_caption.launcher import (
    CaptionLauncher,
    LauncherOptions,
    application_root,
    audio_meter_level,
    blend_hex_colors,
    build_caption_command,
    check_installation_files,
    invalid_large_files,
    is_caption_output,
    launcher_status_from_line,
    probe_cuda_environment,
    required_paths,
    validate_options,
)


class LauncherCommandTests(unittest.TestCase):
    def test_builds_private_local_caption_command(self) -> None:
        root = Path(r"D:\translate")
        command = build_caption_command(
            root, LauncherOptions(language="ru", font_size=20, opacity=0.75)
        )
        self.assertEqual(str(root / ".venv" / "Scripts" / "python.exe"), command[0])
        self.assertIn("live_caption.cli", command)
        self.assertEqual("ru", command[command.index("--language") + 1])
        self.assertEqual("20", command[command.index("--font-size") + 1])
        self.assertEqual("0.75", command[command.index("--overlay-opacity") + 1])
        self.assertIn("--overlay", command)
        self.assertNotIn("--save", command)

    def test_selected_audio_device_is_forwarded_to_worker(self) -> None:
        command = build_caption_command(
            Path(r"D:\translate"), LauncherOptions(device_index=26)
        )
        self.assertEqual("26", command[command.index("--device-index") + 1])

    def test_cpu_mode_uses_int8(self) -> None:
        command = build_caption_command(Path("X:/app"), LauncherOptions(cpu=True))
        self.assertIn("--cpu", command)
        self.assertEqual("int8", command[command.index("--compute") + 1])

    def test_portable_build_uses_packaged_worker(self) -> None:
        root = Path(r"D:\portable")
        command = build_caption_command(root, LauncherOptions(), portable=True)
        self.assertEqual(str(root / "caption-worker.exe"), command[0])
        self.assertNotIn("live_caption.cli", command)

    def test_frozen_application_root_is_executable_directory(self) -> None:
        executable = r"D:\portable\本地实时字幕.exe"
        with patch("live_caption.launcher.sys.frozen", True, create=True), patch(
            "live_caption.launcher.sys.executable", executable
        ):
            self.assertEqual(Path(r"D:\portable"), application_root())

    def test_rejects_unsafe_visual_values(self) -> None:
        with self.assertRaises(ValueError):
            validate_options(LauncherOptions(font_size=40))
        with self.assertRaises(ValueError):
            validate_options(LauncherOptions(opacity=0.2))
        with self.assertRaises(ValueError):
            validate_options(LauncherOptions(device_index=-1))

    def test_required_paths_are_all_project_local(self) -> None:
        root = Path(r"D:\translate")
        paths = required_paths(root)
        self.assertGreater(len(paths), 10)
        self.assertTrue(all(str(path).startswith(str(root)) for path in paths))

    def test_portable_required_paths_include_worker(self) -> None:
        root = Path(r"D:\portable")
        paths = required_paths(root, portable=True)
        self.assertEqual(root / "caption-worker.exe", paths[0])

    def test_detects_partial_large_model_copy(self) -> None:
        root = MagicMock(spec=Path)
        first = MagicMock()
        first.is_file.return_value = True
        first.stat.return_value.st_size = 1
        second = MagicMock()
        second.is_file.return_value = False
        root.__truediv__.side_effect = (first, second)
        self.assertEqual((first,), invalid_large_files(root))

    def test_caption_lines_are_excluded_from_operational_log(self) -> None:
        self.assertTrue(is_caption_output("EN~ partial sentence"))
        self.assertTrue(is_caption_output("ZH✓ 完整译文"))
        self.assertTrue(is_caption_output("RU✓ final sentence"))
        self.assertFalse(is_caption_output("监听设备：Speakers"))

    def test_worker_output_maps_to_clear_startup_progress(self) -> None:
        self.assertEqual(
            ("●  正在加载识别模型  1/3", "#fbbf24"),
            launcher_status_from_line("正在加载本地语音识别模型（RTX/CUDA）"),
        )
        self.assertEqual(
            ("●  正在加载翻译模型  2/3", "#fbbf24"),
            launcher_status_from_line("正在加载本地中文翻译模型……"),
        )
        self.assertEqual(
            ("●  正在连接系统声音  3/3", "#fbbf24"),
            launcher_status_from_line("中文翻译模型已就绪。"),
        )
        self.assertEqual(
            ("●  字幕运行中", "#4ade80"),
            launcher_status_from_line("监听设备：扬声器 [Loopback]"),
        )
        self.assertIsNone(launcher_status_from_line("普通运行日志"))

    def test_opacity_preview_blends_colors(self) -> None:
        self.assertEqual("#808080", blend_hex_colors("#ffffff", "#000000", 0.5))
        self.assertEqual("#ffffff", blend_hex_colors("#ffffff", "#000000", 1.0))
        with self.assertRaises(ValueError):
            blend_hex_colors("#ffffff", "#000000", 1.1)

    def test_audio_meter_uses_logarithmic_scale(self) -> None:
        self.assertEqual(0.0, audio_meter_level(0.0))
        self.assertEqual(0.0, audio_meter_level(0.001))
        self.assertAlmostEqual(1 / 3, audio_meter_level(0.01), places=6)
        self.assertEqual(1.0, audio_meter_level(1.0))

    def test_installation_check_reports_missing_and_complete_files(self) -> None:
        present = MagicMock()
        present.is_file.return_value = True
        missing = MagicMock()
        missing.is_file.return_value = False
        with patch(
            "live_caption.launcher.required_paths", return_value=(present, missing)
        ):
            self.assertEqual(
                (False, "缺少 1 个运行文件"),
                check_installation_files(Path("X:/app")),
            )
        with patch(
            "live_caption.launcher.required_paths", return_value=(present,)
        ), patch("live_caption.launcher.invalid_large_files", return_value=()):
            self.assertEqual(
                (True, "模型与运行文件完整"),
                check_installation_files(Path("X:/app")),
            )

    def test_installation_check_translates_filesystem_errors(self) -> None:
        unreadable = MagicMock()
        unreadable.is_file.side_effect = OSError(5, "access denied")
        with patch(
            "live_caption.launcher.required_paths", return_value=(unreadable,)
        ):
            ready, message = check_installation_files(Path("X:/app"))
        self.assertFalse(ready)
        self.assertIn("无法检查运行文件", message)

    def test_cuda_probe_accepts_float16_capable_device(self) -> None:
        fake_ctranslate2 = MagicMock()
        fake_ctranslate2.get_cuda_device_count.return_value = 1
        fake_ctranslate2.get_supported_compute_types.return_value = {
            "float16",
            "int8_float16",
            "int8",
        }
        with patch.dict("sys.modules", {"ctranslate2": fake_ctranslate2}):
            self.assertEqual(
                (True, True, "NVIDIA CUDA 可用"), probe_cuda_environment()
            )

    def test_cuda_probe_distinguishes_broken_runtime_from_no_gpu(self) -> None:
        with patch.dict("sys.modules", {"ctranslate2": None}):
            runtime_ready, cuda_ready, message = probe_cuda_environment()
        self.assertFalse(runtime_ready)
        self.assertFalse(cuda_ready)
        self.assertIn("CTranslate2 运行库加载失败", message)

        fake_ctranslate2 = MagicMock()
        fake_ctranslate2.get_supported_compute_types.return_value = {"int8"}
        fake_ctranslate2.get_cuda_device_count.return_value = 0
        with patch.dict("sys.modules", {"ctranslate2": fake_ctranslate2}):
            self.assertEqual(
                (True, False, "未检测到可用的 NVIDIA CUDA 设备"),
                probe_cuda_environment(),
            )

    def test_cpu_probe_rejects_nonempty_compute_types_without_int8(self) -> None:
        fake_ctranslate2 = MagicMock()
        fake_ctranslate2.get_supported_compute_types.return_value = {"float32"}
        with patch.dict("sys.modules", {"ctranslate2": fake_ctranslate2}):
            runtime_ready, cuda_ready, message = probe_cuda_environment()
        self.assertFalse(runtime_ready)
        self.assertFalse(cuda_ready)
        self.assertIn("不支持 int8", message)

    def test_device_refresh_populates_loopback_menu(self) -> None:
        launcher = object.__new__(CaptionLauncher)
        launcher._selected_device_index = None
        launcher._controls_running = False
        launcher._audio_test_running = False
        launcher._start_validation_running = False
        launcher._close_requested = False
        launcher._device_menu = MagicMock()
        launcher._device_menu_button = MagicMock()
        launcher._refresh_devices_button = MagicMock()
        launcher._audio_test_button = MagicMock()
        launcher._audio_test_status = MagicMock()
        launcher._set_audio_level = MagicMock()
        launcher._refresh_start_availability = MagicMock()
        launcher._colors = {"dim": "gray"}

        launcher._apply_audio_devices(
            [
                {
                    "index": 26,
                    "name": "USB Speakers [Loopback]",
                    "defaultSampleRate": 48_000,
                    "isLoopbackDevice": True,
                }
            ]
        )

        self.assertTrue(launcher._device_scan_complete)
        self.assertIn(26, launcher._audio_devices)
        launcher._audio_test_button.configure.assert_called_with(state="normal")
        launcher._device_menu.add_separator.assert_called_once_with()

    def test_audio_test_completion_restores_idle_controls(self) -> None:
        launcher = object.__new__(CaptionLauncher)
        launcher._audio_test_running = True
        launcher._controls_running = False
        launcher._audio_devices = {26: {"name": "Speakers"}}
        launcher._start_validation_running = False
        launcher._close_requested = False
        launcher._audio_test_button = MagicMock()
        launcher._audio_test_status = MagicMock()
        launcher._device_menu_button = MagicMock()
        launcher._refresh_devices_button = MagicMock()
        launcher._start_button = MagicMock()
        launcher._set_audio_level = MagicMock()
        launcher._device_scan_complete = True
        launcher._environment_check_complete = True
        launcher._installation_ready = True
        launcher._runtime_available = True

        launcher._finish_audio_test(0.01)

        self.assertFalse(launcher._audio_test_running)
        launcher._audio_test_status.configure.assert_called_with(
            text="检测到系统声音 ✓", fg="#4ade80"
        )
        launcher._audio_test_button.configure.assert_called_with(state="normal")
        launcher._start_button.configure.assert_called_with(state="normal")
        launcher._set_audio_level.assert_called_once_with(audio_meter_level(0.01))

    def test_environment_check_updates_cached_capabilities(self) -> None:
        launcher = object.__new__(CaptionLauncher)
        launcher._append_log = MagicMock()
        launcher._refresh_environment_status = MagicMock()
        launcher._refresh_start_availability = MagicMock()

        launcher._apply_environment_check(
            True,
            "模型与运行文件完整",
            True,
            True,
            "NVIDIA CUDA 可用",
        )

        self.assertTrue(launcher._installation_ready)
        self.assertTrue(launcher._runtime_available)
        self.assertTrue(launcher._cuda_available)
        self.assertTrue(launcher._environment_check_complete)
        launcher._refresh_environment_status.assert_called_once_with()
        launcher._refresh_start_availability.assert_called_once_with()

    def test_environment_worker_always_queues_a_terminal_result(self) -> None:
        launcher = object.__new__(CaptionLauncher)
        launcher._events = queue.Queue()
        launcher._project_root = Path("X:/app")
        launcher._portable = False
        with patch(
            "live_caption.launcher.check_installation_files",
            side_effect=RuntimeError("unexpected probe failure"),
        ):
            launcher._run_environment_check()

        event, values = launcher._events.get_nowait()
        self.assertEqual("environment", event)
        self.assertFalse(values[0])
        self.assertIn("启动前自检失败", values[1])

    def test_language_selection_updates_preview_only_while_idle(self) -> None:
        launcher = object.__new__(CaptionLauncher)
        launcher._controls_running = False
        launcher._start_validation_running = False
        launcher._language = MagicMock()
        launcher._refresh_language_buttons = MagicMock()
        launcher._update_preview_language = MagicMock()

        launcher._select_language("俄语")
        launcher._language.set.assert_called_once_with("俄语")
        launcher._refresh_language_buttons.assert_called_once_with()
        launcher._update_preview_language.assert_called_once_with()

        launcher._controls_running = True
        launcher._language.reset_mock()
        launcher._select_language("自动识别")
        launcher._language.set.assert_not_called()

    def test_russian_language_selection_updates_preview_copy(self) -> None:
        launcher = object.__new__(CaptionLauncher)
        launcher._language = MagicMock()
        launcher._language.get.return_value = "俄语"
        launcher._preview_source = MagicMock()
        launcher._preview_translation = MagicMock()

        launcher._update_preview_language()
        launcher._preview_source.configure.assert_called_once_with(
            text="Всё в порядке, но я предпочитаю утро."
        )
        launcher._preview_translation.configure.assert_called_once_with(
            text="一切都没问题，不过我更喜欢早上。"
        )

    def test_cpu_toggle_updates_only_while_idle(self) -> None:
        launcher = object.__new__(CaptionLauncher)
        launcher._controls_running = False
        launcher._start_validation_running = False
        launcher._cpu = MagicMock()
        launcher._cpu.get.return_value = False
        launcher._refresh_cpu_toggle = MagicMock()
        launcher._refresh_environment_status = MagicMock()
        launcher._refresh_start_availability = MagicMock()

        launcher._toggle_cpu_mode()
        launcher._cpu.set.assert_called_once_with(True)
        launcher._refresh_cpu_toggle.assert_called_once_with()

        launcher._controls_running = True
        launcher._cpu.reset_mock()
        launcher._toggle_cpu_mode()
        launcher._cpu.set.assert_not_called()

    def test_running_controls_swap_the_primary_action_and_lock_settings(self) -> None:
        launcher = object.__new__(CaptionLauncher)
        launcher._start_button = MagicMock()
        launcher._stop_button = MagicMock()
        launcher._font_scale = MagicMock()
        launcher._opacity_scale = MagicMock()
        launcher._refresh_language_buttons = MagicMock()
        launcher._refresh_cpu_toggle = MagicMock()
        launcher._device_menu_button = MagicMock()
        launcher._refresh_devices_button = MagicMock()
        launcher._audio_test_button = MagicMock()
        launcher._audio_devices = {}
        launcher._audio_test_running = False
        launcher._start_validation_running = False
        launcher._close_requested = False
        launcher._device_scan_complete = True
        launcher._environment_check_complete = True
        launcher._installation_ready = True
        launcher._runtime_available = True

        launcher._set_running_controls(True)
        self.assertTrue(launcher._controls_running)
        launcher._start_button.grid_remove.assert_called_once_with()
        launcher._stop_button.grid.assert_called_once_with()
        launcher._stop_button.configure.assert_called_with(state="normal")
        launcher._font_scale.configure.assert_called_with(state="disabled")
        launcher._opacity_scale.configure.assert_called_with(state="disabled")

        launcher._set_running_controls(False)
        self.assertFalse(launcher._controls_running)
        launcher._stop_button.grid_remove.assert_called_once_with()
        launcher._start_button.grid.assert_called_once_with()
        launcher._start_button.configure.assert_called_with(state="disabled")

    def test_start_stays_disabled_until_both_preflights_finish(self) -> None:
        launcher = object.__new__(CaptionLauncher)
        launcher._start_button = MagicMock()
        launcher._controls_running = False
        launcher._audio_test_running = False
        launcher._start_validation_running = False
        launcher._close_requested = False
        launcher._device_scan_complete = True
        launcher._audio_devices = {26: {"name": "Speakers"}}
        launcher._environment_check_complete = False

        launcher._refresh_start_availability()
        launcher._start_button.configure.assert_called_with(state="disabled")

        launcher._environment_check_complete = True
        launcher._installation_ready = True
        launcher._runtime_available = True
        launcher._refresh_start_availability()
        launcher._start_button.configure.assert_called_with(state="normal")

        launcher._installation_ready = False
        launcher._refresh_start_availability()
        launcher._start_button.configure.assert_called_with(state="disabled")

    def test_close_during_audio_test_waits_for_capture_cleanup(self) -> None:
        launcher = object.__new__(CaptionLauncher)
        launcher._process = None
        launcher._audio_test_running = True
        launcher._close_requested = False
        launcher._device_scan_running = False
        launcher._environment_check_running = False
        launcher._start_validation_running = False
        launcher._close_deadline_job = None
        launcher._start_attempt = 0
        launcher._audio_test_cancel = threading.Event()
        launcher._root = MagicMock()
        launcher._start_button = MagicMock()
        launcher._audio_test_button = MagicMock()
        launcher._audio_test_status = MagicMock()

        launcher._request_close()

        self.assertTrue(launcher._close_requested)
        self.assertTrue(launcher._audio_test_cancel.is_set())
        launcher._root.destroy.assert_not_called()

    def test_cancelled_audio_test_stops_capture_before_completion(self) -> None:
        launcher = object.__new__(CaptionLauncher)
        launcher._events = queue.Queue()
        launcher._selected_device_index = 26
        launcher._audio_test_cancel = threading.Event()
        launcher._audio_test_cancel.set()
        capture_class = MagicMock()

        with patch(
            "live_caption.audio_capture.SystemAudioCapture", capture_class
        ):
            launcher._run_audio_test()

        capture_class.assert_not_called()
        events = []
        while not launcher._events.empty():
            events.append(launcher._events.get_nowait()[0])
        self.assertIn("audio_test_cancelled", events)
        self.assertNotIn("audio_test_complete", events)

    def test_audio_test_cancelled_while_open_always_stops_capture(self) -> None:
        launcher = object.__new__(CaptionLauncher)
        launcher._events = queue.Queue()
        launcher._selected_device_index = 26
        launcher._audio_test_cancel = threading.Event()
        capture = MagicMock()

        def start_and_cancel(device_index):
            self.assertEqual(26, device_index)
            launcher._audio_test_cancel.set()
            return {"name": "Speakers"}

        capture.start.side_effect = start_and_cancel
        with patch(
            "live_caption.audio_capture.SystemAudioCapture", return_value=capture
        ):
            launcher._run_audio_test()

        capture.stop.assert_called_once_with()
        events = []
        while not launcher._events.empty():
            events.append(launcher._events.get_nowait()[0])
        self.assertIn("audio_test_cancelled", events)

    def test_close_during_device_scan_waits_for_terminal_event(self) -> None:
        launcher = object.__new__(CaptionLauncher)
        launcher._process = None
        launcher._audio_test_running = False
        launcher._device_scan_running = True
        launcher._environment_check_running = False
        launcher._start_validation_running = False
        launcher._close_requested = False
        launcher._close_deadline_job = None
        launcher._start_attempt = 0
        launcher._audio_test_cancel = threading.Event()
        launcher._root = MagicMock()
        launcher._root.after.return_value = "close-deadline"
        launcher._start_button = MagicMock()
        launcher._audio_test_button = MagicMock()
        launcher._audio_test_status = MagicMock()

        launcher._request_close()

        launcher._root.destroy.assert_not_called()
        launcher._device_scan_running = False
        self.assertTrue(launcher._maybe_finish_close())
        launcher._root.after_cancel.assert_called_once_with("close-deadline")
        launcher._root.destroy.assert_called_once_with()

    def test_manual_device_validation_runs_in_background(self) -> None:
        launcher = object.__new__(CaptionLauncher)
        launcher._start_attempt = 0
        launcher._start_validation_running = False
        launcher._start_button = MagicMock()
        launcher._controls_running = False
        launcher._audio_test_running = False
        launcher._device_scan_complete = True
        launcher._audio_devices = {26: {"name": "Speakers"}}
        launcher._environment_check_complete = True
        launcher._installation_ready = True
        launcher._runtime_available = True
        launcher._close_requested = False
        launcher._device_menu_button = MagicMock()
        launcher._refresh_devices_button = MagicMock()
        launcher._audio_test_button = MagicMock()
        launcher._set_status = MagicMock()

        with patch("live_caption.launcher.threading.Thread") as thread:
            launcher._start_device_validation(26, "Speakers")

        self.assertTrue(launcher._start_validation_running)
        thread.assert_called_once()
        thread.return_value.start.assert_called_once_with()

    def test_device_validation_result_is_generation_scoped(self) -> None:
        launcher = object.__new__(CaptionLauncher)
        launcher._start_validation_running = True
        launcher._start_attempt = 2
        launcher._close_requested = False
        launcher._launch_caption_process = MagicMock()
        launcher._maybe_finish_close = MagicMock()

        launcher._finish_start_device_validation(1, True, None)

        launcher._launch_caption_process.assert_not_called()

    def test_background_close_deadline_invalidates_late_results(self) -> None:
        launcher = object.__new__(CaptionLauncher)
        launcher._close_deadline_job = "deadline"
        launcher._start_attempt = 3
        launcher._device_scan_generation = 4
        launcher._root = MagicMock()

        launcher._force_close_background()

        self.assertEqual(4, launcher._start_attempt)
        self.assertEqual(5, launcher._device_scan_generation)
        launcher._root.destroy.assert_called_once_with()

    def test_start_recheck_handles_file_error_without_spawning(self) -> None:
        launcher = object.__new__(CaptionLauncher)
        launcher._process = None
        launcher._audio_test_running = False
        launcher._start_validation_running = False
        launcher._environment_check_complete = True
        launcher._device_scan_complete = True
        launcher._audio_devices = {26: {"name": "Speakers"}}
        launcher._installation_ready = True
        launcher._installation_message = "ready"
        launcher._runtime_available = True
        launcher._project_root = Path("X:/app")
        launcher._portable = False
        launcher._root = MagicMock()
        launcher._refresh_environment_status = MagicMock()
        launcher._refresh_start_availability = MagicMock()

        with patch(
            "live_caption.launcher.check_installation_files",
            return_value=(False, "无法检查运行文件：access denied"),
        ), patch("tkinter.messagebox.showerror") as show_error, patch(
            "live_caption.launcher.subprocess.Popen"
        ) as popen:
            launcher.start_caption()

        popen.assert_not_called()
        show_error.assert_called_once()
        self.assertFalse(launcher._installation_ready)

    def test_failed_process_start_restores_controls_after_device_validation(self) -> None:
        launcher = object.__new__(CaptionLauncher)
        launcher._close_requested = False
        launcher._process = None
        launcher._project_root = Path("X:/app")
        launcher._portable = False
        launcher._language = MagicMock()
        launcher._language.get.return_value = "英语"
        launcher._font_size = MagicMock()
        launcher._font_size.get.return_value = 16
        launcher._opacity = MagicMock()
        launcher._opacity.get.return_value = 90
        launcher._cpu = MagicMock()
        launcher._cpu.get.return_value = False
        launcher._selected_device_index = 26
        launcher._controls_running = False
        launcher._audio_test_running = False
        launcher._start_validation_running = False
        launcher._device_scan_complete = True
        launcher._audio_devices = {26: {"name": "Speakers"}}
        launcher._environment_check_complete = True
        launcher._installation_ready = True
        launcher._runtime_available = True
        launcher._root = MagicMock()
        launcher._start_button = MagicMock()
        launcher._device_menu_button = MagicMock()
        launcher._refresh_devices_button = MagicMock()
        launcher._audio_test_button = MagicMock()
        launcher._set_status = MagicMock()

        with patch(
            "live_caption.launcher.build_caption_command", return_value=["worker.exe"]
        ), patch(
            "live_caption.launcher.subprocess.Popen", side_effect=OSError("blocked")
        ), patch("tkinter.messagebox.showerror") as show_error:
            launcher._launch_caption_process()

        show_error.assert_called_once()
        launcher._set_status.assert_called_with("●  未运行", "#9ca3af")
        launcher._device_menu_button.configure.assert_called_with(state="normal")
        launcher._refresh_devices_button.configure.assert_called_with(state="normal")
        launcher._audio_test_button.configure.assert_called_with(state="normal")
        launcher._start_button.configure.assert_called_with(state="normal")

    def test_detail_panel_restores_the_previous_window_geometry(self) -> None:
        launcher = object.__new__(CaptionLauncher)
        launcher._details_visible = False
        launcher._collapsed_geometry = None
        launcher._details_panel = MagicMock()
        launcher._details_button = MagicMock()
        launcher._root = MagicMock()
        launcher._root.state.return_value = "normal"
        launcher._root.winfo_width.return_value = 800
        launcher._root.winfo_height.return_value = 650
        launcher._root.winfo_screenheight.return_value = 900

        launcher._show_details()
        launcher._root.geometry.assert_called_with("800x760")
        launcher._hide_details()
        launcher._root.geometry.assert_called_with("800x650")

    def test_failed_taskkill_falls_back_to_session_job(self) -> None:
        class FakeProcess:
            pid = 1234

            @staticmethod
            def poll():
                return None

        class FakeJob:
            terminated = False

            def terminate(self):
                self.terminated = True

        launcher = object.__new__(CaptionLauncher)
        launcher._events = queue.Queue()
        old_job = FakeJob()
        new_job = FakeJob()
        launcher._process_job = new_job
        failed = subprocess.CompletedProcess(("taskkill",), 1)
        with patch("live_caption.launcher.subprocess.run", return_value=failed):
            launcher._terminate_process_tree(FakeProcess(), old_job)
        self.assertTrue(old_job.terminated)
        self.assertFalse(new_job.terminated)
        self.assertEqual("force_error", launcher._events.get_nowait()[0])

    def test_unexpected_exit_preserves_operational_diagnostics(self) -> None:
        launcher = object.__new__(CaptionLauncher)
        launcher._events = queue.Queue()
        launcher._events.put(("exit", 7))
        launcher._process = MagicMock()
        launcher._process_job = None
        launcher._force_stop_job = None
        launcher._stop_requested = False
        launcher._close_requested = False
        launcher._root = MagicMock()
        launcher._clear_log = MagicMock()
        launcher._append_log = MagicMock()
        launcher._set_status = MagicMock()
        launcher._set_running_controls = MagicMock()
        launcher._show_details = MagicMock()
        launcher._poll_events()
        launcher._clear_log.assert_not_called()
        launcher._append_log.assert_called_with("字幕进程已退出（代码 7）。")
        launcher._show_details.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

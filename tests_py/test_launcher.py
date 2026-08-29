from __future__ import annotations

import queue
import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from live_caption.launcher import (
    CaptionLauncher,
    LauncherOptions,
    application_root,
    blend_hex_colors,
    build_caption_command,
    invalid_large_files,
    is_caption_output,
    launcher_status_from_line,
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

    def test_language_selection_updates_preview_only_while_idle(self) -> None:
        launcher = object.__new__(CaptionLauncher)
        launcher._controls_running = False
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
        launcher._cpu = MagicMock()
        launcher._cpu.get.return_value = False
        launcher._refresh_cpu_toggle = MagicMock()

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

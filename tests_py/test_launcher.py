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
    build_caption_command,
    invalid_large_files,
    is_caption_output,
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
        launcher._poll_events()
        launcher._clear_log.assert_not_called()
        launcher._append_log.assert_called_with("字幕进程已退出（代码 7）。")


if __name__ == "__main__":
    unittest.main()

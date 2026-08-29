from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from live_caption.audio_capture import SystemAudioCapture


class SystemAudioCaptureLifecycleTests(unittest.TestCase):
    def test_stop_releases_every_resource_when_stop_stream_fails(self) -> None:
        capture = SystemAudioCapture(MagicMock())
        stream = MagicMock()
        stream.stop_stream.side_effect = RuntimeError("device unplugged")
        audio = MagicMock()
        capture._stream = stream
        capture._pyaudio = audio

        with self.assertRaisesRegex(RuntimeError, "device unplugged"):
            capture.stop()

        stream.close.assert_called_once_with()
        audio.terminate.assert_called_once_with()
        self.assertIsNone(capture._stream)
        self.assertIsNone(capture._pyaudio)

    def test_stop_releases_audio_when_stream_close_fails(self) -> None:
        capture = SystemAudioCapture(MagicMock())
        stream = MagicMock()
        stream.close.side_effect = RuntimeError("stream close failed")
        audio = MagicMock()
        capture._stream = stream
        capture._pyaudio = audio

        with self.assertRaisesRegex(RuntimeError, "stream close failed"):
            capture.stop()

        audio.terminate.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

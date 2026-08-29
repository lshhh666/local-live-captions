from __future__ import annotations

from collections.abc import Callable

import numpy as np


class SystemAudioCapture:
    """WASAPI loopback capture. Import is delayed for testability."""

    def __init__(self, on_samples: Callable[[np.ndarray], None], target_rate: int = 16_000) -> None:
        self._on_samples = on_samples
        self._target_rate = target_rate
        self._pyaudio = None
        self._stream = None

    @staticmethod
    def list_loopback_devices() -> list[dict]:
        import pyaudiowpatch as pyaudio

        with pyaudio.PyAudio() as audio:
            return [dict(device) for device in audio.get_loopback_device_info_generator()]

    def start(self, device_index: int | None = None) -> dict:
        import pyaudiowpatch as pyaudio

        self._pyaudio = pyaudio.PyAudio()
        device = self._resolve_device(device_index)
        source_rate = int(device["defaultSampleRate"])
        channels = int(device["maxInputChannels"])

        self._stream = self._pyaudio.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=source_rate,
            input=True,
            input_device_index=int(device["index"]),
            frames_per_buffer=max(1, source_rate // 20),
            stream_callback=lambda data, frame_count, time_info, status: self._callback(
                data, frame_count, time_info, status, channels, source_rate, pyaudio
            ),
        )
        self._stream.start_stream()
        return dict(device)

    def stop(self) -> None:
        stream = self._stream
        audio = self._pyaudio
        self._stream = None
        self._pyaudio = None
        first_error: Exception | None = None
        try:
            if stream is not None:
                try:
                    stream.stop_stream()
                except Exception as error:
                    first_error = error
                finally:
                    try:
                        stream.close()
                    except Exception as error:
                        first_error = first_error or error
        finally:
            if audio is not None:
                try:
                    audio.terminate()
                except Exception as error:
                    first_error = first_error or error
        if first_error is not None:
            raise first_error

    def _resolve_device(self, device_index: int | None) -> dict:
        assert self._pyaudio is not None
        if device_index is not None:
            device = dict(self._pyaudio.get_device_info_by_index(device_index))
            if not device.get("isLoopbackDevice"):
                raise ValueError("所选设备不是 WASAPI 回环设备")
            return device

        wasapi = self._pyaudio.get_host_api_info_by_type(13)
        output = dict(self._pyaudio.get_device_info_by_index(wasapi["defaultOutputDevice"]))
        for candidate in self._pyaudio.get_loopback_device_info_generator():
            if output["name"] in candidate["name"]:
                return dict(candidate)
        raise RuntimeError("找不到默认输出设备对应的 WASAPI 回环设备")

    def _callback(self, data, frame_count, time_info, status, channels, source_rate, pyaudio):
        del frame_count, time_info, status
        frames = np.frombuffer(data, dtype=np.float32)
        if channels > 1:
            frames = frames.reshape(-1, channels).mean(axis=1)
        if source_rate != self._target_rate and len(frames) > 1:
            output_length = max(1, int(len(frames) * self._target_rate / source_rate))
            source_positions = np.linspace(0, len(frames) - 1, output_length)
            frames = np.interp(source_positions, np.arange(len(frames)), frames).astype(np.float32)
        self._on_samples(frames.copy())
        return (None, pyaudio.paContinue)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

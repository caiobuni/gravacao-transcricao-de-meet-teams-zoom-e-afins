"""Captura de áudio dual-track: WASAPI Loopback (speakers) + Microfone."""

import logging
import threading
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import pyaudiowpatch as pyaudio

log = logging.getLogger(__name__)

from src.config.constants import (
    AUDIO_FORMAT_BITS,
    CHUNK_SIZE,
    MIC_SUFFIX,
    RECORDINGS_DIR,
    SPEAKERS_SUFFIX,
)


class DualTrackCapture:
    """Grava áudio em dois tracks simultâneos: loopback (speakers) e microfone."""

    def __init__(self, output_dir: Path | None = None):
        self._output_dir = output_dir or RECORDINGS_DIR
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._pa = pyaudio.PyAudio()
        self._loopback_stream = None
        self._mic_stream = None
        self._loopback_wave = None
        self._mic_wave = None
        self._is_recording = False
        self._lock = threading.Lock()
        self._start_time: datetime | None = None
        self._loopback_path: Path | None = None
        self._mic_path: Path | None = None

        # Formato dos streams (pode mudar para paFloat32 em fallback)
        self._loopback_format = pyaudio.paInt16
        self._mic_format = pyaudio.paInt16

        # Callbacks para UI (nível de áudio)
        self.on_loopback_level: callable | None = None
        self.on_mic_level: callable | None = None

    def get_loopback_device(self) -> dict:
        """Encontra o dispositivo WASAPI loopback padrão."""
        wasapi_info = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_output = self._pa.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"]
        )

        if default_output.get("isLoopbackDevice"):
            return default_output

        for loopback in self._pa.get_loopback_device_info_generator():
            if default_output["name"] in loopback["name"]:
                return loopback

        raise RuntimeError(
            "Nenhum dispositivo WASAPI loopback encontrado. "
            "Verifique se há um dispositivo de saída de áudio ativo."
        )

    def get_mic_device(self) -> dict:
        """Encontra o microfone padrão."""
        default_input = self._pa.get_default_input_device_info()
        return default_input

    def _loopback_callback(self, in_data, frame_count, time_info, status):
        """Callback para stream loopback (speakers)."""
        if self._is_recording and self._loopback_wave:
            write_data = in_data
            if self._loopback_format == pyaudio.paFloat32:
                samples = np.frombuffer(in_data, dtype=np.float32)
                samples = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
                write_data = samples.tobytes()
            self._loopback_wave.writeframes(write_data)
            if self.on_loopback_level:
                level = _compute_rms(write_data)
                self.on_loopback_level(level)
        return (in_data, pyaudio.paContinue)

    def _mic_callback(self, in_data, frame_count, time_info, status):
        """Callback para stream do microfone."""
        if self._is_recording and self._mic_wave:
            write_data = in_data
            if self._mic_format == pyaudio.paFloat32:
                samples = np.frombuffer(in_data, dtype=np.float32)
                samples = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
                write_data = samples.tobytes()
            self._mic_wave.writeframes(write_data)
            if self.on_mic_level:
                level = _compute_rms(write_data)
                self.on_mic_level(level)
        return (in_data, pyaudio.paContinue)

    def start(self) -> tuple[Path, Path]:
        """Inicia gravação dual-track. Retorna (path_speakers, path_mic)."""
        with self._lock:
            if self._is_recording:
                raise RuntimeError("Gravação já em andamento.")

            self._start_time = datetime.now()
            timestamp = self._start_time.strftime("%Y%m%d_%H%M%S")

            # Dispositivos
            loopback_dev = self.get_loopback_device()
            mic_dev = self.get_mic_device()

            # Paths
            self._loopback_path = self._output_dir / f"{timestamp}{SPEAKERS_SUFFIX}.wav"
            self._mic_path = self._output_dir / f"{timestamp}{MIC_SUFFIX}.wav"

            try:
                # Abrir WAVs — loopback
                loopback_channels = loopback_dev["maxInputChannels"]
                loopback_rate = int(loopback_dev["defaultSampleRate"])

                self._loopback_wave = wave.open(str(self._loopback_path), "wb")
                self._loopback_wave.setnchannels(loopback_channels)
                self._loopback_wave.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
                self._loopback_wave.setframerate(loopback_rate)

                # Abrir WAVs — microfone
                mic_channels = min(mic_dev["maxInputChannels"], 1)  # mono
                mic_rate = int(mic_dev["defaultSampleRate"])

                self._mic_wave = wave.open(str(self._mic_path), "wb")
                self._mic_wave.setnchannels(mic_channels)
                self._mic_wave.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
                self._mic_wave.setframerate(mic_rate)

                # Abrir streams — tenta multiplas configuracoes
                self._loopback_stream, self._loopback_format = self._open_stream(
                    loopback_dev, loopback_channels, loopback_rate,
                    self._loopback_callback,
                )

                self._mic_stream, self._mic_format = self._open_stream(
                    mic_dev, mic_channels, mic_rate,
                    self._mic_callback,
                )

                self._is_recording = True
                return self._loopback_path, self._mic_path
            except Exception:
                self._cleanup_partial()
                raise

    def _open_stream(self, device, channels, rate, callback):
        """Tenta abrir stream com multiplas configuracoes de formato/buffer."""
        configs = [
            (pyaudio.paFloat32, CHUNK_SIZE),
            (pyaudio.paInt16, CHUNK_SIZE),
            (pyaudio.paFloat32, None),
        ]
        last_error = None
        for fmt, buf_size in configs:
            try:
                kwargs = dict(
                    format=fmt,
                    channels=channels,
                    rate=rate,
                    input=True,
                    input_device_index=device["index"],
                    stream_callback=callback,
                )
                if buf_size is not None:
                    kwargs["frames_per_buffer"] = buf_size
                stream = self._pa.open(**kwargs)
                fmt_name = "float32" if fmt == pyaudio.paFloat32 else "int16"
                log.info(
                    "Stream aberto: fmt=%s, buf=%s, dev=%s",
                    fmt_name, buf_size, device["name"],
                )
                return stream, fmt
            except Exception as e:
                last_error = e
                fmt_name = "float32" if fmt == pyaudio.paFloat32 else "int16"
                log.debug(
                    "Falha ao abrir stream (fmt=%s, buf=%s): %s",
                    fmt_name, buf_size, e,
                )
        raise last_error

    def reset(self):
        """Reinicializa PyAudio para limpar estado interno."""
        try:
            self._pa.terminate()
        except Exception:
            pass
        self._pa = pyaudio.PyAudio()

    def _cleanup_partial(self):
        """Limpa estado parcial de um start() que falhou."""
        for stream in (self._loopback_stream, self._mic_stream):
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
        self._loopback_stream = None
        self._mic_stream = None

        for wf in (self._loopback_wave, self._mic_wave):
            if wf:
                try:
                    wf.close()
                except Exception:
                    pass
        self._loopback_wave = None
        self._mic_wave = None

        for path in (self._loopback_path, self._mic_path):
            if path and path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass

    def stop(self) -> tuple[Path, Path, datetime, float]:
        """Para gravação. Retorna (path_speakers, path_mic, start_time, duration_secs)."""
        with self._lock:
            if not self._is_recording:
                raise RuntimeError("Nenhuma gravação em andamento.")

            self._is_recording = False
            end_time = datetime.now()
            duration = (end_time - self._start_time).total_seconds()

            # Fechar streams
            if self._loopback_stream:
                self._loopback_stream.stop_stream()
                self._loopback_stream.close()
                self._loopback_stream = None

            if self._mic_stream:
                self._mic_stream.stop_stream()
                self._mic_stream.close()
                self._mic_stream = None

            # Fechar WAVs
            if self._loopback_wave:
                self._loopback_wave.close()
                self._loopback_wave = None

            if self._mic_wave:
                self._mic_wave.close()
                self._mic_wave = None

            return self._loopback_path, self._mic_path, self._start_time, duration

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def start_time(self) -> datetime | None:
        return self._start_time

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time and self._is_recording:
            return (datetime.now() - self._start_time).total_seconds()
        return 0.0

    def terminate(self):
        """Libera recursos do PyAudio."""
        if self._is_recording:
            self.stop()
        self._pa.terminate()

    def __del__(self):
        try:
            self.terminate()
        except Exception:
            pass


def _compute_rms(audio_data: bytes) -> float:
    """Calcula RMS do áudio para nível de volume."""
    samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples**2)))

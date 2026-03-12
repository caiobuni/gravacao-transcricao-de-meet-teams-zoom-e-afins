import logging
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from src.config.constants import ASSETS_DIR

log = logging.getLogger(__name__)

class MeetSoundDetector:
    """Detects Google Meet join/leave sounds using cross-correlation."""

    def __init__(self):
        self._running = False
        self._thread = None
        self._join_template = None  # Pre-loaded join sound signature
        self._leave_template = None  # Pre-loaded leave sound signature
        self._audio_buffer = np.array([], dtype=np.float32)
        self._buffer_lock = threading.Lock()
        self._last_detection_time = 0.0
        self._cooldown = 5.0  # seconds between detections

        # Callbacks
        self.on_meeting_joined: Callable | None = None
        self.on_meeting_left: Callable | None = None

    def load_templates(self):
        """Load reference sounds for join/leave detection."""
        import soundfile as sf

        join_path = ASSETS_DIR / "meet_join_sound.wav"
        leave_path = ASSETS_DIR / "meet_leave_sound.wav"

        if join_path.exists():
            data, sr = sf.read(str(join_path))
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            self._join_template = data.astype(np.float32)
            log.info("Template de som de entrada do Meet carregado")
        else:
            log.warning(f"Template de entrada não encontrado: {join_path}")

        if leave_path.exists():
            data, sr = sf.read(str(leave_path))
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            self._leave_template = data.astype(np.float32)
            log.info("Template de som de saída do Meet carregado")
        else:
            log.warning(f"Template de saída não encontrado: {leave_path}")

    def feed_audio(self, audio_data: bytes, sample_rate: int = 48000, channels: int = 2):
        """Feed audio data from the loopback stream for analysis."""
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Convert to mono if stereo
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)

        with self._buffer_lock:
            self._audio_buffer = np.concatenate([self._audio_buffer, samples])
            # Keep only last 5 seconds of audio
            max_samples = sample_rate * 5
            if len(self._audio_buffer) > max_samples:
                self._audio_buffer = self._audio_buffer[-max_samples:]

    def start(self, sample_rate: int = 48000):
        """Start the detection loop."""
        self.load_templates()
        self._running = True
        self._sample_rate = sample_rate
        self._thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _detect_loop(self):
        """Periodically check buffer for Meet sounds."""
        while self._running:
            try:
                with self._buffer_lock:
                    if len(self._audio_buffer) < self._sample_rate:
                        time.sleep(0.5)
                        continue
                    buffer_copy = self._audio_buffer.copy()

                now = time.time()
                if now - self._last_detection_time < self._cooldown:
                    time.sleep(0.5)
                    continue

                # Check for join sound
                if self._join_template is not None:
                    score = self._correlate(buffer_copy, self._join_template)
                    if score > 0.6:  # Threshold
                        log.info(f"Som de entrada do Meet detectado (score: {score:.2f})")
                        self._last_detection_time = now
                        if self.on_meeting_joined:
                            self.on_meeting_joined()

                # Check for leave sound
                if self._leave_template is not None:
                    score = self._correlate(buffer_copy, self._leave_template)
                    if score > 0.6:
                        log.info(f"Som de saída do Meet detectado (score: {score:.2f})")
                        self._last_detection_time = now
                        if self.on_meeting_left:
                            self.on_meeting_left()

                time.sleep(0.5)
            except Exception as e:
                log.debug(f"Erro na detecção: {e}")
                time.sleep(1)

    def _correlate(self, signal_data: np.ndarray, template: np.ndarray) -> float:
        """Compute normalized cross-correlation between signal and template."""
        if len(signal_data) < len(template):
            return 0.0

        # Normalize
        template_norm = template - np.mean(template)
        t_std = np.std(template_norm)
        if t_std < 1e-8:
            return 0.0
        template_norm = template_norm / t_std

        # Sliding cross-correlation using numpy
        correlation = np.correlate(signal_data, template_norm, mode='valid')

        # Normalize by window standard deviation
        window_size = len(template)
        # Use cumulative sum for efficient windowed std
        cumsum = np.cumsum(signal_data)
        cumsum2 = np.cumsum(signal_data ** 2)

        window_sum = cumsum[window_size:] - cumsum[:-window_size]
        window_sum = np.concatenate([[cumsum[window_size - 1]], window_sum])

        window_sum2 = cumsum2[window_size:] - cumsum2[:-window_size]
        window_sum2 = np.concatenate([[cumsum2[window_size - 1]], window_sum2])

        window_mean = window_sum / window_size
        window_var = window_sum2 / window_size - window_mean ** 2
        window_std = np.sqrt(np.maximum(window_var, 0))

        # Avoid division by zero
        valid = window_std > 1e-8
        normalized = np.zeros_like(correlation)
        normalized[valid] = correlation[valid] / (window_std[valid] * window_size)

        return float(np.max(normalized)) if len(normalized) > 0 else 0.0

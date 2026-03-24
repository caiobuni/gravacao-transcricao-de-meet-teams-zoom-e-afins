"""Pré-processamento de áudio: resample para 16kHz mono."""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from src.config.constants import WHISPER_SAMPLE_RATE


def prepare_for_transcription(input_path: Path, output_path: Path | None = None) -> Path:
    """Converte áudio para 16kHz mono float32 WAV (formato esperado pelo Whisper)."""
    if output_path is None:
        output_path = input_path.with_stem(input_path.stem + "_16k")

    data, samplerate = sf.read(str(input_path))

    if len(data) == 0:
        raise ValueError(f"Arquivo de audio vazio: {input_path}")

    if samplerate <= 0:
        raise ValueError(f"Sample rate invalido ({samplerate}): {input_path}")

    # Stereo → mono
    if len(data.shape) > 1 and data.shape[1] > 1:
        data = np.mean(data, axis=1)

    # Resample para 16kHz
    if samplerate != WHISPER_SAMPLE_RATE:
        num_samples = int(len(data) * WHISPER_SAMPLE_RATE / samplerate)
        data = signal.resample(data, num_samples)

    # Normalizar
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), data, WHISPER_SAMPLE_RATE, subtype="FLOAT")
    return output_path

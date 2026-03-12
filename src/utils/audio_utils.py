"""Funções utilitárias para áudio."""

import logging
from pathlib import Path

from src.config.constants import GROQ_CHUNK_DURATION_SECONDS

logger = logging.getLogger(__name__)

OVERLAP_SECONDS = 2


def format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS for display."""
    return format_duration(seconds)


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm for SRT format."""
    total_ms = int(round(seconds * 1000))
    h = total_ms // 3_600_000
    total_ms %= 3_600_000
    m = total_ms // 60_000
    total_ms %= 60_000
    s = total_ms // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def split_audio_file(
    input_path: Path,
    chunk_duration_seconds: int = GROQ_CHUNK_DURATION_SECONDS,
    output_dir: Path | None = None,
) -> list[Path]:
    """Split a large audio file into chunks for API upload (max 25MB/15min each).

    Each chunk overlaps by 2 seconds for context continuity.

    Args:
        input_path: Path to the source audio file.
        chunk_duration_seconds: Duration of each chunk in seconds (default 900 = 15min).
        output_dir: Directory for output chunks. Defaults to input_path's parent.

    Returns:
        List of paths to the generated chunk files.
    """
    import soundfile as sf

    if output_dir is None:
        output_dir = input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    data, samplerate = sf.read(input_path)
    total_samples = len(data)
    chunk_samples = chunk_duration_seconds * samplerate
    overlap_samples = OVERLAP_SECONDS * samplerate

    if total_samples <= chunk_samples:
        logger.info("Áudio menor que o tamanho do chunk, retornando arquivo original")
        return [input_path]

    chunks: list[Path] = []
    start = 0
    idx = 0

    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk_data = data[start:end]

        chunk_path = output_dir / f"{input_path.stem}_chunk{idx:03d}{input_path.suffix}"
        sf.write(str(chunk_path), chunk_data, samplerate)
        chunks.append(chunk_path)
        logger.info("Chunk %d criado: %s", idx, chunk_path.name)

        idx += 1
        start = end - overlap_samples if end < total_samples else total_samples

    logger.info("Áudio dividido em %d chunks", len(chunks))
    return chunks

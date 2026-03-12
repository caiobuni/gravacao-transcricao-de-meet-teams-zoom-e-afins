"""Formatação de transcrições em SRT e Markdown."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Union

from src.core.aligner import AlignedSegment
from src.core.transcriber_base import TranscriptionSegment

logger = logging.getLogger(__name__)


def _format_srt_time(seconds: float) -> str:
    """Formata segundos no padrão SRT: HH:MM:SS,mmm."""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_timestamp(seconds: float) -> str:
    """Formata segundos em [HH:MM:SS]."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"


def _format_duration(seconds: float) -> str:
    """Formata duração em HH:MM:SS."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def to_srt(segments: list[Union[TranscriptionSegment, AlignedSegment]]) -> str:
    """Gera formato de legenda SRT.

    Args:
        segments: Lista de segmentos de transcrição ou alinhados.

    Returns:
        String no formato SRT padrão.
    """
    lines: list[str] = []

    for index, seg in enumerate(segments, start=1):
        start_time = _format_srt_time(seg.start)
        end_time = _format_srt_time(seg.end)

        text = seg.text.strip()
        # Inclui nome do falante se for AlignedSegment
        if isinstance(seg, AlignedSegment):
            text = f"{seg.speaker}: {text}"

        lines.append(str(index))
        lines.append(f"{start_time} --> {end_time}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


def to_markdown(
    segments: list[AlignedSegment],
    start_time: datetime,
    duration_seconds: float,
    engine_name: str,
) -> str:
    """Gera arquivo de transcrição final em Markdown.

    Args:
        segments: Lista de segmentos alinhados com falantes.
        start_time: Hora de início da reunião.
        duration_seconds: Duração total em segundos.
        engine_name: Nome do motor de transcrição utilizado.

    Returns:
        String formatada em Markdown.
    """
    end_time = start_time + timedelta(seconds=duration_seconds)
    participants = sorted({seg.speaker for seg in segments})
    duration_str = _format_duration(duration_seconds)

    lines: list[str] = [
        "# Transcrição de Reunião",
        "",
        f"- **Data:** {start_time.strftime('%Y-%m-%d')}",
        f"- **Início:** {start_time.strftime('%H:%M')}",
        f"- **Fim:** {end_time.strftime('%H:%M')}",
        f"- **Duração:** {duration_str}",
        f"- **Participantes:** {', '.join(participants)}",
        f"- **Motor:** {engine_name}",
        "",
        "---",
        "",
        "## Transcrição",
        "",
    ]

    # Mescla segmentos consecutivos do mesmo falante
    current_speaker: str | None = None

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue

        if seg.speaker != current_speaker:
            # Novo falante: adiciona cabeçalho
            timestamp = _format_timestamp(seg.start)
            lines.append(f"**{timestamp} {seg.speaker}:**")
            current_speaker = seg.speaker

        lines.append(text)
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Transcrição gerada automaticamente por gravacao-transcricao*")
    lines.append(f"*Motor: {engine_name} | Diarização: pyannote 3.1*")
    lines.append("")

    return "\n".join(lines)

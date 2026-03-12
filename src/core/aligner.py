"""Alinhamento de transcrições dual-track com diarização."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.core.diarizer import DiarizationSegment
from src.core.transcriber_base import TranscriptionSegment

logger = logging.getLogger(__name__)


@dataclass
class AlignedSegment:
    start: float
    end: float
    speaker: str  # nome do usuário, "Pessoa 1", "Pessoa 2", ou nome identificado
    text: str
    is_user: bool  # True se originário da faixa do microfone


def _compute_overlap(seg_start: float, seg_end: float,
                     dia_start: float, dia_end: float) -> float:
    """Calcula a sobreposição temporal entre dois intervalos em segundos."""
    overlap_start = max(seg_start, dia_start)
    overlap_end = min(seg_end, dia_end)
    return max(0.0, overlap_end - overlap_start)


def _assign_speaker(
    segment: TranscriptionSegment,
    diarization: list[DiarizationSegment],
    speaker_map: dict[str, str],
) -> str:
    """Atribui um rótulo de falante a um segmento com base na sobreposição temporal."""
    if not diarization:
        return "Pessoa desconhecida"

    best_overlap = 0.0
    best_label = diarization[0].speaker

    for dia_seg in diarization:
        overlap = _compute_overlap(
            segment.start, segment.end,
            dia_seg.start, dia_seg.end,
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = dia_seg.speaker

    # Mapeia rótulos pyannote (SPEAKER_00) para nomes legíveis (Pessoa 1)
    if best_label not in speaker_map:
        person_number = len(speaker_map) + 1
        speaker_map[best_label] = f"Pessoa {person_number}"

    return speaker_map[best_label]


def align_dual_track(
    mic_segments: list[TranscriptionSegment],
    speakers_segments: list[TranscriptionSegment],
    diarization: list[DiarizationSegment],
    user_name: str = "Eu",
) -> list[AlignedSegment]:
    """Combina transcrições de duas faixas com diarização.

    Args:
        mic_segments: Segmentos da faixa do microfone (sempre o usuário).
        speakers_segments: Segmentos da faixa dos alto-falantes (outros participantes).
        diarization: Resultado da diarização para a faixa dos alto-falantes.
        user_name: Nome do usuário para identificar segmentos do microfone.

    Returns:
        Lista unificada de segmentos ordenada por tempo de início.
    """
    aligned: list[AlignedSegment] = []
    speaker_map: dict[str, str] = {}

    # Segmentos do microfone sao sempre do usuario
    for seg in mic_segments:
        aligned.append(
            AlignedSegment(
                start=seg.start,
                end=seg.end,
                speaker=user_name,
                text=seg.text,
                is_user=True,
            )
        )

    # Segmentos dos alto-falantes recebem identificação via diarização
    for seg in speakers_segments:
        speaker = _assign_speaker(seg, diarization, speaker_map)
        aligned.append(
            AlignedSegment(
                start=seg.start,
                end=seg.end,
                speaker=speaker,
                text=seg.text,
                is_user=False,
            )
        )

    # Ordena por tempo de início
    aligned.sort(key=lambda s: s.start)

    logger.info(
        "Alinhamento concluído: %d segmentos do mic + %d dos alto-falantes = %d total, %d falantes.",
        len(mic_segments),
        len(speakers_segments),
        len(aligned),
        len({s.speaker for s in aligned}),
    )

    return aligned

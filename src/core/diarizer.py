"""Diarização de falantes via pyannote-audio."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MODELS_CACHE_DIR = Path("models/pyannote")


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: str  # "SPEAKER_00", "SPEAKER_01", etc.


class Diarizer:
    """Diarização de falantes usando pyannote/speaker-diarization-3.1."""

    def __init__(self) -> None:
        self._pipeline = None

    def _load_pipeline(self) -> None:
        """Carrega o pipeline de diarização sob demanda."""
        if self._pipeline is not None:
            return

        import torch
        from pyannote.audio import Pipeline

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "Variável de ambiente HF_TOKEN não definida. "
                "Necessária para baixar o modelo pyannote."
            )

        MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("Carregando pipeline pyannote/speaker-diarization-3.1...")
        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
            cache_dir=str(MODELS_CACHE_DIR),
        )

        if torch.cuda.is_available():
            logger.info("GPU detectada, movendo pipeline para CUDA.")
            self._pipeline.to(torch.device("cuda"))
        else:
            logger.info("GPU não disponível, usando CPU.")

    def diarize(
        self,
        audio_path: Path,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[DiarizationSegment]:
        """Executa diarização no arquivo de áudio.

        Args:
            audio_path: Caminho para o arquivo de áudio.
            num_speakers: Número exato de falantes (se conhecido).
            min_speakers: Número mínimo de falantes esperado.
            max_speakers: Número máximo de falantes esperado.

        Returns:
            Lista de segmentos com identificação de falante.
        """
        self._load_pipeline()

        kwargs: dict = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        logger.info("Executando diarização em %s...", audio_path)
        diarization = self._pipeline(str(audio_path), **kwargs)

        segments: list[DiarizationSegment] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                DiarizationSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker,
                )
            )

        logger.info(
            "Diarização concluída: %d segmentos, %d falantes.",
            len(segments),
            len({s.speaker for s in segments}),
        )
        return segments

    def is_available(self) -> bool:
        """Verifica se o pyannote-audio está instalado."""
        try:
            import pyannote.audio  # noqa: F401
            return bool(os.environ.get("HF_TOKEN"))
        except ImportError:
            return False

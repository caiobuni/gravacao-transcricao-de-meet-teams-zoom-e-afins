"""Transcrição via faster-whisper."""

import logging
from pathlib import Path
from typing import Callable

from src.config.constants import DEFAULT_BEAM_SIZE, DEFAULT_LANGUAGE
from src.core.transcriber_base import TranscriberBase, TranscriptionSegment

logger = logging.getLogger(__name__)


class WhisperTranscriber(TranscriberBase):
    def __init__(self, model_size: str = "large-v3"):
        self._model_size = model_size
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return

        import torch
        from faster_whisper import WhisperModel

        use_cuda = torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
        compute_type = "float16" if use_cuda else "int8"

        logger.info(
            "Carregando faster-whisper %s (device=%s, compute=%s)",
            self._model_size, device, compute_type,
        )
        self._model = WhisperModel(self._model_size, device=device, compute_type=compute_type)
        logger.info("Modelo faster-whisper carregado.")

    def transcribe(self, audio_path: Path, language: str = "pt",
                   on_progress: Callable[[float], None] | None = None) -> list[TranscriptionSegment]:
        self._load_model()

        lang = language or DEFAULT_LANGUAGE
        logger.info("Transcrevendo %s (idioma=%s)", audio_path.name, lang)

        try:
            segments_iter, info = self._model.transcribe(
                str(audio_path),
                language=lang,
                beam_size=DEFAULT_BEAM_SIZE,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=400),
                word_timestamps=True,
                initial_prompt="Transcrição de reunião em português brasileiro.",
            )
        except Exception:
            logger.exception("Erro ao transcrever com faster-whisper")
            raise

        duration = info.duration if info.duration > 0 else 1.0
        result: list[TranscriptionSegment] = []

        for seg in segments_iter:
            result.append(TranscriptionSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
            ))
            if on_progress:
                progress = min(seg.end / duration, 1.0)
                on_progress(progress)

        if on_progress:
            on_progress(1.0)

        logger.info("Transcrição faster-whisper concluída: %d segmentos", len(result))
        return result

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def name(self) -> str:
        return "faster-whisper"

"""Transcrição via Qwen3-ASR-0.6B."""

import logging
import re
from pathlib import Path
from typing import Callable

from src.config.constants import QWEN_LANGUAGE
from src.core.transcriber_base import TranscriberBase, TranscriptionSegment

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r'[.!?;:]\s*$')
_PAUSE_THRESHOLD = 0.8  # seconds


class QwenTranscriber(TranscriberBase):
    def __init__(self, use_forced_aligner: bool = True):
        self._use_forced_aligner = use_forced_aligner
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return

        import torch
        from qwen_asr import Qwen3ASRModel

        use_cuda = torch.cuda.is_available()
        device_map = "cuda:0" if use_cuda else "cpu"
        dtype = torch.bfloat16 if use_cuda else torch.float32

        kwargs = dict(
            model_name="Qwen/Qwen3-ASR-0.6B",
            device_map=device_map,
            torch_dtype=dtype,
        )
        if self._use_forced_aligner:
            kwargs["forced_aligner"] = "Qwen/Qwen3-ForcedAligner-0.6B"

        logger.info("Carregando modelo Qwen ASR (device=%s, dtype=%s)", device_map, dtype)
        self._model = Qwen3ASRModel(**kwargs)
        logger.info("Modelo Qwen ASR carregado.")

    def transcribe(self, audio_path: Path, language: str = "pt",
                   on_progress: Callable[[float], None] | None = None) -> list[TranscriptionSegment]:
        self._load_model()

        lang = QWEN_LANGUAGE if language == "pt" else language
        logger.info("Transcrevendo %s (idioma=%s)", audio_path.name, lang)

        try:
            results = self._model.transcribe(
                str(audio_path),
                language=lang,
                return_time_stamps=self._use_forced_aligner,
            )
        except Exception:
            logger.exception("Erro ao transcrever com Qwen ASR")
            raise

        if on_progress:
            on_progress(0.5)

        result = results[0]

        if self._use_forced_aligner and result.time_stamps:
            segments = self._build_segments(result.time_stamps)
        else:
            segments = [TranscriptionSegment(start=0.0, end=0.0, text=result.text.strip())]

        if on_progress:
            on_progress(1.0)

        logger.info("Transcrição Qwen concluída: %d segmentos", len(segments))
        return segments

    def _build_segments(self, time_stamps) -> list[TranscriptionSegment]:
        segments: list[TranscriptionSegment] = []
        current_words: list[str] = []
        seg_start: float | None = None
        seg_end: float = 0.0
        prev_end: float = 0.0

        for ts in time_stamps:
            gap = ts.start_time - prev_end if prev_end > 0 else 0.0
            word = ts.text

            if current_words and (gap >= _PAUSE_THRESHOLD or _SENTENCE_END.search(current_words[-1])):
                text = " ".join(current_words).strip()
                if text:
                    segments.append(TranscriptionSegment(start=seg_start, end=seg_end, text=text))
                current_words = []
                seg_start = None

            if seg_start is None:
                seg_start = ts.start_time

            current_words.append(word)
            seg_end = ts.end_time
            prev_end = ts.end_time

        if current_words:
            text = " ".join(current_words).strip()
            if text:
                segments.append(TranscriptionSegment(start=seg_start, end=seg_end, text=text))

        return segments

    def is_available(self) -> bool:
        try:
            import qwen_asr  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def name(self) -> str:
        return "Qwen3-ASR"

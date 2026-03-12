"""Transcrição via Groq API (fallback cloud)."""

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Callable

from src.config.constants import (
    DEFAULT_LANGUAGE,
    GROQ_CHUNK_DURATION_SECONDS,
    GROQ_MAX_FILE_SIZE_MB,
    GROQ_MODEL,
)
from src.core.transcriber_base import TranscriberBase, TranscriptionSegment

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0


class GroqTranscriber(TranscriberBase):
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.environ.get("GROQ_API_KEY")
            except ImportError:
                pass

        if not api_key:
            raise RuntimeError("GROQ_API_KEY não encontrada nas variáveis de ambiente.")

        self._client = Groq(api_key=api_key)
        return self._client

    def transcribe(self, audio_path: Path, language: str = "pt",
                   on_progress: Callable[[float], None] | None = None) -> list[TranscriptionSegment]:
        lang = language or DEFAULT_LANGUAGE
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        logger.info("Transcrevendo %s via Groq (%.1f MB, idioma=%s)", audio_path.name, file_size_mb, lang)

        if file_size_mb > GROQ_MAX_FILE_SIZE_MB:
            return self._transcribe_chunked(audio_path, lang, on_progress)

        return self._transcribe_single(audio_path, lang, on_progress)

    def _transcribe_single(self, audio_path: Path, language: str,
                           on_progress: Callable[[float], None] | None) -> list[TranscriptionSegment]:
        client = self._get_client()

        if on_progress:
            on_progress(0.1)

        response = self._api_call(client, audio_path, language)
        segments = self._parse_response(response)

        if on_progress:
            on_progress(1.0)

        logger.info("Transcrição Groq concluída: %d segmentos", len(segments))
        return segments

    def _transcribe_chunked(self, audio_path: Path, language: str,
                            on_progress: Callable[[float], None] | None) -> list[TranscriptionSegment]:
        import soundfile as sf

        client = self._get_client()
        data, samplerate = sf.read(str(audio_path))
        total_samples = len(data)
        chunk_samples = GROQ_CHUNK_DURATION_SECONDS * samplerate
        chunks = []

        # Split into chunks
        offset = 0
        while offset < total_samples:
            end = min(offset + chunk_samples, total_samples)
            chunks.append((offset, end))
            offset = end

        logger.info("Arquivo grande: dividido em %d chunks de ~%d min", len(chunks), GROQ_CHUNK_DURATION_SECONDS // 60)

        all_segments: list[TranscriptionSegment] = []

        for i, (start_sample, end_sample) in enumerate(chunks):
            chunk_data = data[start_sample:end_sample]
            time_offset = start_sample / samplerate

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                sf.write(str(tmp_path), chunk_data, samplerate)

            try:
                response = self._api_call(client, tmp_path, language)
                chunk_segments = self._parse_response(response, time_offset=time_offset)
                all_segments.extend(chunk_segments)
            finally:
                tmp_path.unlink(missing_ok=True)

            if on_progress:
                on_progress((i + 1) / len(chunks))

        logger.info("Transcrição Groq (chunked) concluída: %d segmentos", len(all_segments))
        return all_segments

    def _api_call(self, client, audio_path: Path, language: str):
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                with open(audio_path, "rb") as audio_file:
                    return client.audio.transcriptions.create(
                        file=audio_file,
                        model=GROQ_MODEL,
                        language=language,
                        response_format="verbose_json",
                        timestamp_granularities=["segment"],
                    )
            except Exception as e:
                is_rate_limit = "rate" in str(e).lower() or "429" in str(e)
                if attempt < _MAX_RETRIES and is_rate_limit:
                    delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning("Rate limit Groq, tentativa %d/%d. Aguardando %.1fs...", attempt, _MAX_RETRIES, delay)
                    time.sleep(delay)
                else:
                    logger.exception("Erro na API Groq (tentativa %d/%d)", attempt, _MAX_RETRIES)
                    raise

    def _parse_response(self, response, time_offset: float = 0.0) -> list[TranscriptionSegment]:
        segments: list[TranscriptionSegment] = []

        resp_segments = getattr(response, "segments", None)
        if resp_segments:
            for seg in resp_segments:
                segments.append(TranscriptionSegment(
                    start=seg["start"] + time_offset,
                    end=seg["end"] + time_offset,
                    text=seg["text"].strip(),
                ))
        else:
            text = getattr(response, "text", "").strip()
            if text:
                segments.append(TranscriptionSegment(start=time_offset, end=time_offset, text=text))

        return segments

    def is_available(self) -> bool:
        try:
            import groq  # noqa: F401
        except ImportError:
            return False
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.environ.get("GROQ_API_KEY")
            except ImportError:
                pass
        return bool(api_key)

    @property
    def name(self) -> str:
        return "Groq"

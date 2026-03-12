"""Logger de eventos de gravacao."""

import logging
import re
from datetime import datetime
from pathlib import Path

from src.config.constants import LOG_DIR, LOG_FILE, TRANSCRIPTION_OUTPUT_DIR

logger = logging.getLogger(__name__)


def _format_duration(seconds: float) -> str:
    """Formata segundos como HH:MM:SS."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


class RecordingLogger:
    """Logs recording events to a structured log file."""

    def __init__(self, log_file: Path | None = None) -> None:
        self.log_file = log_file or LOG_FILE
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, event_type: str, details: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} | {event_type:17s} | {details}\n"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line)

    def log_start(self, trigger: str = "manual") -> None:
        """Log recording start event."""
        self._write("INÍCIO GRAVAÇÃO", trigger)

    def log_stop(self, duration_seconds: float, audio_file: str) -> None:
        """Log recording stop event."""
        duration = _format_duration(duration_seconds)
        self._write("FIM GRAVAÇÃO", f"duração: {duration} | arquivo: {audio_file}")

    def log_transcription(self, output_file: str, engine: str) -> None:
        """Log successful transcription event."""
        self._write("TRANSCRIÇÃO OK", f"arquivo: {output_file} | motor: {engine}")

    def log_transcription_failed(self, error: str) -> None:
        """Log failed transcription event."""
        self._write("TRANSCRIÇÃO FALHOU", f"erro: {error}")

    def log_audio_deleted(self, audio_file: str) -> None:
        """Log audio file deletion event."""
        self._write("ÁUDIO DELETADO", audio_file)

    def log_audio_kept(self, audio_file: str, reason: str) -> None:
        """Log audio file kept event."""
        self._write("ÁUDIO MANTIDO", f"{audio_file} | motivo: {reason}")

    def get_last_transcription_path(self) -> Path | None:
        """Parse log file to find the most recent TRANSCRIÇÃO OK entry and return its file path."""
        if not self.log_file.exists():
            return None

        pattern = re.compile(r"TRANSCRIÇÃO OK\s+\| arquivo: (.+?) \| motor:")
        last_match: str | None = None

        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    last_match = m.group(1).strip()

        if last_match is None:
            return None

        path = Path(last_match)
        if path.is_absolute():
            return path
        return TRANSCRIPTION_OUTPUT_DIR / path

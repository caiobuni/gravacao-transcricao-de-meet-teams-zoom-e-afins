"""Interface base para transcrição de áudio."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class TranscriptionSegment:
    start: float  # seconds
    end: float    # seconds
    text: str


class TranscriberBase(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path, language: str = "pt",
                   on_progress: Callable[[float], None] | None = None) -> list[TranscriptionSegment]:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

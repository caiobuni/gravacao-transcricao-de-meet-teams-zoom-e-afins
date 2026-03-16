"""Configurações persistentes do aplicativo."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from src.config.constants import CONFIG_FILE, TRANSCRIPTION_OUTPUT_DIR


@dataclass
class Settings:
    """Configurações do usuário, persistidas em JSON."""

    # Identidade
    user_name: str = "Eu"

    # Motor de transcrição: "whisper", "groq"
    transcription_engine: str = "whisper"

    # Modelo faster-whisper
    whisper_model: str = "large-v3"
    whisper_compute_type: str = "auto"  # auto, float16, int8

    # Idioma
    language: str = "pt"

    # Caminhos
    transcription_output_dir: str = str(TRANSCRIPTION_OUTPUT_DIR)

    # Gravação
    auto_delete_audio: bool = True

    # Meet
    auto_detect_meet: bool = True

    # Windows
    start_with_windows: bool = True

    # Vexa
    vexa_api_key: str = ""
    vexa_base_url: str = "https://api.vexa.ai"
    vexa_bot_name: str = "Caio: bot de transcrição"
    vexa_bot_image: str = "📝"
    vexa_auto_download: bool = True

    # Voice profiles
    voice_profiles: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Settings":
        """Carrega configurações do arquivo JSON."""
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
            except (json.JSONDecodeError, TypeError):
                pass
        return cls()

    def save(self) -> None:
        """Salva configurações no arquivo JSON."""
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

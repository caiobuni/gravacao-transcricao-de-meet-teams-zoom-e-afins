"""Constantes globais do aplicativo."""

from pathlib import Path

# Diretórios
APP_DIR = Path(__file__).resolve().parent.parent.parent
RECORDINGS_DIR = APP_DIR / "recordings"
VEXA_RECORDINGS_DIR = RECORDINGS_DIR / "vexa"
LOG_DIR = APP_DIR / "log"
MODELS_DIR = APP_DIR / "models"
DATA_DIR = APP_DIR / "data"
ASSETS_DIR = APP_DIR / "assets"
VOICE_PROFILES_DIR = DATA_DIR / "voice_profiles"

# Caminho de saída das transcrições
TRANSCRIPTION_OUTPUT_DIR = Path(r"H:\Meu Drive\Tactiq Transcription")

# Arquivo de log
LOG_FILE = LOG_DIR / "gravacoes.log"

# Arquivo de configuração do usuário
CONFIG_FILE = DATA_DIR / "config.json"
TASK_QUEUE_FILE = DATA_DIR / "task_queue.json"

# Áudio
WHISPER_SAMPLE_RATE = 16000
DEFAULT_SAMPLE_RATE = 48000
AUDIO_FORMAT_BITS = 16
AUDIO_CHANNELS_MONO = 1
AUDIO_CHANNELS_STEREO = 2
CHUNK_SIZE = 1024

# Groq API
GROQ_MODEL = "whisper-large-v3"
GROQ_MAX_FILE_SIZE_MB = 25
GROQ_CHUNK_DURATION_SECONDS = 900  # 15 minutos por chunk

# Transcrição
DEFAULT_LANGUAGE = "pt"
DEFAULT_BEAM_SIZE = 5

# Meet
MEET_PROCESS_NAMES = ["chrome.exe", "msedge.exe"]
MEET_URL_PATTERN = "meet.google.com"

# Nomes de arquivos
SPEAKERS_SUFFIX = "_speakers"
MIC_SUFFIX = "_mic"

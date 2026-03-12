"""Download all required ML models for offline use."""

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.config.constants import MODELS_DIR


def download_qwen_asr():
    print("=== Downloading Qwen3-ASR-0.6B ===")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download("Qwen/Qwen3-ASR-0.6B", local_dir=MODELS_DIR / "qwen" / "Qwen3-ASR-0.6B")
        print("Qwen3-ASR-0.6B downloaded.")

        print("=== Downloading Qwen3-ForcedAligner-0.6B ===")
        snapshot_download("Qwen/Qwen3-ForcedAligner-0.6B", local_dir=MODELS_DIR / "qwen" / "Qwen3-ForcedAligner-0.6B")
        print("Qwen3-ForcedAligner-0.6B downloaded.")
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
    except Exception as e:
        print(f"Error downloading Qwen3-ASR: {e}")


def download_whisper():
    print("\n=== Downloading faster-whisper large-v3 ===")
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("large-v3", download_root=str(MODELS_DIR / "whisper"))
        del model
        print("faster-whisper large-v3 downloaded.")
    except ImportError:
        print("faster-whisper not installed. Run: pip install faster-whisper")
    except Exception as e:
        print(f"Error downloading Whisper: {e}")


def download_pyannote():
    print("\n=== Downloading pyannote speaker-diarization-3.1 ===")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        hf_token = input("Enter your HuggingFace token (needed for pyannote): ").strip()
        if not hf_token:
            print("Skipping pyannote (no token provided).")
            return

    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
            cache_dir=str(MODELS_DIR / "pyannote"),
        )
        del pipeline
        print("pyannote speaker-diarization-3.1 downloaded.")
    except ImportError:
        print("pyannote.audio not installed. Run: pip install pyannote.audio")
    except Exception as e:
        print(f"Error downloading pyannote: {e}")


def main():
    print("Gravacao e Transcricao — Download de Modelos")
    print("=" * 50)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    download_qwen_asr()
    download_whisper()
    download_pyannote()

    print("\n" + "=" * 50)
    print("Download concluido! Voce pode rodar o app offline agora.")
    print("Execute: python -m src.main")


if __name__ == "__main__":
    main()

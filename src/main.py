"""Ponto de entrada do aplicativo de gravacao e transcricao."""

import logging
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config.constants import LOG_DIR, RECORDINGS_DIR, DATA_DIR
from src.config.settings import Settings
from src.tray.tray_app import TrayApp


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
        ],
    )


def ensure_dirs():
    for d in [RECORDINGS_DIR, LOG_DIR, DATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def main():
    setup_logging()
    ensure_dirs()
    log = logging.getLogger(__name__)
    log.info("Iniciando Gravacao e Transcricao v0.1.0")

    settings = Settings.load()
    app = TrayApp(settings)

    try:
        app.run()
    except KeyboardInterrupt:
        log.info("Encerrado pelo usuario")
    except Exception as e:
        log.error(f"Erro fatal: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

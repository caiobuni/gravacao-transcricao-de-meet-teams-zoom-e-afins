"""System Tray application using pystray."""

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

from PIL import Image, ImageDraw
import pystray

from src.config.constants import LOG_FILE, TRANSCRIPTION_OUTPUT_DIR
from src.config.settings import Settings
from src.core.pipeline import Pipeline, PipelineStage
from src.core.process_monitor import MeetProcessMonitor
from src.core.meet_detector import MeetSoundDetector
from src.utils.logger import RecordingLogger

log = logging.getLogger(__name__)


class TrayApp:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._pipeline = Pipeline(settings)
        self._process_monitor = MeetProcessMonitor()
        self._sound_detector = MeetSoundDetector()
        self._rec_logger = RecordingLogger()
        self._icon: pystray.Icon | None = None
        self._status_text = "Pronto"

        # Wire callbacks
        self._pipeline.on_stage_change = self._on_stage_change
        self._pipeline.on_complete = self._on_complete
        self._pipeline.on_error = self._on_error

        if settings.auto_detect_meet:
            self._process_monitor.on_meet_opened = self._on_meet_opened
            self._sound_detector.on_meeting_joined = self._on_meeting_joined
            self._sound_detector.on_meeting_left = self._on_meeting_left

    def _create_icon_image(self, color: str) -> Image.Image:
        """Create a simple colored circle icon."""
        size = 64
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        colors = {
            "gray": (128, 128, 128, 255),
            "red": (220, 50, 50, 255),
            "yellow": (220, 180, 50, 255),
        }
        fill = colors.get(color, colors["gray"])
        draw.ellipse([4, 4, size - 4, size - 4], fill=fill, outline=(255, 255, 255, 200), width=2)
        return img

    def _build_menu(self) -> pystray.Menu:
        is_rec = self._pipeline.is_recording
        return pystray.Menu(
            pystray.MenuItem("Iniciar Gravacao", self._start_recording, enabled=not is_rec),
            pystray.MenuItem("Parar Gravacao", self._stop_recording, enabled=is_rec),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(f"Status: {self._status_text}", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Ultima transcricao", self._open_last_transcription),
            pystray.MenuItem("Abrir pasta de transcricoes", self._open_transcription_folder),
            pystray.MenuItem("Abrir log", self._open_log),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Configuracoes", self._open_settings),
            pystray.MenuItem("Sair", self._quit),
        )

    def _update_icon(self):
        if not self._icon:
            return
        stage = self._pipeline.stage
        if stage == PipelineStage.RECORDING:
            self._icon.icon = self._create_icon_image("red")
        elif stage in (PipelineStage.IDLE, PipelineStage.COMPLETE):
            self._icon.icon = self._create_icon_image("gray")
        else:
            self._icon.icon = self._create_icon_image("yellow")
        self._icon.menu = self._build_menu()
        # Update tooltip
        if self._pipeline.is_recording:
            from src.utils.audio_utils import format_duration
            elapsed = format_duration(self._pipeline.elapsed_seconds)
            self._icon.title = f"Gravacao em andamento: {elapsed}"
        else:
            self._icon.title = f"Gravacao e Transcricao — {self._status_text}"

    def _on_stage_change(self, stage: PipelineStage):
        self._status_text = stage.value
        self._update_icon()

    def _on_complete(self, output_path: Path):
        self._status_text = f"Concluido: {output_path.name}"
        self._update_icon()
        # Windows toast notification
        try:
            from plyer import notification
            notification.notify(
                title="Transcricao Concluida",
                message=f"Arquivo: {output_path.name}",
                timeout=5,
            )
        except Exception:
            pass  # plyer not installed, skip notification

    def _on_error(self, error: str):
        self._status_text = f"Erro: {error[:50]}"
        self._update_icon()

    def _start_recording(self, icon=None, item=None):
        try:
            self._pipeline.start_recording(trigger="manual")
        except Exception as e:
            log.error(f"Erro ao iniciar gravacao: {e}")

    def _stop_recording(self, icon=None, item=None):
        try:
            self._pipeline.stop_recording()
        except Exception as e:
            log.error(f"Erro ao parar gravacao: {e}")

    def _open_last_transcription(self, icon=None, item=None):
        last = self._rec_logger.get_last_transcription_path()
        if last and last.exists():
            os.startfile(str(last))
        else:
            log.info("Nenhuma transcricao recente encontrada")

    def _open_transcription_folder(self, icon=None, item=None):
        folder = Path(self._settings.transcription_output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder))

    def _open_log(self, icon=None, item=None):
        if LOG_FILE.exists():
            os.startfile(str(LOG_FILE))

    def _open_settings(self, icon=None, item=None):
        log.info("Configuracoes (a implementar)")
        # Will be wired to settings_window.py later

    def _quit(self, icon=None, item=None):
        log.info("Encerrando aplicativo")
        self._process_monitor.stop()
        self._sound_detector.stop()
        self._pipeline.terminate()
        if self._icon:
            self._icon.stop()

    def _on_meet_opened(self):
        log.info("Meet detectado — ativando modo escuta")
        # Start listening for join/leave sounds via audio
        # The sound detector needs to be fed audio data from the loopback
        # This will be done by starting a lightweight listen-only capture

    def _on_meeting_joined(self):
        log.info("Reuniao iniciada (som detectado) — iniciando gravacao")
        if not self._pipeline.is_recording:
            self._pipeline.start_recording(trigger="meet_auto")

    def _on_meeting_left(self):
        log.info("Reuniao encerrada (som detectado) — parando gravacao")
        if self._pipeline.is_recording:
            self._pipeline.stop_recording()

    def run(self):
        """Start the system tray application."""
        # Start monitors
        if self._settings.auto_detect_meet:
            self._process_monitor.start()

        self._icon = pystray.Icon(
            name="gravacao-transcricao",
            icon=self._create_icon_image("gray"),
            title="Gravacao e Transcricao — Pronto",
            menu=self._build_menu(),
        )

        log.info("App iniciado no System Tray")
        self._icon.run()  # This blocks

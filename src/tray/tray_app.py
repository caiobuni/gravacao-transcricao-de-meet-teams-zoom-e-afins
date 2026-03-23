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
from src.core.pipeline import Pipeline, PipelineStage, PROCESSING_STAGES
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
        self._pipeline.on_queue_change = self._on_queue_change

        if settings.auto_detect_meet:
            self._process_monitor.on_meet_opened = self._on_meet_opened
            self._process_monitor.on_meet_closed = self._on_meet_closed
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
            "green": (50, 180, 50, 255),
        }
        fill = colors.get(color, colors["gray"])
        draw.ellipse([4, 4, size - 4, size - 4], fill=fill, outline=(255, 255, 255, 200), width=2)
        return img

    def _build_menu(self) -> pystray.Menu:
        is_rec = self._pipeline.is_recording
        is_vexa = self._pipeline.is_vexa_active
        is_paused = self._pipeline.is_paused
        stats = self._pipeline.task_queue.get_stats()
        pending = stats["pending"]
        in_progress = stats["in_progress"]

        # Dynamic pause/resume
        if is_paused:
            pause_label = "Retomar Processamento"
            pause_action = self._resume_processing
        else:
            pause_label = "Pausar Processamento"
            pause_action = self._pause_processing

        # Queue status text
        queue_parts = []
        if in_progress:
            queue_parts.append("1 processando")
        if pending:
            queue_parts.append(f"{pending} pendente(s)")
        queue_text = f"Fila: {', '.join(queue_parts)}" if queue_parts else "Fila: vazia"

        has_work = pending > 0 or in_progress > 0

        # Vexa bot label
        if is_vexa:
            vexa_label = "Parar bot Vexa"
            vexa_action = self._stop_vexa_bot
            vexa_enabled = True
        else:
            vexa_label = "Enviar bot para reuniao"
            vexa_action = self._start_vexa_bot
            vexa_enabled = not is_rec

        return pystray.Menu(
            pystray.MenuItem("Iniciar Gravacao", self._start_recording, enabled=not is_rec and not is_vexa),
            pystray.MenuItem("Parar Gravacao", self._stop_recording, enabled=is_rec),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(vexa_label, vexa_action, enabled=vexa_enabled),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(pause_label, pause_action, enabled=has_work or is_paused),
            pystray.MenuItem(queue_text, None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(f"Status: {self._status_text}", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Ultima transcricao", self._open_last_transcription),
            pystray.MenuItem("Ver fila de tarefas", self._open_task_queue),
            pystray.MenuItem("Abrir pasta de transcricoes", self._open_transcription_folder),
            pystray.MenuItem("Transcrever audio anterior", self._open_recordings),
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
            color = "red"
        elif stage in (PipelineStage.VEXA_WAITING, PipelineStage.VEXA_RECORDING):
            color = "red"
        elif stage == PipelineStage.PAUSED:
            color = "yellow"
        elif stage in PROCESSING_STAGES or stage == PipelineStage.QUEUED:
            color = "green"
        else:
            color = "gray"

        self._icon.icon = self._create_icon_image(color)
        self._icon.menu = self._build_menu()

        # Tooltip
        if self._pipeline.is_vexa_active and self._pipeline.is_recording:
            self._icon.title = f"Vexa + Gravacao: {self._status_text}"
        elif self._pipeline.is_vexa_active:
            self._icon.title = f"Vexa: {self._status_text}"
        elif self._pipeline.is_recording:
            from src.utils.audio_utils import format_duration
            elapsed = format_duration(self._pipeline.elapsed_seconds)
            self._icon.title = f"Gravacao em andamento: {elapsed}"
        else:
            stats = self._pipeline.task_queue.get_stats()
            pending = stats["pending"]
            if pending > 0:
                self._icon.title = f"Gravacao e Transcricao — {self._status_text} (Fila: {pending})"
            else:
                self._icon.title = f"Gravacao e Transcricao — {self._status_text}"

    def _on_stage_change(self, stage: PipelineStage):
        self._status_text = stage.value
        self._update_icon()

    def _on_complete(self, output_path: Path):
        self._status_text = f"Concluido: {output_path.name}"
        self._update_icon()
        try:
            from plyer import notification
            notification.notify(
                title="Transcricao Concluida",
                message=f"Arquivo: {output_path.name}",
                timeout=5,
            )
        except Exception:
            pass

    def _on_error(self, error: str):
        self._status_text = f"Erro: {error[:50]}"
        self._update_icon()

    def _on_queue_change(self):
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

    def _start_vexa_bot(self, icon=None, item=None):
        """Abre dialog para colar URL e envia bot Vexa."""
        from src.gui.tk_root import run_on_tk
        run_on_tk(self._show_vexa_dialog)

    def _show_vexa_dialog(self):
        import customtkinter as ctk
        dialog = ctk.CTkInputDialog(
            text="Cole a URL da reuniao (Meet, Teams ou Zoom):",
            title="Enviar bot Vexa",
        )
        url = dialog.get_input()
        if url and url.strip():
            def _send():
                try:
                    self._pipeline.start_vexa_bot(url.strip())
                except Exception as e:
                    log.error("Erro ao enviar bot Vexa: %s", e)
                    self._status_text = f"Erro Vexa: {str(e)[:40]}"
                    self._update_icon()
            threading.Thread(target=_send, daemon=True).start()

    def _stop_vexa_bot(self, icon=None, item=None):
        """Para o bot Vexa e enfileira transcricao."""
        def _stop():
            try:
                self._pipeline.stop_vexa_bot()
            except Exception as e:
                log.error("Erro ao parar bot Vexa: %s", e)
                self._status_text = f"Erro Vexa: {str(e)[:40]}"
                self._update_icon()

        threading.Thread(target=_stop, daemon=True).start()

    def _pause_processing(self, icon=None, item=None):
        self._pipeline.pause_processing()
        self._status_text = "Pausado"
        self._update_icon()

    def _resume_processing(self, icon=None, item=None):
        self._pipeline.resume_processing()
        self._status_text = "Retomando..."
        self._update_icon()

    def _open_last_transcription(self, icon=None, item=None):
        last = self._rec_logger.get_last_transcription_path()
        if last and last.exists():
            os.startfile(str(last))
        else:
            log.info("Nenhuma transcricao recente encontrada")

    def _open_task_queue(self, icon=None, item=None):
        import tempfile
        md = self._pipeline.task_queue.to_markdown()
        tmp = Path(tempfile.gettempdir()) / "fila_tarefas.md"
        tmp.write_text(md, encoding="utf-8")
        os.startfile(str(tmp))

    def _open_transcription_folder(self, icon=None, item=None):
        folder = Path(self._settings.transcription_output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder))

    def _open_recordings(self, icon=None, item=None):
        """Abre janela para selecionar gravacao e transcrever localmente."""
        from src.gui.tk_root import run_on_tk

        def _show():
            from src.gui.recordings_window import RecordingsWindow
            win = RecordingsWindow(on_transcribe=self._pipeline.enqueue_manual_transcription)
            win.show()

        run_on_tk(_show)

    def _open_log(self, icon=None, item=None):
        if LOG_FILE.exists():
            os.startfile(str(LOG_FILE))

    def _open_settings(self, icon=None, item=None):
        log.info("Configuracoes (a implementar)")

    def _quit(self, icon=None, item=None):
        log.info("Encerrando aplicativo")
        self._process_monitor.stop()
        self._sound_detector.stop()
        self._pipeline.terminate()
        if self._icon:
            self._icon.stop()

    def _on_meet_opened(self):
        log.info("Meet detectado — ativando modo escuta")

    def _on_meet_closed(self):
        """Meet fechado — para gravacao/bot Vexa (modo hibrido para ambos)."""
        log.info("Google Meet encerrado — verificando gravacao ativa")
        try:
            if self._pipeline.is_vexa_active:
                threading.Thread(target=self._pipeline.stop_vexa_bot, daemon=True).start()
            elif self._pipeline.is_recording:
                self._pipeline.stop_recording()
        except Exception:
            log.error("Erro ao tratar fechamento do Meet", exc_info=True)

    def _on_meeting_joined(self):
        log.info("Reuniao iniciada (som detectado) — iniciando gravacao")
        try:
            if not self._pipeline.is_recording:
                self._pipeline.start_recording(trigger="meet_auto")
        except Exception:
            log.error("Erro ao iniciar gravacao automatica", exc_info=True)

    def _on_meeting_left(self):
        log.info("Reuniao encerrada (som detectado) — parando gravacao")
        try:
            if self._pipeline.is_vexa_active:
                threading.Thread(target=self._pipeline.stop_vexa_bot, daemon=True).start()
            elif self._pipeline.is_recording:
                self._pipeline.stop_recording()
        except Exception:
            log.error("Erro ao tratar saida da reuniao", exc_info=True)

    def run(self):
        """Start the system tray application."""
        if self._settings.auto_detect_meet:
            self._process_monitor.start()
            self._sound_detector.start()

        self._icon = pystray.Icon(
            name="gravacao-transcricao",
            icon=self._create_icon_image("gray"),
            title="Gravacao e Transcricao — Pronto",
            menu=self._build_menu(),
        )

        log.info("App iniciado no System Tray")
        self._icon.run()

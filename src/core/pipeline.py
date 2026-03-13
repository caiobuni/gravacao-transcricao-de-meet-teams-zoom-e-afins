import logging
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

from src.config.constants import RECORDINGS_DIR, TRANSCRIPTION_OUTPUT_DIR
from src.config.settings import Settings
from src.core.audio_capture import DualTrackCapture
from src.core.audio_preprocessing import prepare_for_transcription
from src.core.transcriber_base import TranscriberBase, TranscriptionSegment
from src.core.diarizer import Diarizer
from src.core.aligner import align_dual_track
from src.core.formatter import to_markdown, to_srt
from src.core.speaker_identifier import SpeakerIdentifier
from src.core.task_queue import TaskQueue, TaskStatus, TranscriptionTask
from src.utils.logger import RecordingLogger

log = logging.getLogger(__name__)

class PipelineStage(Enum):
    IDLE = "Pronto"
    RECORDING = "Gravando..."
    QUEUED = "Na fila..."
    PREPROCESSING = "Preprocessando audio..."
    TRANSCRIBING_MIC = "Transcrevendo audio do microfone..."
    TRANSCRIBING_SPEAKERS = "Transcrevendo audio dos participantes..."
    DIARIZING = "Identificando falantes..."
    ALIGNING = "Alinhando transcricoes..."
    IDENTIFYING = "Identificando nomes..."
    FORMATTING = "Gerando transcricao..."
    CLEANING = "Limpando arquivos..."
    COMPLETE = "Concluido!"
    FAILED = "Falha na transcricao"
    PAUSED = "Pausado"

PROCESSING_STAGES = {
    PipelineStage.PREPROCESSING,
    PipelineStage.TRANSCRIBING_MIC,
    PipelineStage.TRANSCRIBING_SPEAKERS,
    PipelineStage.DIARIZING,
    PipelineStage.ALIGNING,
    PipelineStage.IDENTIFYING,
    PipelineStage.FORMATTING,
    PipelineStage.CLEANING,
}

class Pipeline:
    def __init__(self, settings: Settings | None = None):
        self._settings = settings or Settings.load()
        self._capture = DualTrackCapture()
        self._rec_logger = RecordingLogger()
        self._speaker_id = SpeakerIdentifier()
        self._transcriber: TranscriberBase | None = None
        self._diarizer: Diarizer | None = None
        self._stage = PipelineStage.IDLE

        # Task queue
        self._task_queue = TaskQueue()
        self._current_task_id: str | None = None

        # Pause: Event SET = running, CLEAR = paused
        self._run_event = threading.Event()
        self._run_event.set()
        self._paused = False
        self._auto_paused = False

        # Processor loop control
        self._processor_stop = threading.Event()

        # Callbacks
        self.on_stage_change: Callable[[PipelineStage], None] | None = None
        self.on_progress: Callable[[float], None] | None = None
        self.on_complete: Callable[[Path], None] | None = None
        self.on_error: Callable[[str], None] | None = None
        self.on_queue_change: Callable[[], None] | None = None

        # Crash recovery
        self._task_queue.mark_in_progress_as_pending()
        self._task_queue.validate_audio_files()
        self._task_queue.remove_completed()

        # Start processor loop
        self._processor_thread = threading.Thread(
            target=self._processor_loop,
            name="pipeline-processor",
            daemon=True,
        )
        self._processor_thread.start()

    def _set_stage(self, stage: PipelineStage):
        self._stage = stage
        if self.on_stage_change:
            self.on_stage_change(stage)

    def _get_transcriber(self) -> TranscriberBase:
        """Get the configured transcriber, with fallback chain."""
        if self._transcriber and self._transcriber.is_available():
            return self._transcriber

        engine = self._settings.transcription_engine

        if engine == "whisper":
            from src.core.transcriber_whisper import WhisperTranscriber
            t = WhisperTranscriber(model_size=self._settings.whisper_model)
            if t.is_available():
                self._transcriber = t
                return t

        from src.core.transcriber_groq import GroqTranscriber
        t = GroqTranscriber()
        if t.is_available():
            self._transcriber = t
            return t

        raise RuntimeError("Nenhum motor de transcricao disponivel")

    # --- Recording ---

    def start_recording(self, trigger: str = "manual"):
        """Start dual-track recording."""
        # Auto-pause processing to free hardware for recording
        if not self._paused and self.is_processing:
            self._auto_paused = True
            self._run_event.clear()
            log.info("Processamento auto-pausado para gravacao")

        self._set_stage(PipelineStage.RECORDING)
        speakers_path, mic_path = self._capture.start()
        self._rec_logger.log_start(trigger)
        log.info("Gravacao iniciada: speakers=%s, mic=%s", speakers_path, mic_path)

    def stop_recording(self):
        """Stop recording and enqueue transcription task."""
        speakers_path, mic_path, start_time, duration = self._capture.stop()
        self._rec_logger.log_stop(duration, speakers_path.name)
        log.info("Gravacao finalizada: %.1fs", duration)

        task = self._task_queue.add_task(speakers_path, mic_path, start_time, duration)
        self._set_stage(PipelineStage.QUEUED)

        if self.on_queue_change:
            self.on_queue_change()

        log.info("Tarefa enfileirada: %s", task.id)

        # Auto-resume if was auto-paused (not manually paused)
        if self._auto_paused:
            self._auto_paused = False
            self._run_event.set()
            log.info("Processamento auto-retomado apos gravacao")

    # --- Processor loop ---

    def _processor_loop(self):
        """Loop continuo que processa tarefas da fila uma por vez."""
        log.info("Processor loop iniciado")
        while not self._processor_stop.is_set():
            self._run_event.wait()

            if self._processor_stop.is_set():
                break

            task = self._task_queue.get_next_pending()
            if task is None:
                self._processor_stop.wait(timeout=2.0)
                continue

            self._current_task_id = task.id
            self._task_queue.update_status(task.id, TaskStatus.IN_PROGRESS)

            try:
                self._process_task(task)
            except Exception as e:
                log.error("Tarefa %s falhou: %s", task.id, e, exc_info=True)
                self._task_queue.update_status(
                    task.id, TaskStatus.FAILED, error=str(e)
                )
                self._set_stage(PipelineStage.FAILED)
                self._rec_logger.log_transcription_failed(str(e))
                self._rec_logger.log_audio_kept(
                    Path(task.speakers_path).name, f"falha: {e}"
                )
                if self.on_error:
                    self.on_error(str(e))
            finally:
                self._current_task_id = None
                if self.on_queue_change:
                    self.on_queue_change()

            if not self._task_queue.has_pending_work():
                self._set_stage(PipelineStage.IDLE)

        log.info("Processor loop encerrado")

    def _check_pause(self, task_id: str) -> bool:
        """Bloqueia se pausado. Retorna True para continuar, False para abortar."""
        if self._run_event.is_set():
            return True

        log.info("Pipeline pausado entre estagios (tarefa %s)", task_id)
        prev_stage = self._stage
        self._set_stage(PipelineStage.PAUSED)

        while not self._run_event.is_set():
            if self._processor_stop.is_set():
                return False
            self._run_event.wait(timeout=0.5)

        log.info("Pipeline retomado (tarefa %s)", task_id)
        self._set_stage(prev_stage)
        return True

    def _process_task(self, task: TranscriptionTask):
        """Executa o pipeline completo para uma tarefa da fila."""
        speakers_path = Path(task.speakers_path)
        mic_path = Path(task.mic_path)
        start_time = datetime.fromisoformat(task.start_time)
        duration = task.duration

        if not speakers_path.exists() or not mic_path.exists():
            raise FileNotFoundError(
                f"Arquivos de audio nao encontrados: {speakers_path}, {mic_path}"
            )

        transcriber = self._get_transcriber()
        engine_name = transcriber.name

        # Preprocess
        self._set_stage(PipelineStage.PREPROCESSING)
        self._task_queue.update_status(
            task.id, TaskStatus.IN_PROGRESS, current_stage="PREPROCESSING"
        )
        speakers_16k = prepare_for_transcription(speakers_path)
        mic_16k = prepare_for_transcription(mic_path)

        if not self._check_pause(task.id):
            return

        # Transcribe mic track
        self._set_stage(PipelineStage.TRANSCRIBING_MIC)
        self._task_queue.update_status(
            task.id, TaskStatus.IN_PROGRESS, current_stage="TRANSCRIBING_MIC"
        )
        language = self._settings.language
        mic_segments = transcriber.transcribe(
            mic_16k, language=language, on_progress=self.on_progress
        )

        if not self._check_pause(task.id):
            return

        # Transcribe speakers track
        self._set_stage(PipelineStage.TRANSCRIBING_SPEAKERS)
        self._task_queue.update_status(
            task.id, TaskStatus.IN_PROGRESS, current_stage="TRANSCRIBING_SPEAKERS"
        )
        speakers_segments = transcriber.transcribe(
            speakers_16k, language=language, on_progress=self.on_progress
        )

        if not self._check_pause(task.id):
            return

        # Diarize speakers track
        self._set_stage(PipelineStage.DIARIZING)
        self._task_queue.update_status(
            task.id, TaskStatus.IN_PROGRESS, current_stage="DIARIZING"
        )
        diarization = []
        try:
            if not self._diarizer:
                self._diarizer = Diarizer()
            if self._diarizer.is_available():
                diarization = self._diarizer.diarize(speakers_16k)
        except Exception as e:
            log.warning("Diarizacao falhou: %s", e)

        if not self._check_pause(task.id):
            return

        # Align
        self._set_stage(PipelineStage.ALIGNING)
        self._task_queue.update_status(
            task.id, TaskStatus.IN_PROGRESS, current_stage="ALIGNING"
        )
        aligned = align_dual_track(
            mic_segments=mic_segments,
            speakers_segments=speakers_segments,
            diarization=diarization,
            user_name=self._settings.user_name,
        )

        if not self._check_pause(task.id):
            return

        # Identify speakers
        self._set_stage(PipelineStage.IDENTIFYING)
        self._task_queue.update_status(
            task.id, TaskStatus.IN_PROGRESS, current_stage="IDENTIFYING"
        )
        name_mapping = self._speaker_id.identify_by_context(aligned)
        if name_mapping:
            for seg in aligned:
                if seg.speaker in name_mapping:
                    seg.speaker = name_mapping[seg.speaker]

        if not self._check_pause(task.id):
            return

        # Format and save
        self._set_stage(PipelineStage.FORMATTING)
        self._task_queue.update_status(
            task.id, TaskStatus.IN_PROGRESS, current_stage="FORMATTING"
        )
        output_dir = Path(self._settings.transcription_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = start_time.strftime("%Y%m%d-%H%M") + ".md"
        output_path = output_dir / filename

        markdown = to_markdown(aligned, start_time, duration, engine_name)
        output_path.write_text(markdown, encoding="utf-8")

        self._rec_logger.log_transcription(filename, engine_name)
        log.info("Transcricao salva: %s", output_path)

        # Clean up
        self._set_stage(PipelineStage.CLEANING)
        self._task_queue.update_status(
            task.id, TaskStatus.IN_PROGRESS, current_stage="CLEANING"
        )
        if self._settings.auto_delete_audio:
            for f in [speakers_path, mic_path, speakers_16k, mic_16k]:
                if f.exists():
                    f.unlink()
            self._rec_logger.log_audio_deleted(speakers_path.name)

        # Mark complete
        self._task_queue.update_status(
            task.id,
            TaskStatus.COMPLETED,
            output_path=str(output_path),
            current_stage="COMPLETE",
        )
        self._set_stage(PipelineStage.COMPLETE)

        if self.on_complete:
            self.on_complete(output_path)

    # --- Pause / Resume ---

    def pause_processing(self):
        if not self._paused:
            self._paused = True
            self._auto_paused = False  # manual pause takes precedence
            self._run_event.clear()
            log.info("Processamento pausado pelo usuario")

    def resume_processing(self):
        if self._paused:
            self._paused = False
            self._run_event.set()
            log.info("Processamento retomado pelo usuario")

    # --- Properties ---

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def task_queue(self) -> TaskQueue:
        return self._task_queue

    @property
    def is_processing(self) -> bool:
        return self._current_task_id is not None

    @property
    def stage(self) -> PipelineStage:
        return self._stage

    @property
    def is_recording(self) -> bool:
        return self._capture.is_recording

    @property
    def elapsed_seconds(self) -> float:
        return self._capture.elapsed_seconds

    def terminate(self):
        """Shutdown graceful do processador."""
        self._processor_stop.set()
        self._run_event.set()  # desbloqueia se pausado

        if self._processor_thread and self._processor_thread.is_alive():
            self._processor_thread.join(timeout=5.0)

        if self._capture.is_recording:
            self._capture.stop()
        self._capture.terminate()

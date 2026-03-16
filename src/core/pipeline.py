import logging
import os
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

from src.config.constants import RECORDINGS_DIR, VEXA_RECORDINGS_DIR, TRANSCRIPTION_OUTPUT_DIR
from src.config.settings import Settings
from src.core.audio_capture import DualTrackCapture
from src.core.audio_preprocessing import prepare_for_transcription
from src.core.transcriber_base import TranscriberBase, TranscriptionSegment
from src.core.diarizer import Diarizer
from src.core.aligner import AlignedSegment, align_dual_track
from src.core.formatter import to_markdown, to_srt
from src.core.speaker_identifier import SpeakerIdentifier
from src.core.task_queue import TaskQueue, TaskStatus, TranscriptionTask
from src.core.vexa_client import VexaClient, VexaSegment
from src.utils.logger import RecordingLogger
from src.utils.meeting_url_parser import parse_meeting_url

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
    VEXA_WAITING = "Aguardando bot entrar na reuniao..."
    VEXA_RECORDING = "Bot gravando reuniao..."
    VEXA_FETCHING = "Baixando transcricao do Vexa..."
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
    PipelineStage.VEXA_FETCHING,
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

        # Vexa state
        self._vexa_client: VexaClient | None = None
        self._vexa_platform: str = ""
        self._vexa_meeting_id: str = ""
        self._vexa_active = False

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

    # --- Vexa bot ---

    def _get_vexa_client(self) -> VexaClient:
        """Obtem ou cria o cliente Vexa."""
        if self._vexa_client:
            return self._vexa_client
        api_key = self._settings.vexa_api_key or os.environ.get("VEXA_API_KEY", "")
        if not api_key:
            raise RuntimeError("VEXA_API_KEY nao configurada")
        self._vexa_client = VexaClient(api_key, self._settings.vexa_base_url)
        return self._vexa_client

    def start_vexa_bot(self, meeting_url: str):
        """Envia bot Vexa para uma reuniao."""
        info = parse_meeting_url(meeting_url)
        client = self._get_vexa_client()

        self._set_stage(PipelineStage.VEXA_WAITING)
        client.create_bot(
            platform=info.platform,
            meeting_id=info.meeting_id,
            language=self._settings.language,
            passcode=info.passcode,
            bot_name=self._settings.vexa_bot_name,
            bot_image=self._settings.vexa_bot_image,
        )
        self._vexa_platform = info.platform
        self._vexa_meeting_id = info.meeting_id
        self._vexa_active = True
        self._set_stage(PipelineStage.VEXA_RECORDING)
        log.info("Bot Vexa enviado: %s/%s", info.platform, info.meeting_id)

    def stop_vexa_bot(self):
        """Para o bot Vexa e enfileira transcricao."""
        if not self._vexa_active:
            return

        client = self._get_vexa_client()
        platform = self._vexa_platform
        meeting_id = self._vexa_meeting_id

        # Para o bot
        try:
            client.stop_bot(platform, meeting_id)
        except Exception as e:
            log.warning("Erro ao parar bot Vexa: %s", e)

        self._set_stage(PipelineStage.VEXA_FETCHING)

        # Busca transcricao
        transcript = client.get_transcript(platform, meeting_id)
        vexa_json = VexaClient.segments_to_json(transcript.segments)

        # Calcula duracao
        start_time = transcript.start_time or datetime.now()
        end_time = transcript.end_time or datetime.now()
        duration = (end_time - start_time).total_seconds() if transcript.end_time else 0.0

        # Baixa audio se configurado
        audio_path = None
        if self._settings.vexa_auto_download and transcript.recordings:
            VEXA_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            rec = transcript.recordings[0]
            rec_id = rec.get("id", "")
            media_files = rec.get("media_files", [])
            if rec_id and media_files:
                media_id = media_files[0].get("id", "") if isinstance(media_files[0], dict) else ""
                if media_id:
                    fname = start_time.strftime("%Y%m%d_%H%M%S") + "_vexa.wav"
                    audio_path = VEXA_RECORDINGS_DIR / fname
                    try:
                        client.download_recording(rec_id, media_id, audio_path)
                    except Exception as e:
                        log.warning("Falha ao baixar audio Vexa: %s", e)
                        audio_path = None

        # Enfileira tarefa
        task = self._task_queue.add_vexa_task(
            platform=platform,
            meeting_id=meeting_id,
            start_time=start_time,
            duration=duration,
            vexa_transcript_json=vexa_json,
            audio_path=audio_path,
        )

        self._vexa_active = False
        self._vexa_platform = ""
        self._vexa_meeting_id = ""
        self._set_stage(PipelineStage.QUEUED)

        if self.on_queue_change:
            self.on_queue_change()

        log.info("Tarefa Vexa enfileirada: %s", task.id)

    @property
    def is_vexa_active(self) -> bool:
        return self._vexa_active

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
        if task.source == "vexa":
            return self._process_vexa_task(task)
        if task.source == "manual":
            return self._process_manual_task(task)
        return self._process_loopback_task(task)

    def _process_vexa_task(self, task: TranscriptionTask):
        """Processa tarefa originada do Vexa (segmentos ja identificados)."""
        start_time = datetime.fromisoformat(task.start_time)
        duration = task.duration

        # Converte segmentos Vexa para AlignedSegments
        self._set_stage(PipelineStage.FORMATTING)
        self._task_queue.update_status(
            task.id, TaskStatus.IN_PROGRESS, current_stage="FORMATTING"
        )

        vexa_segments = VexaClient.segments_from_json(task.vexa_transcript_json)
        aligned: list[AlignedSegment] = []
        for seg in vexa_segments:
            aligned.append(AlignedSegment(
                start=seg.start_time,
                end=seg.end_time,
                speaker=seg.speaker,
                text=seg.text,
                is_user=False,
            ))
        aligned.sort(key=lambda s: s.start)

        engine_name = "Vexa"

        # Format and save
        output_dir = Path(self._settings.transcription_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = start_time.strftime("%Y%m%d-%H%M") + ".md"
        output_path = output_dir / filename

        markdown = to_markdown(aligned, start_time, duration, engine_name)
        output_path.write_text(markdown, encoding="utf-8")

        self._rec_logger.log_transcription(filename, engine_name)
        log.info("Transcricao Vexa salva: %s", output_path)

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

    def enqueue_manual_transcription(self, audio_path: Path) -> TranscriptionTask:
        """Enfileira um arquivo de audio para transcricao local com whisper + pyannote."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Arquivo nao encontrado: {audio_path}")

        import wave
        try:
            with wave.open(str(audio_path), "rb") as wf:
                duration = wf.getnframes() / wf.getframerate()
        except Exception:
            duration = 0.0

        start_time = datetime.fromtimestamp(audio_path.stat().st_mtime)
        task = self._task_queue.add_manual_task(audio_path, start_time, duration)

        if self.on_queue_change:
            self.on_queue_change()

        log.info("Transcricao manual enfileirada: %s", audio_path.name)
        return task

    def _process_manual_task(self, task: TranscriptionTask):
        """Processa tarefa manual (single-track: whisper + pyannote, sem mic)."""
        audio_path = Path(task.speakers_path)
        start_time = datetime.fromisoformat(task.start_time)
        duration = task.duration

        if not audio_path.exists():
            raise FileNotFoundError(f"Arquivo de audio nao encontrado: {audio_path}")

        transcriber = self._get_transcriber()
        engine_name = transcriber.name

        # Preprocess
        self._set_stage(PipelineStage.PREPROCESSING)
        self._task_queue.update_status(
            task.id, TaskStatus.IN_PROGRESS, current_stage="PREPROCESSING"
        )
        audio_16k = prepare_for_transcription(audio_path)

        if not self._check_pause(task.id):
            return

        # Transcribe
        self._set_stage(PipelineStage.TRANSCRIBING_SPEAKERS)
        self._task_queue.update_status(
            task.id, TaskStatus.IN_PROGRESS, current_stage="TRANSCRIBING_SPEAKERS"
        )
        language = self._settings.language
        speakers_segments = transcriber.transcribe(
            audio_16k, language=language, on_progress=self.on_progress
        )

        if not self._check_pause(task.id):
            return

        # Free GPU memory before diarization
        self._transcriber = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                log.info("VRAM liberada apos transcricao")
        except Exception:
            pass

        # Diarize
        self._set_stage(PipelineStage.DIARIZING)
        self._task_queue.update_status(
            task.id, TaskStatus.IN_PROGRESS, current_stage="DIARIZING"
        )
        diarization = []
        try:
            if not self._diarizer:
                self._diarizer = Diarizer()
            if self._diarizer.is_available():
                diarization = self._diarizer.diarize(audio_16k)
        except Exception as e:
            log.warning("Diarizacao falhou: %s", e)
        finally:
            self._diarizer = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        if not self._check_pause(task.id):
            return

        # Align (single-track: mic vazio, tudo via diarizacao)
        self._set_stage(PipelineStage.ALIGNING)
        self._task_queue.update_status(
            task.id, TaskStatus.IN_PROGRESS, current_stage="ALIGNING"
        )
        aligned = align_dual_track(
            mic_segments=[],
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
        log.info("Transcricao manual salva: %s", output_path)

        # Clean up temp 16k file (nao deleta o original)
        if audio_16k != audio_path and audio_16k.exists():
            audio_16k.unlink()

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

    def _process_loopback_task(self, task: TranscriptionTask):
        """Executa o pipeline completo para uma tarefa de gravacao loopback."""
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

        # Free GPU memory before diarization (whisper + pyannote don't fit together)
        self._transcriber = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                log.info("VRAM liberada apos transcricao")
        except Exception:
            pass

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
        finally:
            # Free diarizer GPU memory
            self._diarizer = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

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

        if self._vexa_active:
            try:
                self._get_vexa_client().stop_bot(
                    self._vexa_platform, self._vexa_meeting_id
                )
            except Exception:
                pass
            self._vexa_active = False

        if self._capture.is_recording:
            self._capture.stop()
        self._capture.terminate()

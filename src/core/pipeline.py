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
from src.utils.logger import RecordingLogger

log = logging.getLogger(__name__)

class PipelineStage(Enum):
    IDLE = "Pronto"
    RECORDING = "Gravando..."
    PREPROCESSING = "Preprocessando áudio..."
    TRANSCRIBING_MIC = "Transcrevendo áudio do microfone..."
    TRANSCRIBING_SPEAKERS = "Transcrevendo áudio dos participantes..."
    DIARIZING = "Identificando falantes..."
    ALIGNING = "Alinhando transcrições..."
    IDENTIFYING = "Identificando nomes..."
    FORMATTING = "Gerando transcrição..."
    CLEANING = "Limpando arquivos..."
    COMPLETE = "Concluído!"
    FAILED = "Falha na transcrição"

class Pipeline:
    def __init__(self, settings: Settings | None = None):
        self._settings = settings or Settings.load()
        self._capture = DualTrackCapture()
        self._rec_logger = RecordingLogger()
        self._speaker_id = SpeakerIdentifier()
        self._transcriber: TranscriberBase | None = None
        self._diarizer: Diarizer | None = None
        self._stage = PipelineStage.IDLE
        self._processing_thread: threading.Thread | None = None

        # Callbacks
        self.on_stage_change: Callable[[PipelineStage], None] | None = None
        self.on_progress: Callable[[float], None] | None = None
        self.on_complete: Callable[[Path], None] | None = None
        self.on_error: Callable[[str], None] | None = None

    def _set_stage(self, stage: PipelineStage):
        self._stage = stage
        if self.on_stage_change:
            self.on_stage_change(stage)

    def _get_transcriber(self) -> TranscriberBase:
        """Get the configured transcriber, with fallback chain."""
        if self._transcriber and self._transcriber.is_available():
            return self._transcriber

        engine = self._settings.transcription_engine

        # Try configured engine first
        if engine == "qwen":
            from src.core.transcriber_qwen import QwenTranscriber
            t = QwenTranscriber(model_name=self._settings.qwen_model, aligner_name=self._settings.qwen_aligner)
            if t.is_available():
                self._transcriber = t
                return t

        if engine == "whisper" or (engine == "qwen" and not self._transcriber):
            from src.core.transcriber_whisper import WhisperTranscriber
            t = WhisperTranscriber(model_size=self._settings.whisper_model, compute_type=self._settings.whisper_compute_type)
            if t.is_available():
                self._transcriber = t
                return t

        # Fallback to Groq API
        from src.core.transcriber_groq import GroqTranscriber
        t = GroqTranscriber()
        if t.is_available():
            self._transcriber = t
            return t

        raise RuntimeError("Nenhum motor de transcrição disponível")

    def start_recording(self, trigger: str = "manual"):
        """Start dual-track recording."""
        self._set_stage(PipelineStage.RECORDING)
        speakers_path, mic_path = self._capture.start()
        self._rec_logger.log_start(trigger)
        log.info(f"Gravação iniciada: speakers={speakers_path}, mic={mic_path}")

    def stop_recording(self):
        """Stop recording and start transcription pipeline in background."""
        speakers_path, mic_path, start_time, duration = self._capture.stop()
        self._rec_logger.log_stop(duration, speakers_path.name)
        log.info(f"Gravação finalizada: {duration:.1f}s")

        # Process in background
        self._processing_thread = threading.Thread(
            target=self._process,
            args=(speakers_path, mic_path, start_time, duration),
            daemon=True,
        )
        self._processing_thread.start()

    def _process(self, speakers_path: Path, mic_path: Path, start_time: datetime, duration: float):
        """Run the full transcription pipeline."""
        try:
            transcriber = self._get_transcriber()
            engine_name = transcriber.name

            # Preprocess
            self._set_stage(PipelineStage.PREPROCESSING)
            speakers_16k = prepare_for_transcription(speakers_path)
            mic_16k = prepare_for_transcription(mic_path)

            # Transcribe mic track
            self._set_stage(PipelineStage.TRANSCRIBING_MIC)
            language = self._settings.language
            mic_segments = transcriber.transcribe(mic_16k, language=language, on_progress=self.on_progress)

            # Transcribe speakers track
            self._set_stage(PipelineStage.TRANSCRIBING_SPEAKERS)
            speakers_segments = transcriber.transcribe(speakers_16k, language=language, on_progress=self.on_progress)

            # Diarize speakers track
            self._set_stage(PipelineStage.DIARIZING)
            diarization = []
            try:
                if not self._diarizer:
                    self._diarizer = Diarizer()
                if self._diarizer.is_available():
                    diarization = self._diarizer.diarize(speakers_16k)
            except Exception as e:
                log.warning(f"Diarização falhou: {e}")

            # Align
            self._set_stage(PipelineStage.ALIGNING)
            aligned = align_dual_track(
                mic_segments=mic_segments,
                speakers_segments=speakers_segments,
                diarization=diarization,
                user_name=self._settings.user_name,
            )

            # Identify speakers
            self._set_stage(PipelineStage.IDENTIFYING)
            name_mapping = self._speaker_id.identify_by_context(aligned)
            if name_mapping:
                for seg in aligned:
                    if seg.speaker in name_mapping:
                        seg.speaker = name_mapping[seg.speaker]

            # Format and save
            self._set_stage(PipelineStage.FORMATTING)
            output_dir = Path(self._settings.transcription_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = start_time.strftime("%Y%m%d-%H%M") + ".md"
            output_path = output_dir / filename

            markdown = to_markdown(aligned, start_time, duration, engine_name)
            output_path.write_text(markdown, encoding="utf-8")

            self._rec_logger.log_transcription(filename, engine_name)
            log.info(f"Transcrição salva: {output_path}")

            # Clean up
            self._set_stage(PipelineStage.CLEANING)
            if self._settings.auto_delete_audio:
                for f in [speakers_path, mic_path, speakers_16k, mic_16k]:
                    if f.exists():
                        f.unlink()
                self._rec_logger.log_audio_deleted(speakers_path.name)

            self._set_stage(PipelineStage.COMPLETE)
            if self.on_complete:
                self.on_complete(output_path)

        except Exception as e:
            log.error(f"Pipeline falhou: {e}", exc_info=True)
            self._set_stage(PipelineStage.FAILED)
            self._rec_logger.log_transcription_failed(str(e))
            self._rec_logger.log_audio_kept(speakers_path.name, f"falha: {e}")
            if self.on_error:
                self.on_error(str(e))

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
        if self._capture.is_recording:
            self._capture.stop()
        self._capture.terminate()

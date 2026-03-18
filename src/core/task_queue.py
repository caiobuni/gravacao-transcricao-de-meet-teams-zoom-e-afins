"""Fila persistente de tarefas de transcricao."""

import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from src.config.constants import TASK_QUEUE_FILE

log = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TranscriptionTask:
    id: str
    speakers_path: str
    mic_path: str
    start_time: str  # ISO format
    duration: float
    status: str = TaskStatus.PENDING.value
    created_at: str = ""
    updated_at: str = ""
    error: str = ""
    output_path: str = ""
    current_stage: str = ""
    retry_count: int = 0
    # Vexa fields
    source: str = "loopback"          # "loopback", "vexa", "hybrid" ou "manual"
    vexa_meeting_id: str = ""
    vexa_platform: str = ""
    vexa_transcript_json: str = ""    # JSON serializado dos segmentos do Vexa
    recording_start_time: str = ""   # ISO format, para alinhamento hybrid

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at


class TaskQueue:
    """Fila persistente thread-safe backed por JSON."""

    def __init__(self, queue_file: Path | None = None):
        self._file = queue_file or TASK_QUEUE_FILE
        self._lock = threading.Lock()
        self._tasks: list[TranscriptionTask] = []
        self._load()

    def _load(self) -> None:
        if not self._file.exists():
            self._tasks = []
            return
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
            self._tasks = [TranscriptionTask(**t) for t in data]
            log.info("Fila carregada: %d tarefa(s)", len(self._tasks))
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            log.error("Erro ao carregar fila: %s", e)
            self._tasks = []

    def _save(self) -> None:
        """Persiste estado atual. Deve ser chamado sob self._lock."""
        self._file.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(t) for t in self._tasks]
        self._file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_task(
        self,
        speakers_path: Path,
        mic_path: Path,
        start_time: datetime,
        duration: float,
    ) -> TranscriptionTask:
        task = TranscriptionTask(
            id=uuid.uuid4().hex[:12],
            speakers_path=str(speakers_path),
            mic_path=str(mic_path),
            start_time=start_time.isoformat(),
            duration=duration,
        )
        with self._lock:
            self._tasks.append(task)
            self._save()
        log.info("Tarefa adicionada: %s (%.0fs)", task.id, duration)
        return task

    def add_vexa_task(
        self,
        platform: str,
        meeting_id: str,
        start_time: datetime,
        duration: float,
        vexa_transcript_json: str,
        audio_path: Path | None = None,
    ) -> TranscriptionTask:
        """Adiciona tarefa originada do Vexa."""
        task = TranscriptionTask(
            id=uuid.uuid4().hex[:12],
            speakers_path=str(audio_path) if audio_path else "",
            mic_path="",
            start_time=start_time.isoformat(),
            duration=duration,
            source="vexa",
            vexa_meeting_id=meeting_id,
            vexa_platform=platform,
            vexa_transcript_json=vexa_transcript_json,
        )
        with self._lock:
            self._tasks.append(task)
            self._save()
        log.info("Tarefa Vexa adicionada: %s (%s/%s)", task.id, platform, meeting_id)
        return task

    def add_hybrid_task(
        self,
        speakers_path: Path,
        mic_path: Path,
        start_time: datetime,
        duration: float,
        vexa_transcript_json: str,
        recording_start_iso: str = "",
    ) -> TranscriptionTask:
        """Adiciona tarefa hibrida: audio local + segmentos Vexa."""
        task = TranscriptionTask(
            id=uuid.uuid4().hex[:12],
            speakers_path=str(speakers_path),
            mic_path=str(mic_path),
            start_time=start_time.isoformat(),
            duration=duration,
            source="hybrid",
            vexa_transcript_json=vexa_transcript_json,
            recording_start_time=recording_start_iso,
        )
        with self._lock:
            self._tasks.append(task)
            self._save()
        log.info("Tarefa hybrid adicionada: %s (%.0fs)", task.id, duration)
        return task

    def add_manual_task(
        self,
        audio_path: Path,
        start_time: datetime,
        duration: float,
    ) -> TranscriptionTask:
        """Adiciona tarefa de transcricao manual (single-track)."""
        task = TranscriptionTask(
            id=uuid.uuid4().hex[:12],
            speakers_path=str(audio_path),
            mic_path="",
            start_time=start_time.isoformat(),
            duration=duration,
            source="manual",
        )
        with self._lock:
            self._tasks.append(task)
            self._save()
        log.info("Tarefa manual adicionada: %s (%s)", task.id, audio_path.name)
        return task

    def get_next_pending(self) -> TranscriptionTask | None:
        with self._lock:
            for t in self._tasks:
                if t.status == TaskStatus.PENDING.value:
                    return t
        return None

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        error: str = "",
        output_path: str = "",
        current_stage: str = "",
    ) -> None:
        with self._lock:
            for t in self._tasks:
                if t.id == task_id:
                    t.status = status.value
                    t.updated_at = datetime.now().isoformat()
                    t.error = error
                    if output_path:
                        t.output_path = output_path
                    if current_stage:
                        t.current_stage = current_stage
                    break
            self._save()

    def mark_in_progress_as_pending(self) -> int:
        """Crash recovery: reseta tasks IN_PROGRESS para PENDING."""
        count = 0
        with self._lock:
            for t in self._tasks:
                if t.status == TaskStatus.IN_PROGRESS.value:
                    t.status = TaskStatus.PENDING.value
                    t.error = ""
                    t.retry_count += 1
                    t.updated_at = datetime.now().isoformat()
                    count += 1
            if count:
                self._save()
        if count:
            log.info("Recuperacao: %d tarefa(s) in_progress -> pending", count)
        return count

    def validate_audio_files(self) -> list[str]:
        """Marca tasks pendentes com arquivos ausentes como FAILED."""
        missing = []
        with self._lock:
            for t in self._tasks:
                if t.status == TaskStatus.PENDING.value:
                    # Tarefas Vexa com transcricao ja pronta nao precisam de audio
                    if t.source == "vexa" and t.vexa_transcript_json:
                        continue
                    # Tarefas hybrid precisam dos arquivos de audio
                    if t.source == "hybrid":
                        if (not t.speakers_path or not Path(t.speakers_path).exists() or
                                not t.mic_path or not Path(t.mic_path).exists()):
                            missing.append(t.id)
                            t.status = TaskStatus.FAILED.value
                            t.error = "Arquivos de audio nao encontrados (hybrid)"
                            t.updated_at = datetime.now().isoformat()
                        continue
                    # Tarefas manuais so precisam do speakers_path
                    if t.source == "manual":
                        if not t.speakers_path or not Path(t.speakers_path).exists():
                            missing.append(t.id)
                            t.status = TaskStatus.FAILED.value
                            t.error = "Arquivo de audio nao encontrado"
                            t.updated_at = datetime.now().isoformat()
                        continue
                    if not Path(t.speakers_path).exists() or not Path(t.mic_path).exists():
                        missing.append(t.id)
                        t.status = TaskStatus.FAILED.value
                        t.error = "Arquivos de audio nao encontrados"
                        t.updated_at = datetime.now().isoformat()
            if missing:
                self._save()
        return missing

    def remove_completed(self, max_age_hours: int = 24) -> int:
        cutoff = datetime.now()
        count = 0
        with self._lock:
            kept = []
            for t in self._tasks:
                if t.status == TaskStatus.COMPLETED.value:
                    try:
                        updated = datetime.fromisoformat(t.updated_at)
                        if (cutoff - updated).total_seconds() > max_age_hours * 3600:
                            count += 1
                            continue
                    except ValueError:
                        pass
                kept.append(t)
            self._tasks = kept
            if count:
                self._save()
        return count

    @property
    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._tasks if t.status == TaskStatus.PENDING.value)

    def get_stats(self) -> dict[str, int]:
        with self._lock:
            stats = {"pending": 0, "in_progress": 0, "completed": 0, "failed": 0}
            for t in self._tasks:
                if t.status in stats:
                    stats[t.status] += 1
            return stats

    def has_pending_work(self) -> bool:
        with self._lock:
            return any(
                t.status in (TaskStatus.PENDING.value, TaskStatus.IN_PROGRESS.value)
                for t in self._tasks
            )

    def to_markdown(self) -> str:
        """Gera relatorio Markdown da fila de tarefas."""
        lines = ["# Fila de Tarefas de Processamento", ""]
        with self._lock:
            if not self._tasks:
                lines.append("Nenhuma tarefa na fila.")
                return "\n".join(lines)

            status_emoji = {
                "pending": "\u23f3",
                "in_progress": "\U0001f504",
                "completed": "\u2705",
                "failed": "\u274c",
            }
            for t in self._tasks:
                emoji = status_emoji.get(t.status, "\u2753")
                dt = t.start_time[:16].replace("T", " ")
                dur_min = t.duration / 60
                source_tags = {"vexa": " [Vexa]", "hybrid": " [Hybrid]", "manual": " [Manual]"}
                source_tag = source_tags.get(t.source, "")
                stage = f" \u2014 {t.current_stage}" if t.current_stage else ""
                error = f" \u2014 Erro: {t.error}" if t.error else ""
                output = f" \u2014 [{Path(t.output_path).name}]" if t.output_path else ""
                lines.append(
                    f"- {emoji} **{dt}** ({dur_min:.0f}min){source_tag} \u2014 {t.status}{stage}{error}{output}"
                )
        return "\n".join(lines)

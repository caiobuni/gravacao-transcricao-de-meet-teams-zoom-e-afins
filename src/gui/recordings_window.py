"""Janela para listar e selecionar gravacoes anteriores para transcricao local."""

import logging
from datetime import datetime
from pathlib import Path

import customtkinter as ctk

from src.config.constants import (
    MIC_SUFFIX,
    RECORDINGS_DIR,
    SPEAKERS_SUFFIX,
    VEXA_RECORDINGS_DIR,
)

log = logging.getLogger(__name__)


class RecordingsWindow:
    def __init__(self, on_transcribe: callable):
        """
        Args:
            on_transcribe: callback(speakers_path: Path, mic_path: Path | None)
                           chamado ao clicar Transcrever.
        """
        self._on_transcribe = on_transcribe
        self._window: ctk.CTkToplevel | None = None
        self._recordings: list[dict] = []

    def show(self):
        if self._window and self._window.winfo_exists():
            self._window.focus()
            return

        self._window = ctk.CTkToplevel()
        self._window.title("Gravacoes anteriores")
        self._window.geometry("650x450")
        self._window.resizable(False, True)

        self._window.grid_columnconfigure(0, weight=1)
        self._window.grid_rowconfigure(1, weight=1)

        # Header
        ctk.CTkLabel(
            self._window,
            text="Selecione uma gravacao para transcrever:",
            font=ctk.CTkFont(size=13),
        ).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")

        # Scrollable frame for recordings list
        self._list_frame = ctk.CTkScrollableFrame(self._window)
        self._list_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self._list_frame.grid_columnconfigure(0, weight=1)

        # Buttons
        btn_frame = ctk.CTkFrame(self._window, fg_color="transparent")
        btn_frame.grid(row=2, column=0, padx=10, pady=(5, 10), sticky="ew")
        btn_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(
            btn_frame, text="Atualizar lista", width=120, command=self._refresh
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkButton(
            btn_frame, text="Fechar", width=80, fg_color="gray", command=self._close
        ).grid(row=0, column=1, padx=(10, 0))

        self._selected_var = ctk.IntVar(value=-1)
        self._refresh()

    def _scan_recordings(self) -> list[dict]:
        """Escaneia diretorios e agrupa pares dual-track (speakers + mic)."""
        # Coleta todos os .wav
        raw_files: list[tuple[Path, str]] = []
        for search_dir, origin in [(RECORDINGS_DIR, "local"), (VEXA_RECORDINGS_DIR, "vexa")]:
            if not search_dir.exists():
                continue
            for wav in search_dir.glob("*.wav"):
                raw_files.append((wav, origin))

        # Agrupa por prefixo de timestamp
        groups: dict[str, dict] = {}
        for wav, origin in raw_files:
            stem = wav.stem  # ex: 20260318_153045_speakers
            key = stem
            track_type = "solo"

            if stem.endswith(SPEAKERS_SUFFIX):
                key = stem[: -len(SPEAKERS_SUFFIX)]
                track_type = "speakers"
            elif stem.endswith(MIC_SUFFIX):
                key = stem[: -len(MIC_SUFFIX)]
                track_type = "mic"

            group_key = f"{wav.parent}|{key}"
            if group_key not in groups:
                groups[group_key] = {
                    "key": key,
                    "dir": wav.parent,
                    "origin": origin,
                    "speakers_path": None,
                    "mic_path": None,
                    "solo_path": None,
                    "total_size": 0,
                    "mtime": 0.0,
                }

            try:
                size = wav.stat().st_size
                mtime = wav.stat().st_mtime
            except OSError:
                continue

            g = groups[group_key]
            g["total_size"] += size
            g["mtime"] = max(g["mtime"], mtime)

            if track_type == "speakers":
                g["speakers_path"] = wav
            elif track_type == "mic":
                g["mic_path"] = wav
            else:
                g["solo_path"] = wav

        # Converte para lista
        result = []
        for g in groups.values():
            speakers = g["speakers_path"]
            mic = g["mic_path"]
            solo = g["solo_path"]

            if speakers:
                # Par dual-track (ou so speakers sem mic)
                tracks = "speakers + mic" if mic else "speakers"
                label_name = g["key"]
            elif solo:
                # Arquivo solo (sem sufixo _speakers/_mic)
                speakers = solo
                mic = None
                tracks = "solo"
                label_name = solo.stem
            else:
                # So mic sem speakers — improvavel mas tratamos
                speakers = mic
                mic = None
                tracks = "mic"
                label_name = g["key"]

            result.append({
                "speakers_path": speakers,
                "mic_path": mic,
                "origin": g["origin"],
                "size_mb": g["total_size"] / 1e6,
                "mtime": g["mtime"],
                "date_str": datetime.fromtimestamp(g["mtime"]).strftime("%d/%m/%Y %H:%M"),
                "tracks": tracks,
                "label_name": label_name,
            })

        result.sort(key=lambda f: f["mtime"], reverse=True)
        return result

    def _refresh(self):
        """Atualiza a lista de gravacoes."""
        for widget in self._list_frame.winfo_children():
            widget.destroy()

        self._recordings = self._scan_recordings()
        self._selected_var.set(-1)

        if not self._recordings:
            ctk.CTkLabel(
                self._list_frame,
                text="Nenhuma gravacao encontrada em recordings/",
                text_color="gray",
            ).grid(row=0, column=0, padx=10, pady=20)
            return

        for i, rec in enumerate(self._recordings):
            origin_tag = "[Vexa]" if rec["origin"] == "vexa" else "[Local]"
            tracks_tag = f"[{rec['tracks']}]"
            label_text = (
                f"{rec['date_str']}  |  {rec['size_mb']:.1f} MB  |  "
                f"{origin_tag} {tracks_tag}  |  {rec['label_name']}"
            )

            rb = ctk.CTkRadioButton(
                self._list_frame,
                text=label_text,
                variable=self._selected_var,
                value=i,
                font=ctk.CTkFont(size=12),
            )
            rb.grid(row=i, column=0, padx=5, pady=2, sticky="w")

        # Transcribe button at end
        ctk.CTkButton(
            self._list_frame,
            text="Transcrever selecionado",
            width=200,
            command=self._transcribe_selected,
        ).grid(row=len(self._recordings), column=0, padx=5, pady=(15, 5), sticky="w")

    def _transcribe_selected(self):
        idx = self._selected_var.get()
        if idx < 0 or idx >= len(self._recordings):
            return

        rec = self._recordings[idx]
        speakers_path = rec["speakers_path"]
        mic_path = rec["mic_path"]
        log.info(
            "Transcricao manual solicitada: speakers=%s, mic=%s",
            speakers_path, mic_path,
        )

        try:
            self._on_transcribe(speakers_path, mic_path)
        except Exception as e:
            log.error("Erro ao enfileirar transcricao: %s", e)

        self._close()

    def _close(self):
        if self._window:
            self._window.destroy()
            self._window = None

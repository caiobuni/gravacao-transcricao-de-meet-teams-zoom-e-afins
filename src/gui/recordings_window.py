"""Janela para listar e selecionar gravacoes anteriores para transcricao local."""

import logging
import os
from datetime import datetime
from pathlib import Path

import customtkinter as ctk

from src.config.constants import RECORDINGS_DIR, VEXA_RECORDINGS_DIR

log = logging.getLogger(__name__)


class RecordingsWindow:
    def __init__(self, on_transcribe: callable):
        """
        Args:
            on_transcribe: callback(audio_path: Path) chamado ao clicar Transcrever.
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
        self._window.geometry("600x450")
        self._window.resizable(False, True)

        self._window.grid_columnconfigure(0, weight=1)
        self._window.grid_rowconfigure(1, weight=1)

        # Header
        ctk.CTkLabel(
            self._window,
            text="Selecione um audio para transcrever com whisper + pyannote:",
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
        """Escaneia diretorios de gravacoes e retorna lista de arquivos."""
        files = []
        for search_dir, origin in [(RECORDINGS_DIR, "local"), (VEXA_RECORDINGS_DIR, "vexa")]:
            if not search_dir.exists():
                continue
            for wav in search_dir.glob("*.wav"):
                try:
                    size = wav.stat().st_size
                    mtime = wav.stat().st_mtime
                except OSError:
                    continue
                files.append({
                    "path": wav,
                    "name": wav.name,
                    "origin": origin,
                    "size_mb": size / 1e6,
                    "mtime": mtime,
                    "date_str": datetime.fromtimestamp(mtime).strftime("%d/%m/%Y %H:%M"),
                })

        files.sort(key=lambda f: f["mtime"], reverse=True)
        return files

    def _refresh(self):
        """Atualiza a lista de gravacoes."""
        # Clear existing widgets
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
            label_text = f"{rec['date_str']}  |  {rec['size_mb']:.1f} MB  |  {origin_tag}  |  {rec['name']}"

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
        audio_path = rec["path"]
        log.info("Transcricao manual solicitada: %s", audio_path)

        try:
            self._on_transcribe(audio_path)
        except Exception as e:
            log.error("Erro ao enfileirar transcricao: %s", e)

        self._close()

    def _close(self):
        if self._window:
            self._window.destroy()
            self._window = None

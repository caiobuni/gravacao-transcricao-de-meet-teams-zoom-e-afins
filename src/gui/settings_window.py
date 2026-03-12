"""Janela de configuracoes usando CustomTkinter."""

import logging
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from src.config.settings import Settings

log = logging.getLogger(__name__)


class SettingsWindow:
    def __init__(self, settings: Settings, on_save: callable | None = None):
        self._settings = settings
        self._on_save = on_save
        self._window: ctk.CTkToplevel | None = None

    def show(self):
        if self._window and self._window.winfo_exists():
            self._window.focus()
            return

        self._window = ctk.CTkToplevel()
        self._window.title("Configuracoes — Gravacao e Transcricao")
        self._window.geometry("500x550")
        self._window.resizable(False, False)

        # Configure grid weights for the main window
        self._window.grid_columnconfigure(1, weight=1)

        row = 0

        # --- User Name ---
        ctk.CTkLabel(self._window, text="Nome do usuario:").grid(
            row=row, column=0, padx=10, pady=(15, 5), sticky="w"
        )
        self._user_name_var = ctk.StringVar(value=self._settings.user_name)
        ctk.CTkEntry(self._window, textvariable=self._user_name_var, width=300).grid(
            row=row, column=1, padx=10, pady=(15, 5), sticky="ew"
        )
        row += 1

        # --- Transcription Engine ---
        ctk.CTkLabel(self._window, text="Motor de transcricao:").grid(
            row=row, column=0, padx=10, pady=5, sticky="w"
        )
        engine_options = ["Qwen3-ASR", "faster-whisper", "Groq API"]
        engine_map = {"qwen": "Qwen3-ASR", "whisper": "faster-whisper", "groq": "Groq API"}
        current_engine = engine_map.get(self._settings.transcription_engine, "Qwen3-ASR")
        self._engine_var = ctk.StringVar(value=current_engine)
        ctk.CTkOptionMenu(
            self._window,
            variable=self._engine_var,
            values=engine_options,
            width=300,
        ).grid(row=row, column=1, padx=10, pady=5, sticky="ew")
        row += 1

        # --- Whisper Model ---
        ctk.CTkLabel(self._window, text="Modelo Whisper:").grid(
            row=row, column=0, padx=10, pady=5, sticky="w"
        )
        whisper_options = ["large-v3", "medium", "small"]
        self._whisper_model_var = ctk.StringVar(value=self._settings.whisper_model)
        ctk.CTkOptionMenu(
            self._window,
            variable=self._whisper_model_var,
            values=whisper_options,
            width=300,
        ).grid(row=row, column=1, padx=10, pady=5, sticky="ew")
        row += 1

        # --- Language ---
        ctk.CTkLabel(self._window, text="Idioma:").grid(
            row=row, column=0, padx=10, pady=5, sticky="w"
        )
        language_options = ["pt", "en"]
        self._language_var = ctk.StringVar(value=self._settings.language)
        ctk.CTkOptionMenu(
            self._window,
            variable=self._language_var,
            values=language_options,
            width=300,
        ).grid(row=row, column=1, padx=10, pady=5, sticky="ew")
        row += 1

        # --- Output Folder ---
        ctk.CTkLabel(self._window, text="Pasta de saida:").grid(
            row=row, column=0, padx=10, pady=5, sticky="w"
        )
        output_frame = ctk.CTkFrame(self._window, fg_color="transparent")
        output_frame.grid(row=row, column=1, padx=10, pady=5, sticky="ew")
        output_frame.grid_columnconfigure(0, weight=1)

        self._output_dir_var = ctk.StringVar(value=self._settings.transcription_output_dir)
        ctk.CTkEntry(output_frame, textvariable=self._output_dir_var).grid(
            row=0, column=0, sticky="ew", padx=(0, 5)
        )
        ctk.CTkButton(
            output_frame, text="...", width=40, command=self._browse_output_dir
        ).grid(row=0, column=1)
        row += 1

        # --- Auto-delete audio ---
        self._auto_delete_var = ctk.BooleanVar(value=self._settings.auto_delete_audio)
        ctk.CTkCheckBox(
            self._window,
            text="Deletar audio apos transcricao",
            variable=self._auto_delete_var,
        ).grid(row=row, column=0, columnspan=2, padx=10, pady=10, sticky="w")
        row += 1

        # --- Auto-detect Meet ---
        self._auto_detect_var = ctk.BooleanVar(value=self._settings.auto_detect_meet)
        ctk.CTkCheckBox(
            self._window,
            text="Detectar Google Meet automaticamente",
            variable=self._auto_detect_var,
        ).grid(row=row, column=0, columnspan=2, padx=10, pady=(0, 15), sticky="w")
        row += 1

        # --- Buttons ---
        btn_frame = ctk.CTkFrame(self._window, fg_color="transparent")
        btn_frame.grid(row=row, column=0, columnspan=2, padx=10, pady=(10, 15), sticky="e")

        ctk.CTkButton(
            btn_frame, text="Cancelar", width=100, fg_color="gray", command=self._cancel
        ).grid(row=0, column=0, padx=(0, 10))
        ctk.CTkButton(
            btn_frame, text="Salvar", width=100, command=self._save
        ).grid(row=0, column=1)

    def _browse_output_dir(self):
        folder = filedialog.askdirectory(
            title="Selecionar pasta de saida",
            initialdir=self._output_dir_var.get(),
        )
        if folder:
            self._output_dir_var.set(folder)

    def _save(self):
        # Map display name back to engine key
        engine_reverse = {
            "Qwen3-ASR": "qwen",
            "faster-whisper": "whisper",
            "Groq API": "groq",
        }

        self._settings.user_name = self._user_name_var.get().strip() or "Eu"
        self._settings.transcription_engine = engine_reverse.get(
            self._engine_var.get(), "qwen"
        )
        self._settings.whisper_model = self._whisper_model_var.get()
        self._settings.language = self._language_var.get()
        self._settings.transcription_output_dir = self._output_dir_var.get()
        self._settings.auto_delete_audio = self._auto_delete_var.get()
        self._settings.auto_detect_meet = self._auto_detect_var.get()

        self._settings.save()
        log.info("Configuracoes salvas")

        if self._on_save:
            self._on_save(self._settings)

        if self._window:
            self._window.destroy()
            self._window = None

    def _cancel(self):
        if self._window:
            self._window.destroy()
            self._window = None

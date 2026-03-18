"""Raiz CTk persistente e oculta para janelas do app.

Resolve o problema de janelas CustomTkinter que nao reabrem apos fechar
quando criadas em threads a partir do pystray. A raiz Tk roda em thread
dedicada e todas as janelas sao criadas via run_on_tk() para thread-safety.
"""

import threading

import customtkinter as ctk

_root: ctk.CTk | None = None
_ready = threading.Event()


def start():
    """Inicia a raiz CTk oculta em thread dedicada. Chamar 1x no main."""
    def _run():
        global _root
        _root = ctk.CTk()
        _root.withdraw()
        _root.protocol("WM_DELETE_WINDOW", lambda: None)
        _ready.set()
        _root.mainloop()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    _ready.wait()


def run_on_tk(callback):
    """Agenda callback no mainloop do Tk (thread-safe)."""
    if _root:
        _root.after(0, callback)


def get_root():
    return _root

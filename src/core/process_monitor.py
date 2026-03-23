import ctypes
import ctypes.wintypes as wintypes
import logging
import threading
import time
from typing import Callable

import psutil

from src.config.constants import MEET_PROCESS_NAMES, MEET_URL_PATTERN

log = logging.getLogger(__name__)

# Win32 callback type for EnumWindows
_WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

# Nomes de processo do navegador (lowercase)
_BROWSER_NAMES = {n.lower() for n in MEET_PROCESS_NAMES}

# Padroes de titulo que indicam Google Meet ativo
_MEET_TITLE_PATTERNS = ["meet.google.com", "google meet"]


class MeetProcessMonitor:
    """Monitors if Google Meet is running via process cmdline and window titles."""

    def __init__(self, check_interval: float = 5.0):
        self._check_interval = check_interval
        self._running = False
        self._thread = None
        self._meet_detected = False
        self.on_meet_opened: Callable | None = None
        self.on_meet_closed: Callable | None = None
        # Cache de PIDs do navegador para filtrar janelas
        self._browser_pids: set[int] = set()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        log.info("Process monitor iniciado (cmdline + titulo de janela)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    def _monitor_loop(self):
        while self._running:
            is_meet_running = self._check_meet_running()

            if is_meet_running and not self._meet_detected:
                self._meet_detected = True
                log.info("Google Meet detectado")
                if self.on_meet_opened:
                    try:
                        self.on_meet_opened()
                    except Exception:
                        log.error("Erro no callback on_meet_opened", exc_info=True)
            elif not is_meet_running and self._meet_detected:
                self._meet_detected = False
                log.info("Google Meet encerrado")
                if self.on_meet_closed:
                    try:
                        self.on_meet_closed()
                    except Exception:
                        log.error("Erro no callback on_meet_closed", exc_info=True)

            time.sleep(self._check_interval)

    def _check_meet_running(self) -> bool:
        """Check Meet via process cmdline OR window title."""
        # 1. Verificar cmdline (funciona para PWAs)
        self._browser_pids.clear()
        try:
            for proc in psutil.process_iter(['name', 'pid', 'cmdline']):
                try:
                    name = proc.info.get('name', '').lower()
                    if name not in _BROWSER_NAMES:
                        continue
                    self._browser_pids.add(proc.info['pid'])
                    cmdline = proc.info.get('cmdline') or []
                    cmdline_str = ' '.join(cmdline).lower()
                    if MEET_URL_PATTERN in cmdline_str or '--app=https://meet.google.com' in cmdline_str:
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            log.debug("Erro ao verificar processos: %s", e)

        # 2. Verificar titulo de janela (funciona para abas normais)
        if self._browser_pids:
            return self._check_meet_window_title()

        return False

    def _check_meet_window_title(self) -> bool:
        """Check if any browser window title contains Meet patterns."""
        found = False

        def enum_callback(hwnd, _lparam):
            nonlocal found
            if found:
                return True  # ja encontrou, pode parar

            # Verifica se a janela eh visivel
            if not ctypes.windll.user32.IsWindowVisible(hwnd):
                return True

            # Verifica se pertence a um processo de navegador
            pid = wintypes.DWORD()
            ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value not in self._browser_pids:
                return True

            # Le o titulo da janela
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True

            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value.lower()

            for pattern in _MEET_TITLE_PATTERNS:
                if pattern in title:
                    found = True
                    return True

            return True

        try:
            ctypes.windll.user32.EnumWindows(_WNDENUMPROC(enum_callback), 0)
        except Exception as e:
            log.debug("Erro ao enumerar janelas: %s", e)

        return found

    @property
    def is_meet_detected(self) -> bool:
        return self._meet_detected

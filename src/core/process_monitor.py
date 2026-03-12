import logging
import threading
import time
from typing import Callable

import psutil

from src.config.constants import MEET_PROCESS_NAMES, MEET_URL_PATTERN

log = logging.getLogger(__name__)

class MeetProcessMonitor:
    """Monitors if the Google Meet Chrome Web App is running."""

    def __init__(self, check_interval: float = 5.0):
        self._check_interval = check_interval
        self._running = False
        self._thread = None
        self._meet_detected = False
        self.on_meet_opened: Callable | None = None
        self.on_meet_closed: Callable | None = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

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
                    self.on_meet_opened()
            elif not is_meet_running and self._meet_detected:
                self._meet_detected = False
                log.info("Google Meet encerrado")
                if self.on_meet_closed:
                    self.on_meet_closed()

            time.sleep(self._check_interval)

    def _check_meet_running(self) -> bool:
        """Check if any Chrome/Edge process has Meet in its command line."""
        try:
            for proc in psutil.process_iter(['name', 'cmdline']):
                try:
                    name = proc.info.get('name', '').lower()
                    if name not in [n.lower() for n in MEET_PROCESS_NAMES]:
                        continue
                    cmdline = proc.info.get('cmdline') or []
                    cmdline_str = ' '.join(cmdline).lower()
                    if MEET_URL_PATTERN in cmdline_str or '--app=https://meet.google.com' in cmdline_str:
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            log.debug(f"Erro ao verificar processos: {e}")
        return False

    @property
    def is_meet_detected(self) -> bool:
        return self._meet_detected

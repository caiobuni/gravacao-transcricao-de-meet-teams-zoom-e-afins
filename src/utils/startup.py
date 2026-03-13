"""Gerencia auto-start do app com o Windows via registro."""

import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_REG_KEY_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"
_REG_VALUE_NAME = "GravacaoTranscricao"


def _get_launch_command() -> str:
    """Monta o comando para iniciar o app sem janela de console."""
    app_root = Path(__file__).resolve().parent.parent.parent
    # Prioriza .venv (ambiente principal com todas as dependencias)
    venv_pythonw = app_root / ".venv" / "Scripts" / "pythonw.exe"
    if venv_pythonw.exists():
        pythonw = venv_pythonw
    else:
        exe = Path(sys.executable)
        pythonw = exe.parent / "pythonw.exe"
        if not pythonw.exists():
            pythonw = exe

    main_py = app_root / "src" / "main.py"
    return f'"{pythonw}" "{main_py}"'


def is_startup_enabled() -> bool:
    """Verifica se o app esta registrado para iniciar com o Windows."""
    import winreg

    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, _REG_KEY_PATH, 0, winreg.KEY_READ)
        try:
            winreg.QueryValueEx(key, _REG_VALUE_NAME)
            return True
        except FileNotFoundError:
            return False
        finally:
            winreg.CloseKey(key)
    except OSError:
        return False


def enable_startup() -> None:
    """Registra o app para iniciar com o Windows."""
    import winreg

    cmd = _get_launch_command()
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, _REG_KEY_PATH, 0, winreg.KEY_SET_VALUE)
        try:
            winreg.SetValueEx(key, _REG_VALUE_NAME, 0, winreg.REG_SZ, cmd)
            log.info("Auto-start habilitado: %s", cmd)
        finally:
            winreg.CloseKey(key)
    except OSError as e:
        log.error("Erro ao habilitar auto-start: %s", e)


def disable_startup() -> None:
    """Remove o registro de auto-start."""
    import winreg

    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, _REG_KEY_PATH, 0, winreg.KEY_SET_VALUE)
        try:
            winreg.DeleteValue(key, _REG_VALUE_NAME)
            log.info("Auto-start desabilitado")
        except FileNotFoundError:
            pass  # ja nao existia
        finally:
            winreg.CloseKey(key)
    except OSError as e:
        log.error("Erro ao desabilitar auto-start: %s", e)


def set_startup(enabled: bool) -> None:
    """Habilita ou desabilita o auto-start."""
    if enabled:
        enable_startup()
    else:
        disable_startup()

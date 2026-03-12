"""Detecção de disponibilidade de GPU/CUDA."""

import logging

logger = logging.getLogger(__name__)


def check_gpu() -> dict:
    """Returns info about GPU availability.

    Returns:
        dict with keys: available (bool), device_name (str | None), vram_gb (float | None)
    """
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = round(vram_bytes / (1024**3), 2)
            logger.info("GPU detectada: %s (%.2f GB VRAM)", device_name, vram_gb)
            return {"available": True, "device_name": device_name, "vram_gb": vram_gb}
    except ImportError:
        logger.debug("torch não instalado, GPU indisponível")
    except Exception:
        logger.warning("Erro ao verificar GPU", exc_info=True)

    return {"available": False, "device_name": None, "vram_gb": None}


def get_optimal_device() -> str:
    """Returns 'cuda:0' if GPU available, otherwise 'cpu'."""
    gpu_info = check_gpu()
    device = "cuda:0" if gpu_info["available"] else "cpu"
    logger.info("Dispositivo selecionado: %s", device)
    return device


def get_optimal_dtype():
    """Returns torch.bfloat16 for GPU, torch.float32 for CPU."""
    import torch

    if check_gpu()["available"]:
        return torch.bfloat16
    return torch.float32

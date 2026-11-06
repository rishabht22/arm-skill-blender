"""Local caches and explicit device selection shared by Mac and Linux."""
import os
from pathlib import Path


def configure_caches():
    root = Path(__file__).resolve().parents[1] / 'data/runtime-cache'
    for variable, folder in (('MPLCONFIGDIR', 'matplotlib'), ('NUMBA_CACHE_DIR', 'numba')):
        if variable not in os.environ:
            target = root / folder
            target.mkdir(parents=True, exist_ok=True)
            os.environ[variable] = str(target)


def resolve_device(requested='auto'):
    import torch
    requested = str(requested)
    if requested == 'auto':
        allocated = bool(os.environ.get('CUDA_VISIBLE_DEVICES', '').strip())
        requested = 'cuda:0' if allocated and torch.cuda.is_available() else 'cpu'
    if requested.isdigit():
        requested = f'cuda:{requested}'
    device = torch.device(requested)
    if device.type == 'cuda':
        if not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count():
            raise ValueError(f'Requested device {device} is not available')
    elif device.type == 'mps':
        if not torch.backends.mps.is_available():
            raise ValueError('Requested MPS device is not available')
    elif device.type != 'cpu':
        raise ValueError(f'Unsupported device: {device}')
    return str(device)

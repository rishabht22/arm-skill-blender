"""Machine-dependent choices shared by training and evaluation."""

import os
import sys
from pathlib import Path


def configure_caches():
    root = Path(__file__).resolve().parents[1] / "data/runtime-cache"
    for name, folder in (("HF_HOME", "huggingface"), ("TRANSFORMERS_CACHE", "transformers"),
                         ("MPLCONFIGDIR", "matplotlib"), ("NUMBA_CACHE_DIR", "numba")):
        if name not in os.environ:
            directory = root / folder
            directory.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(directory)


def resolve_device(requested="auto"):
    import torch
    requested = str(requested)
    if requested == "auto":
        # CPU is the portable Mac default. MPS support varies across policy ops.
        # Shared lab servers require explicitly selecting a GPU for this session.
        selected = bool(os.environ.get("CUDA_VISIBLE_DEVICES", "").strip())
        requested = "cuda:0" if selected and torch.cuda.is_available() else "cpu"
    elif requested.isdigit():
        requested = f"cuda:{requested}"  # Accept the original CALVIN --device 0.
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count():
            raise ValueError(f"Requested device {device} is not available")
    elif device.type == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise ValueError("Requested MPS device is not available")
    elif device.type != "cpu":
        raise ValueError(f"Unsupported device: {device}")
    return str(device)


def has_display():
    return sys.platform == "darwin" or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def resolve_renderer(requested="auto", *, device="cpu"):
    if requested == "auto":
        return "egl" if sys.platform == "linux" and str(device).startswith("cuda") else "tiny"
    if requested not in ("egl", "tiny"):
        raise ValueError("renderer must be auto, egl, or tiny")
    if requested == "egl" and sys.platform != "linux":
        raise ValueError("EGL rendering is supported here on Linux; use tiny on Mac")
    return requested

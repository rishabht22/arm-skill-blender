from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from src.runtime import resolve_device, resolve_renderer, has_display
from src.envs.calvin_reset import initial_observations
from experiments import run as launcher


def test_cpu_fallback_and_explicit_cuda_failure(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device() == "cpu"
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError, match="not available"):
        resolve_device("cuda")


def test_visible_cuda_device_selection(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert resolve_device() == "cuda:0"
    assert resolve_device("0") == "cuda:0"
    with pytest.raises(ValueError, match="not available"):
        resolve_device("cuda:1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert resolve_device() == "cpu"


def test_mac_software_and_linux_gpu_rendering(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert resolve_renderer(device="cpu") == "tiny"
    with pytest.raises(ValueError):
        resolve_renderer("egl")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert not has_display()
    assert resolve_renderer(device="cuda:0") == "egl"
    assert resolve_renderer(device="cpu") == "tiny"


def test_launcher_uses_same_runtime_and_commands_on_both_machines(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    for backend in ("calvin",):
        directory = tmp_path / f".venv-{backend}/bin"
        directory.mkdir(parents=True)
        (directory / "python").touch()
        (directory / "mjpython").touch()
    command = launcher.command_for("train", [], platform="darwin")
    assert Path(command[0]) == tmp_path / ".venv-calvin/bin/python"
    command = launcher.command_for("evaluate", [], platform="darwin")
    assert Path(command[0]).name == "python"
    command = launcher.command_for("evaluate", [], platform="linux")
    assert Path(command[0]).name == "python"
    command = launcher.command_for("train", ["--timesteps", "8"], platform="linux")
    assert command[-3:] == ["train", "--timesteps", "8"]


def test_calvin_initial_state_is_reproducible_without_optional_pyhash():
    state = dict(slider="left", drawer="closed", lightbulb=0, led=0,
                 red_block="table", blue_block="slider_left", pink_block="table")
    robot, scene = initial_observations(state, seed=7)
    robot2, scene2 = initial_observations(state, seed=7)
    assert robot.shape == (15,) and scene.shape == (24,)
    np.testing.assert_array_equal(robot, robot2)
    np.testing.assert_array_equal(scene, scene2)
    assert scene[1] == 0 and scene[0] == .28
    state["drawer"] = "open"
    assert initial_observations(state)[1][1] == .22

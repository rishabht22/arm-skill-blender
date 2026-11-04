from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3.common.env_checker import check_env

from src.coordination.tasks import Task, load_tasks
from src.envs.composition import CompositionEnv
from src.policies.frozen_skills import parameter_digest
from src.training.composition import Settings, Experiment, evaluate, train

ROOT = Path(__file__).resolve().parents[1]


class FakeBackend:
    """No pretrained model or simulator required for head-pipeline tests."""
    def __init__(self, fail=None, lose_final=False):
        self.model = torch.nn.Linear(1, 1).eval()
        self.model.requires_grad_(False)
        self.original_digest = parameter_digest(self.model)
        self.fail, self.lose_final = fail, lose_final
        self.executed, self.reset_calls, self.closed = [], 0, False

    def reset(self, state, seed):
        self.reset_calls += 1
        self.executed = []
        return self.state()

    def state(self):
        return np.zeros(39, dtype=np.float32)

    def execute(self, skill, max_steps):
        start = {"before": len(self.executed)}
        self.executed.append(skill)
        return skill != self.fail, min(2, max_steps), start

    def achieved(self, skill, start):
        return not self.lose_final and skill in self.executed

    def assert_frozen(self):
        assert parameter_digest(self.model) == self.original_digest
        assert not any(p.requires_grad for p in self.model.parameters())

    def close(self):
        self.closed = True


def tiny_env(**kwargs):
    task = Task("cycle", "train", ("open", "close"), {}, ("close",))
    return CompositionEnv(FakeBackend(**kwargs), ("open", "close", "other"), [task], option_steps=3, decision_limit=3)


def test_native_gym_contract_and_continuous_scene():
    env = tiny_env()
    check_env(env, warn=True)
    obs, _ = env.reset(seed=3)
    resets = env.backend.reset_calls
    assert env.observation_space.contains(obs)
    _, reward, terminated, truncated, info = env.step(0)
    assert reward > 0 and not terminated and not truncated
    _, reward, terminated, truncated, info = env.step(1)
    assert terminated and not truncated and info["is_success"]
    assert env.backend.reset_calls == resets
    assert env.backend.executed == ["open", "close"]
    with pytest.raises(RuntimeError):
        env.step(0)


def test_invalid_selection_does_not_execute_and_time_limit_is_truncated():
    env = tiny_env()
    env.reset(seed=0)
    for action in (1, 2, 1):
        _, reward, terminated, truncated, info = env.step(action)
        assert reward < 0 and not terminated and not info["selection_valid"]
    assert truncated and info["reason"] == "decision_limit"
    assert not env.backend.executed


def test_low_level_failure_and_lost_final_condition_are_not_success():
    env = tiny_env(fail="open")
    env.reset(seed=0)
    _, _, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and info["reason"] == "frozen_skill_failed"
    env = tiny_env(lose_final=True)
    env.reset(seed=0)
    env.step(0)
    _, _, terminated, _, info = env.step(1)
    assert terminated and not info["is_success"] and info["reason"] == "final_condition_lost"


def test_task_splits_and_oracle_evaluation_restore_rng():
    skills, tasks = load_tasks(ROOT / "experiments/configs/composition_tasks.json")
    assert len(skills) == 12
    assert not ({t.sequence for t in tasks if t.split == "train"} & {t.sequence for t in tasks if t.split == "test"})
    env = tiny_env()
    torch.manual_seed(77)
    expected = torch.random.get_rng_state().clone()
    result = evaluate(env, seeds=[10, 11])
    assert result["success_rate"] == 1
    assert torch.equal(expected, torch.random.get_rng_state())


def test_dummy_training_saves_loadable_head_and_does_not_update_base(tmp_path, monkeypatch):
    # Eight decisions in a fake environment only; no real policy training.
    settings = Settings(total_timesteps=8, n_steps=4, batch_size=2, n_epochs=1,
                        hidden_sizes=(8,), warmstart_updates=1, warmstart_batch_size=2,
                        checkpoint_every=4, evaluate_every=4, preflight_seeds=(1,), validation_seeds=(2,))
    experiment = Experiment(settings, ROOT / "experiments/configs/composition_tasks.json", tmp_path)
    backends = []

    def factory(split="train"):
        backend = FakeBackend()
        backends.append(backend)
        return CompositionEnv(backend, experiment.skills, [t for t in experiment.tasks if t.split == split],
                              option_steps=3, decision_limit=12)

    monkeypatch.setattr(experiment, "make_env", factory)
    monkeypatch.setattr(experiment, "contract", lambda env: {"test": True})
    class FakeValidation:
        def __init__(self, experiment, run_dir):
            self.env = factory()
        def evaluate(self, model, tag):
            return evaluate(self.env, model=model, seeds=(2,))
        def close(self):
            self.env.close()
    monkeypatch.setattr("src.training.validation.SubprocessValidation", FakeValidation)
    run_dir = train(experiment, tmp_path / "runs")
    assert json.loads((run_dir / "status.json").read_text())["status"] == "complete"
    assert all((run_dir / name).is_file() for name in ("head.zip", "best_head.zip", "contract.json", "base_preflight.json"))
    from stable_baselines3 import PPO
    model = PPO.load(run_dir / "head.zip", device="cpu")
    env = factory("test")
    obs, _ = env.reset(seed=9)
    action = model.predict(obs, deterministic=True)[0]
    assert env.action_space.contains(action)
    resumed = train(experiment, tmp_path / "runs", resume_dir=run_dir)
    assert json.loads((resumed / "status.json").read_text())["actual_timesteps"] == 16
    assert (resumed / "resume.json").is_file()
    for backend in backends:
        backend.assert_frozen()


def test_training_stops_before_optimization_when_base_fails(tmp_path, monkeypatch):
    settings = Settings(total_timesteps=4, qualify_training_tasks=False)
    experiment = Experiment(settings, ROOT / "experiments/configs/composition_tasks.json", tmp_path)
    monkeypatch.setattr(experiment, "make_env", lambda split: CompositionEnv(
        FakeBackend(fail="open_drawer"), experiment.skills, [t for t in experiment.tasks if t.split == split]))
    monkeypatch.setattr(experiment, "contract", lambda env: {"test": True})
    with pytest.raises(RuntimeError, match="baseline threshold"):
        train(experiment, tmp_path / "runs")
    directory = next((tmp_path / "runs").iterdir())
    assert json.loads((directory / "status.json").read_text())["status"] == "failed"
    assert not (directory / "head.zip").exists()


def test_frozen_digest_detects_mutations():
    backend = FakeBackend()
    with torch.no_grad():
        next(backend.model.parameters()).add_(1)
    with pytest.raises(AssertionError):
        backend.assert_frozen()


def test_saved_environment_contract_is_path_independent_and_detects_drift(tmp_path):
    from types import SimpleNamespace
    from gymnasium import spaces
    import shutil
    settings = Settings()
    assets = tmp_path / "mac_assets"
    files = ["models/HULC_D_D/saved_models/HULC_D_D.ckpt", "models/HULC_D_D/.hydra/config.yaml",
             "datasets/task_D_D/validation/.hydra/merged_config.yaml", "datasets/task_D_D/validation/statistics.yaml",
             "datasets/task_D_D/validation/lang_paraphrase-MiniLM-L3-v2/embeddings.npy",
             "vendor/calvin/calvin_env/calvin_env/env.py", "vendor/calvin/calvin_models/calvin_agent/transforms.py",
             "vendor/calvin/calvin_env/data/robot.urdf", "vendor/hulc/hulc/model.py"]
    for name in files:
        path = assets / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test fixture")
    server_assets = tmp_path / "server_assets"
    shutil.copytree(assets, server_assets)
    tasks_path = ROOT / "experiments/configs/composition_tasks.json"
    mac = Experiment(settings, tasks_path, assets)
    server = Experiment(settings, tasks_path, server_assets)
    def fixture(root):
        return SimpleNamespace(backend=SimpleNamespace(checkpoint=root / files[0], original_digest="same weights"),
                               observation_space=spaces.Box(-10., 10., shape=(209,), dtype=np.float32))
    # All versioned data is identical despite completely different paths.
    expected = mac.contract(fixture(assets))
    assert expected == server.contract(fixture(server_assets))
    (server_assets / "datasets/task_D_D/validation/statistics.yaml").write_text("modified")
    assert expected != server.contract(fixture(server_assets))

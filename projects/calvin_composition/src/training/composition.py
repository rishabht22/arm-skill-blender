"""Train only the skill selector; keep the released visuomotor policy frozen."""

from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from src.coordination.tasks import load_tasks
from src.envs.calvin_reset import initial_observations
from src.envs.composition import CompositionEnv, encode_observation
from src.policies.frozen_skills import FrozenSkills
from src.runtime import resolve_device, resolve_renderer

ROOT = Path(__file__).resolve().parents[2]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hash(root, *, python_only=False):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing pinned source or scene assets: {root}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if python_only and path.suffix != ".py":
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(file_hash(path).encode())
    return digest.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


@dataclass
class Settings:
    seed: int = 0
    total_timesteps: int = 4096  # High-level decisions, not physics steps.
    option_steps: int = 360
    decision_limit: int = 12
    n_steps: int = 128
    batch_size: int = 64
    n_epochs: int = 5
    learning_rate: float = .0003
    gamma: float = .98
    ent_coef: float = .01
    target_kl: float = .02
    hidden_sizes: tuple = (128, 128)
    warmstart_updates: int = 200
    warmstart_batch_size: int = 128
    checkpoint_every: int = 256
    evaluate_every: int = 2048
    preflight_seeds: tuple = (9000,)
    minimum_base_success: float = .8
    qualify_training_tasks: bool = True
    minimum_qualified_tasks: int = 2
    minimum_qualified_compositions: int = 1
    validation_seeds: tuple = (19000,)
    test_seeds: tuple = (29000, 29001, 29002)
    device: str = "auto"
    renderer: str = "tiny"

    def __post_init__(self):
        for name in ("total_timesteps", "option_steps", "decision_limit", "n_steps", "batch_size",
                     "n_epochs", "warmstart_batch_size", "checkpoint_every", "evaluate_every"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_steps < 2 or self.batch_size < 2 or self.n_steps % self.batch_size:
            raise ValueError("batch_size must divide n_steps, and both must be at least 2")
        if not 0 < self.gamma <= 1 or not 0 <= self.minimum_base_success <= 1:
            raise ValueError("Invalid discount or baseline success threshold")
        if self.learning_rate <= 0 or self.target_kl <= 0 or self.ent_coef < 0:
            raise ValueError("Invalid optimizer settings")
        if self.warmstart_updates < 0 or not self.hidden_sizes or not all(isinstance(n, int) and n > 0 for n in self.hidden_sizes):
            raise ValueError("Invalid warm-start or network settings")
        if self.minimum_qualified_tasks < 1 or self.minimum_qualified_compositions < 0:
            raise ValueError("minimum_qualified_tasks must be positive")
        for seeds in (self.preflight_seeds, self.validation_seeds, self.test_seeds):
            if not seeds or any(not isinstance(s, int) or s < 0 for s in seeds):
                raise ValueError("Evaluation seeds must be nonempty nonnegative integers")
            if len(set(seeds)) != len(seeds):
                raise ValueError("Configured evaluation seeds must be unique")


class Experiment:
    def __init__(self, settings, tasks_path, asset_root, catalog_path=None):
        self.settings = settings
        self.tasks_path = Path(tasks_path)
        self.asset_root = Path(asset_root).resolve()
        self.catalog_path = Path(catalog_path or ROOT / "experiments/configs/calvin_skills.json")
        self.task_data = json.loads(self.tasks_path.read_text())
        self.catalog_data = json.loads(self.catalog_path.read_text())
        self.skills, self.tasks = load_tasks(self.task_data)
        catalog_ids = {item["id"] for item in self.catalog_data["skills"]}
        if set(self.skills) - catalog_ids:
            raise ValueError("Task bank uses unknown CALVIN skills")
        if settings.decision_limit < max(len(t.sequence) for t in self.tasks):
            raise ValueError("decision_limit is shorter than a requested composition")
        self.device = resolve_device(settings.device)
        self.renderer = resolve_renderer(settings.renderer, device=self.device)

    def make_env(self, split="train"):
        backend = FrozenSkills(self.asset_root, self.catalog_path, device=self.device, renderer=self.renderer)
        return CompositionEnv(backend, self.skills, [t for t in self.tasks if t.split == split],
                              option_steps=self.settings.option_steps, decision_limit=self.settings.decision_limit)

    def contract(self, env):
        dataset = self.asset_root / "datasets/task_D_D"
        paths = {
            "checkpoint": env.backend.checkpoint,
            "policy_config": self.asset_root / "models/HULC_D_D/.hydra/config.yaml",
            "scene": dataset / "validation/.hydra/merged_config.yaml",
            "statistics": dataset / "validation/statistics.yaml",
            "embeddings": dataset / "validation/lang_paraphrase-MiniLM-L3-v2/embeddings.npy",
            "task_bank": self.tasks_path,
            "catalog": self.catalog_path,
        }
        import importlib.metadata
        hashes = {name: file_hash(path) for name, path in paths.items()}
        for name in ("task_bank", "catalog"):
            data = self.task_data if name == "task_bank" else self.catalog_data
            hashes[name] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        return {
            "files": hashes,
            "source_and_assets": {
                "calvin_environment": source_hash(self.asset_root / "vendor/calvin/calvin_env/calvin_env", python_only=True),
                "calvin_transforms": source_hash(self.asset_root / "vendor/calvin/calvin_models/calvin_agent", python_only=True),
                "robot_and_scene_assets": source_hash(self.asset_root / "vendor/calvin/calvin_env/data"),
                "hulc": source_hash(self.asset_root / "vendor/hulc/hulc", python_only=True),
            },
            "packages": {name: importlib.metadata.version(name).split("+")[0]
                         for name in ("torch", "numpy", "pybullet", "stable-baselines3", "gymnasium")},
            "base_parameters": env.backend.original_digest,
            "action_order": list(self.skills), "observation_shape": list(env.observation_space.shape),
            "option_steps": self.settings.option_steps, "decision_limit": self.settings.decision_limit,
            "renderer": self.renderer, "head_observation": "privileged_state_and_goal_graph_v1",
            "reward": "node_1_terminal_5_failure_minus1_invalid_minus0.25_frame_cost0.0001_v1",
            "discount_unit": "one_skill_selection",
            "episode_reset": "reload_native_scene_v1",
        }


@contextmanager
def preserved_rng():
    state, python_state = np.random.get_state(), random.getstate()
    devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        np.random.set_state(state)
        random.setstate(python_state)


def evaluate(env, *, model=None, seeds=(29000,), random_actions=False):
    rows = []
    with preserved_rng():
        for task in env.tasks:
            for seed in seeds:
                set_random_seed(seed)
                obs, info = env.reset(seed=seed, options={"task": task.name})
                rng = np.random.default_rng(seed)
                history, total_reward = [], 0.
                while True:
                    if random_actions:
                        action = int(rng.integers(env.action_space.n))
                    elif model is None:
                        action = env.teacher_action()
                    else:
                        action = int(model.predict(obs, deterministic=True)[0])
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    history.append(dict(info))
                    if terminated or truncated:
                        break
                rows.append(dict(info, seed=seed, reward=total_reward, history=history))
                print(f"{task.name} seed={seed}: {info['reason']} ({info['completed']}/{info['required']})", flush=True)
    env.backend.assert_frozen()
    rates = {task.name: float(np.mean([row["is_success"] for row in rows if row["task"] == task.name])) for task in env.tasks}
    selections = [step for row in rows for step in row["history"]]
    valid = [step for step in selections if step["selection_valid"]]
    return dict(success_rate=float(np.mean([row["is_success"] for row in rows])), per_task=rates, episodes=rows,
                valid_selection_rate=float(np.mean([step["selection_valid"] for step in selections])),
                frozen_skill_success_rate=float(np.mean([step["reason"] not in ("frozen_skill_failed",) for step in valid])) if valid else 0.)


def warmstart(model, experiment):
    """Supervise graph sequencing on training goals only, without sim rollouts."""
    settings = experiment.settings
    rng = np.random.default_rng(settings.seed)
    tasks = [t for t in experiment.tasks if t.split == "train"]
    for _ in range(settings.warmstart_updates):
        observations, actions = [], []
        for _ in range(settings.warmstart_batch_size):
            task = tasks[int(rng.integers(len(tasks)))]
            count = int(rng.integers(len(task.sequence)))
            robot, scene = initial_observations(task.initial_state, seed=int(rng.integers(2**31)))
            observations.append(encode_observation(
                np.concatenate([robot, scene]), experiment.skills, task, set(task.sequence[:count]),
                count, count * settings.option_steps // 2, settings.decision_limit, settings.option_steps,
            ))
            actions.append(experiment.skills.index(task.sequence[count]))
        obs = torch.as_tensor(np.array(observations), device=model.device)
        labels = torch.as_tensor(actions, dtype=torch.long, device=model.device)
        _, log_prob, _ = model.policy.evaluate_actions(obs, labels)
        loss = -log_prob.mean()
        if not torch.isfinite(loss):
            raise ValueError("Non-finite head warm-start loss")
        model.policy.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.policy.parameters(), .5)
        model.policy.optimizer.step()


class HeadChecks(BaseCallback):
    def __init__(self, experiment, run_dir, eval_env, initial_score=-float("inf")):
        super().__init__()
        self.experiment, self.run_dir, self.eval_env = experiment, Path(run_dir), eval_env
        self.best_score = initial_score

    def _on_step(self):
        settings = self.experiment.settings
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                record = {key: info[key] for key in ("task", "episode_seed", "is_success", "reason", "frames", "decision_trace")}
                record["head_timesteps"] = self.num_timesteps
                with (self.run_dir / "episodes.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
        if self.num_timesteps % settings.checkpoint_every == 0:
            self.model.save(self.run_dir / "checkpoints" / f"head_{self.num_timesteps}")
        if self.num_timesteps % settings.evaluate_every == 0:
            result = self.eval_env.evaluate(self.model, f"step_{self.num_timesteps}")
            save_json(self.run_dir / "validation" / f"step_{self.num_timesteps}.json", result)
            self.logger.record("composition/validation_success", result["success_rate"])
            if result["success_rate"] > self.best_score:
                self.best_score = result["success_rate"]
                self.model.save(self.run_dir / "best_head")
        return True


def train(experiment, output_root, *, allow_weak_base=False, resume_dir=None):
    settings = experiment.settings
    set_random_seed(settings.seed)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(output_root) / stamp
    run_dir.mkdir(parents=True)
    save_json(run_dir / "config.json", asdict(settings))
    save_json(run_dir / "tasks.json", experiment.task_data)
    save_json(run_dir / "catalog.json", experiment.catalog_data)
    save_json(run_dir / "status.json", {"status": "preflight", "allow_weak_base": allow_weak_base})
    env, validation, vector = None, None, None
    try:
        env = experiment.make_env("train")
        contract = experiment.contract(env)
        save_json(run_dir / "contract.json", contract)
        if resume_dir is not None:
            resume_dir = Path(resume_dir)
            if json.loads((resume_dir / "contract.json").read_text()) != contract:
                raise ValueError("Resume checkpoint does not match the frozen skills/environment contract")
        base = evaluate(env, seeds=settings.preflight_seeds)
        save_json(run_dir / "base_preflight.json", base)
        weak = [name for name, rate in base["per_task"].items() if rate < settings.minimum_base_success]
        qualified = [task for task in env.tasks if task.name not in weak]
        if resume_dir is not None:
            old_names = json.loads((resume_dir / "qualified_tasks.json").read_text())["used_for_training"]
            if not allow_weak_base and set(old_names) & set(weak):
                raise RuntimeError("Previously qualified tasks no longer pass the frozen-skill baseline on this runtime")
            qualified = [task for task in env.tasks if task.name in old_names]
        if weak and not allow_weak_base and not settings.qualify_training_tasks:
            raise RuntimeError(f"Frozen skill chains failed the required baseline threshold: {weak}. See {run_dir / 'base_preflight.json'}")
        if allow_weak_base:
            qualified = list(env.tasks) if resume_dir is None else qualified
        save_json(run_dir / "qualified_tasks.json", {
            "used_for_training": [task.name for task in qualified], "failed_qualification": weak,
            "per_task_success": base["per_task"], "seeds": list(settings.preflight_seeds),
            "qualified_compositions": [task.name for task in qualified if len(task.sequence) > 1],
        })
        if len(qualified) < settings.minimum_qualified_tasks and not allow_weak_base:
            raise RuntimeError(f"Only {len(qualified)} chains passed frozen-skill qualification; need {settings.minimum_qualified_tasks}. See {run_dir / 'base_preflight.json'}")
        if sum(len(task.sequence) > 1 for task in qualified) < settings.minimum_qualified_compositions and not allow_weak_base:
            raise RuntimeError("No sufficiently reliable multi-skill training curriculum. Inspect qualified_tasks.json before training a composition head")
        print(f"Qualified training tasks: {[t.name for t in qualified]}; excluded: {weak}", flush=True)
        env.tasks = tuple(qualified)
        from src.training.validation import SubprocessValidation
        validation = SubprocessValidation(experiment, run_dir)
        vector = DummyVecEnv([lambda: Monitor(env, str(run_dir / "episodes.monitor.csv"),
                              info_keywords=("is_success", "frames", "task", "completed", "reason", "episode_seed", "decisions"))])
        if resume_dir is None:
            model = PPO(
                "MlpPolicy", vector, policy_kwargs=dict(net_arch=dict(pi=list(settings.hidden_sizes), vf=list(settings.hidden_sizes))),
                n_steps=settings.n_steps, batch_size=settings.batch_size, n_epochs=settings.n_epochs,
                learning_rate=settings.learning_rate, gamma=settings.gamma, ent_coef=settings.ent_coef,
                target_kl=settings.target_kl, seed=settings.seed, device=experiment.device, verbose=1,
            )
        else:
            checkpoint = resume_dir / "head.zip"
            if not checkpoint.is_file():
                checkpoints = list((resume_dir / "checkpoints").glob("head_*.zip"))
                if not checkpoints:
                    raise FileNotFoundError("No final or periodic head checkpoint found to resume")
                checkpoint = max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))
            model = PPO.load(str(checkpoint), env=vector, device=experiment.device)
            save_json(run_dir / "resume.json", {"checkpoint": str(checkpoint.resolve()), "sha256": file_hash(checkpoint),
                                                "previous_timesteps": model.num_timesteps,
                                                "continuation": "optimizer restored; fresh episodes and RNG, not bit-identical continuation"})
        head_ids = {id(p) for group in model.policy.optimizer.param_groups for p in group["params"]}
        if head_ids & {id(p) for p in env.backend.model.parameters()}:
            raise RuntimeError("Base-policy parameters entered the head optimizer")
        if resume_dir is None:
            # Supervise only the training task graph distribution that qualified.
            original_tasks = experiment.tasks
            try:
                experiment.tasks = list(qualified)
                warmstart(model, experiment)
            finally:
                experiment.tasks = original_tasks
        env.backend.assert_frozen()
        model.save(run_dir / "initial_head")
        if resume_dir is None:
            model.save(run_dir / "warmstart_head")
        initial_validation = validation.evaluate(model, "initial")
        save_json(run_dir / "initial_validation.json", initial_validation)
        model.save(run_dir / "best_head")
        from stable_baselines3.common.logger import configure
        model.set_logger(configure(str(run_dir), ["stdout", "csv"]))
        (run_dir / "checkpoints").mkdir()
        save_json(run_dir / "status.json", {"status": "training", "allow_weak_base": allow_weak_base})
        model.learn(total_timesteps=settings.total_timesteps,
                    callback=HeadChecks(experiment, run_dir, validation, initial_validation["success_rate"]),
                    reset_num_timesteps=resume_dir is None)
        env.backend.assert_frozen()
        model.save(run_dir / "head")
        final = validation.evaluate(model, "final")
        save_json(run_dir / "final_validation.json", final)
        save_json(run_dir / "status.json", {"status": "complete", "base_frozen": True,
                                            "allow_weak_base": allow_weak_base, "actual_timesteps": model.num_timesteps})
        print(f"Saved trained head: {run_dir}", flush=True)
        return run_dir
    except BaseException as error:
        save_json(run_dir / "status.json", {"status": "failed", "error": str(error), "allow_weak_base": allow_weak_base})
        raise
    finally:
        if vector is not None:
            vector.close()
        elif env is not None:
            env.close()
        if validation is not None:
            validation.close()

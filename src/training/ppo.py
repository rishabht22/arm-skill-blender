"""Build, check, and train the state-based PPO baseline."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.policy_env import PolicyEnv
from src.policies.basic import BasicPolicy


@dataclass
class PPOConfig:
    seed: int = 0
    horizon: int = 500
    total_timesteps: int = 100000
    learning_rate: float = 0.0003
    n_steps: int = 1024
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    hidden_sizes: tuple = (128, 128)
    checkpoint_every: int = 10000
    device: str = "cpu"

    def __post_init__(self):
        for name in ("horizon", "total_timesteps", "n_epochs", "checkpoint_every"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_steps < 2 or self.batch_size < 2:
            raise ValueError("n_steps and batch_size must be at least 2")
        if self.n_steps % self.batch_size:
            raise ValueError("batch_size must divide n_steps for this single-environment baseline")
        if self.learning_rate <= 0 or not 0 < self.gamma <= 1:
            raise ValueError("learning_rate must be positive and gamma must be in (0, 1]")
        if not self.hidden_sizes or any(
            not isinstance(size, int) or size < 1 for size in self.hidden_sizes
        ):
            raise ValueError("hidden_sizes must contain positive integers")


def run(config, *, train=False, output_root=Path("outputs/basic_policies")):
    """Default to an environment/inference check; only train when requested."""
    raw_env = PolicyEnv(seed=config.seed, horizon=config.horizon)
    vector_env = None
    logger = None
    try:
        environment_info = {
            "robots": raw_env.robot_names,
            "observation_keys": raw_env.observation_keys,
            "observation_dim": raw_env.observation_space.shape[0],
            "action_dim": raw_env.action_space.shape[0],
            "reward_shaping": True,
        }
        print(json.dumps(environment_info, indent=2), flush=True)

        run_dir = None
        if train:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            run_dir = Path(output_root) / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
            (run_dir / "environment.json").write_text(json.dumps(environment_info, indent=2) + "\n")
        else:
            check_env(raw_env, warn=True)

        monitor_file = str(run_dir / "episodes.monitor.csv") if run_dir else None
        vector_env = VecNormalize(
            DummyVecEnv([lambda: Monitor(raw_env, filename=monitor_file, info_keywords=("is_success",))]),
            norm_obs=True,
            norm_reward=False,
        )
        vector_env.seed(config.seed)
        model = PPO(
            BasicPolicy,
            vector_env,
            policy_kwargs={"hidden_sizes": config.hidden_sizes},
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            seed=config.seed,
            device=config.device,
            verbose=1,
        )

        if not train:
            observation = vector_env.reset()
            action, _ = model.predict(observation, deterministic=True)
            next_observation, reward, _, _ = vector_env.step(action)
            if not (np.isfinite(action).all() and np.isfinite(next_observation).all() and np.isfinite(reward).all()):
                raise ValueError("Preflight produced a non-finite action, observation, or reward")
            print("Preflight passed. No policy optimization or checkpoint writing was performed.")
            return

        logger = configure(str(run_dir), ["stdout", "csv"])
        model.set_logger(logger)
        callback = CheckpointCallback(
            save_freq=config.checkpoint_every,
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo",
            save_vecnormalize=True,
        )
        print(f"Training output: {run_dir}", flush=True)
        model.learn(total_timesteps=config.total_timesteps, callback=callback)
        model.save(str(run_dir / "policy.zip"))
        # Load these statistics along with policy.zip when evaluating the policy.
        vector_env.save(str(run_dir / "vecnormalize.pkl"))
        print(f"Saved policy and observation normalization to {run_dir}", flush=True)
    finally:
        if vector_env is not None:
            vector_env.close()
        else:
            raw_env.close()
        if logger is not None:
            logger.close()

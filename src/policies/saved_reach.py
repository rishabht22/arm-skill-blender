"""Load a trained reaching policy and its frozen observation normalization."""

import json
import pickle
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize


class SavedReachPolicy:
    def __init__(self, run_dir, *, robot_index, device="cpu"):
        self.run_dir = Path(run_dir).resolve()
        for name in ("policy.zip", "vecnormalize.pkl", "config.json", "environment.json"):
            if not (self.run_dir / name).is_file():
                raise FileNotFoundError(f"Missing saved run file: {self.run_dir / name}")
        self.config = json.loads((self.run_dir / "config.json").read_text())
        self.metadata = json.loads((self.run_dir / "environment.json").read_text())
        if self.metadata.get("task") != "reach_handle":
            raise ValueError(f"{self.run_dir} is not an independent reaching policy")
        self.reach_settings = self.metadata["reach"].copy()
        self.reach_settings.setdefault("robot_index", 0)
        self.metadata["reach"] = self.reach_settings
        if self.reach_settings["robot_index"] != robot_index:
            raise ValueError(f"{self.run_dir} was trained for the wrong arm")
        self.robot_index = robot_index
        # VecNormalize saves its statistics without an attached simulation.
        # normalize_obs() is a pure operation and does not update those statistics.
        with (self.run_dir / "vecnormalize.pkl").open("rb") as stream:
            self.normalization = pickle.load(stream)
        if not isinstance(self.normalization, VecNormalize):
            raise ValueError(f"Invalid observation normalization in {self.run_dir}")
        self.normalization.training = False
        self.normalization.norm_reward = False
        self.model = PPO.load(str(self.run_dir / "policy.zip"), device=device)
        self.model.policy.set_training_mode(False)
        expected_shape = (self.metadata["observation_dim"],)
        if self.model.observation_space.shape != expected_shape:
            raise ValueError("Saved model observation dimensions do not match its metadata")
        if self.normalization.observation_space.shape != expected_shape:
            raise ValueError("Saved normalization dimensions do not match the policy")
        if self.model.action_space.shape != (3,) or self.metadata["action_dim"] != 3:
            raise ValueError("An independent reaching policy must output three XYZ commands")

    def validate_environment(self, actual):
        for key, value in actual.items():
            if self.metadata.get(key) != value:
                raise ValueError(
                    f"Arm {self.robot_index + 1} checkpoint expects {key}={self.metadata.get(key)}, "
                    f"but the current scene has {value}"
                )

    def predict(self, observation):
        normalized = self.normalization.normalize_obs(observation)
        action, _ = self.model.predict(normalized, deterministic=True)
        if not np.isfinite(action).all():
            raise ValueError(f"Policy for arm {self.robot_index + 1} produced a non-finite action")
        return action

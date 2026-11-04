"""Train only a small coordinator; fixed arm controllers have no optimizer."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.runtime import resolve_device
from .composition_env import CompositionEnv
from .frozen import ROOT, read_config, verify


def environment(config, **kwargs):
    return CompositionEnv(frames_per_action=config["frames_per_action"],
                          max_decisions=config["max_decisions"], **kwargs)


def evaluate(model, config, *, seed, episodes, env=None):
    owned = env is None
    env = environment(config) if owned else env
    rows = []
    try:
        for value in range(seed, seed + episodes):
            obs, _ = env.reset(seed=value)
            total_reward = 0.
            for _ in range(env.max_decisions):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(int(action))
                total_reward += reward
                if terminated or truncated:
                    break
            row = env.scene.report()
            row.update(seed=value, reward=total_reward, decisions=env.decisions)
            rows.append(row)
    finally:
        if owned:
            env.close()
    return rows


def warmstart(model, env, config, output):
    observations, actions = [], []
    teacher_rows = []
    for seed in range(config["training_seed"], config["training_seed"] + config["warmstart_episodes"]):
        obs, _ = env.reset(seed=seed)
        while True:
            action = env.scene.baseline_action()
            observations.append(obs.copy())
            actions.append(action)
            obs, _, done, truncated, _ = env.step(action)
            if done or truncated:
                break
        teacher_rows.append(env.scene.report())
        print(f"teacher {seed}: success={env.scene.success}, decisions={env.decisions}", flush=True)
        if not env.scene.success:
            raise RuntimeError("Physical teacher failed. Fix the frozen skills before head training.")
    (output / "teacher.json").write_text(json.dumps(teacher_rows, indent=2) + "\n")
    obs = torch.as_tensor(np.asarray(observations), device=model.device)
    labels = torch.as_tensor(actions, dtype=torch.long, device=model.device)
    counts = torch.bincount(labels, minlength=4).float()
    if torch.any(counts == 0):
        raise RuntimeError("Teacher data is missing a coordinator action")
    weights = counts.sum() / (4 * counts)
    model.policy.set_training_mode(True)
    for epoch in range(config["bc_epochs"]):
        order = torch.randperm(len(obs), device=model.device)
        for batch in order.split(128):
            logits = model.policy.get_distribution(obs[batch]).distribution.logits
            loss = F.cross_entropy(logits, labels[batch], weight=weights)
            model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), .5)
            model.policy.optimizer.step()
        if epoch % 25 == 0 or epoch == config["bc_epochs"] - 1:
            with torch.no_grad():
                prediction = model.policy.get_distribution(obs).distribution.logits.argmax(1)
                per_action = [float((prediction[labels == i] == i).float().mean()) for i in range(4)]
            print(f"warmstart epoch {epoch + 1}: accuracy by action={per_action}", flush=True)
    np.savez_compressed(output / "teacher_observations.npz", observations=obs.cpu().numpy(),
                        actions=labels.cpu().numpy())


class ValidationCallback(BaseCallback):
    def __init__(self, config, output, frozen, initial_score):
        super().__init__()
        self.config, self.output, self.frozen = config, output, frozen
        self.best_score = initial_score
        self.next_check = 1024

    def _on_step(self):
        if self.num_timesteps < self.next_check:
            return True
        self.next_check += 1024
        self.validate()
        return True

    def validate(self):
        if verify() != self.frozen:
            raise RuntimeError("Frozen controller bundle changed during training")
        rows = evaluate(self.model, self.config, seed=self.config["validation_seed"],
                        episodes=self.config["evaluation_episodes"])
        score = (sum(row["success"] for row in rows), sum(row["reward"] for row in rows))
        print(f"validation at {self.num_timesteps}: {score[0]}/{len(rows)} successful", flush=True)
        (self.output / f"validation_{self.num_timesteps}.json").write_text(json.dumps(rows, indent=2) + "\n")
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.output / "best_head")


def train(*, timesteps=None, device="auto", warmstart_episodes=None, bc_epochs=None):
    config = read_config()
    for key, value in (("timesteps", timesteps), ("warmstart_episodes", warmstart_episodes), ("bc_epochs", bc_epochs)):
        if value is not None:
            config[key] = value
    if config["timesteps"] < 1 or config["warmstart_episodes"] < 1 or config["bc_epochs"] < 1:
        raise ValueError("Training and warmstart budgets must be positive")
    frozen = verify()
    torch.set_num_threads(1)
    device = resolve_device(device)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = ROOT / "outputs/mujoco_cooperation" / stamp
    output.mkdir(parents=True)
    metadata = dict(config=config, frozen=frozen, device=device, trainable="coordinator only")
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    raw = environment(config)
    env = Monitor(raw, filename=str(output / "monitor.csv"))
    start = time.monotonic()
    try:
        model = PPO("MlpPolicy", env, device=device, seed=config["training_seed"], verbose=1,
                    n_steps=config["n_steps"], batch_size=config["batch_size"],
                    learning_rate=config["learning_rate"], n_epochs=5, ent_coef=.005,
                    gamma=.99, policy_kwargs={"net_arch": config["net_arch"]})
        metadata["trainable_parameters"] = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
        print(f"Output: {output}\nTrainable head parameters: {metadata['trainable_parameters']}", flush=True)
        warmstart(model, raw, config, output)
        rows = evaluate(model, config, seed=config["validation_seed"], episodes=config["evaluation_episodes"])
        (output / "validation_warmstart.json").write_text(json.dumps(rows, indent=2) + "\n")
        score = (sum(row["success"] for row in rows), sum(row["reward"] for row in rows))
        model.save(output / "warmstart_head")
        model.save(output / "best_head")
        print(f"warmstart validation: {score[0]}/{len(rows)}", flush=True)
        if score[0] == 0:
            raise RuntimeError("Warmstart failed validation; stopped before spending time on PPO. Inspect teacher/validation logs.")
        callback = ValidationCallback(config, output, frozen, score)
        model.learn(total_timesteps=config["timesteps"], callback=callback)
        model.save(output / "final_head")
        callback.validate()
        if verify() != frozen:
            raise RuntimeError("Frozen controllers changed")
        metadata.update(seconds=time.monotonic() - start, actual_timesteps=model.num_timesteps,
                        best_validation_successes=callback.best_score[0], completed=True)
        (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        latest = ROOT / "outputs/mujoco_cooperation/latest.json"
        latest.write_text(json.dumps({"run": output.name}, indent=2) + "\n")
        print(f"Finished. Evaluate: python -m experiments.run evaluate --checkpoint {output / 'best_head.zip'}", flush=True)
    finally:
        env.close()
    return output


def load_head(checkpoint=None, device="cpu"):
    frozen = verify()
    if checkpoint is None:
        latest = ROOT / "outputs/mujoco_cooperation/latest.json"
        if not latest.is_file():
            raise FileNotFoundError("No completed head training. Run train, or use baseline to watch the fixed skills.")
        checkpoint = latest.parent / json.loads(latest.read_text())["run"] / "best_head.zip"
    checkpoint = Path(checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    metadata = json.loads((checkpoint.parent / "run.json").read_text())
    if metadata["frozen"] != frozen:
        raise RuntimeError("Checkpoint was trained with different frozen skills or scene")
    return PPO.load(checkpoint, device=resolve_device(device)), metadata["config"]

"""Check, train, and evaluate the graph-conditioned frozen-skill baseline."""

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    from src.runtime import configure_caches
    configure_caches()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("list", "check", "preflight", "train", "evaluate", "baseline"))
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/configs/composition_training.json")
    parser.add_argument("--tasks", type=Path, default=ROOT / "experiments/configs/composition_tasks.json")
    parser.add_argument("--asset-root", type=Path, default=ROOT / "data/calvin")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/composition")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="Continue a saved head run, with its optimizer, into a new run directory")
    parser.add_argument("--device", help="auto, cpu, or cuda:0; auto uses a selected visible CUDA GPU, else CPU")
    parser.add_argument("--renderer", choices=("tiny", "egl", "auto"))
    parser.add_argument("--timesteps", type=int, help="Head decisions, not low-level simulation frames")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--task", help="Evaluate or check one named task; does not filter training")
    parser.add_argument("--seeds", type=int, nargs="+", help="Override evaluation seeds")
    parser.add_argument("--policy", choices=("head", "scripted", "random"), default="head")
    parser.add_argument("--checkpoint", choices=("best", "final", "initial", "warmstart"), default="best")
    parser.add_argument("--allow-weak-base", action="store_true", help="Explicitly allow training despite failed frozen-skill preflight")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=10)
    args = parser.parse_args()
    if args.mode == "list":
        data = json.loads((ROOT / "experiments/configs/calvin_skills.json").read_text())
        for skill in data["skills"]:
            print(f"{skill['id']}: {skill['instruction']}")
        return
    missing = [name for name in ("torch", "gymnasium", "stable_baselines3", "calvin_env", "hulc")
               if importlib.util.find_spec(name) is None]
    if missing:
        parser.exit(1, f"Missing composition dependencies: {', '.join(missing)}. Run: python -m experiments.setup\n")
    from src.training.composition import Experiment, Settings, evaluate, train, save_json
    from stable_baselines3 import PPO
    from datetime import datetime, timezone

    values = json.loads(args.config.read_text())
    catalog = None
    run_dir = args.run_dir
    if args.resume is not None:
        if args.mode != "train":
            parser.error("--resume is only used for training")
        values = json.loads((args.resume / "config.json").read_text())
        args.tasks = args.resume / "tasks.json"
        catalog = args.resume / "catalog.json"
    if args.mode == "evaluate" and args.policy == "head":
        if run_dir is None:
            completed = [p for p in args.output_dir.glob("*")
                         if (p / "head.zip").is_file() and (p / "status.json").is_file()
                         and json.loads((p / "status.json").read_text()).get("status") == "complete"]
            if not completed:
                parser.error("No completed training run found; specify --run-dir or train the head first")
            run_dir = max(completed)
        values = json.loads((run_dir / "config.json").read_text())
        args.tasks = run_dir / "tasks.json"
        catalog = run_dir / "catalog.json"
    for key in ("device", "renderer"):
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    if args.timesteps is not None:
        values["total_timesteps"] = args.timesteps
    settings = Settings(**values)
    experiment = Experiment(settings, args.tasks, args.asset_root, catalog_path=catalog)
    if args.mode == "train":
        if args.task or args.seeds:
            parser.error("Use the task/config files to change training; --task and --seeds are evaluation options")
        train(experiment, args.output_dir, allow_weak_base=args.allow_weak_base, resume_dir=args.resume)
        return
    split = "train" if args.mode == "check" else args.split
    env = experiment.make_env(split)
    writer = None
    try:
        if args.mode == "evaluate" and args.policy == "head" and split == "train":
            qualified = json.loads((run_dir / "qualified_tasks.json").read_text())["used_for_training"]
            env.tasks = tuple(t for t in env.tasks if t.name in qualified)
        if args.task:
            env.tasks = tuple(t for t in env.tasks if t.name == args.task)
            if not env.tasks:
                parser.error(f"Task {args.task} does not belong to split {split}")
        if args.mode == "check":
            obs, _ = env.reset(seed=settings.seed)
            assert env.observation_space.contains(obs)
            head = PPO("MlpPolicy", env, n_steps=2, batch_size=2, device=experiment.device,
                       policy_kwargs=dict(net_arch=dict(pi=list(settings.hidden_sizes), vf=list(settings.hidden_sizes))))
            action = int(head.predict(obs, deterministic=True)[0])
            assert env.action_space.contains(action)
            assert not ({id(p) for p in head.policy.parameters()} & {id(p) for p in env.backend.model.parameters()})
            if args.smoke_steps < 1:
                parser.error("--smoke-steps must be positive")
            skill = env.task.sequence[0]
            success, steps, _ = env.backend.execute(skill, args.smoke_steps)
            env.backend.assert_frozen()
            print(json.dumps(dict(head_observation_dim=len(obs), active_skills=len(env.skills),
                                  head_forward_pass=True,
                                  frozen_parameters_unchanged=True, smoke_frames=steps,
                                  smoke_skill_success=success, training_performed=False), indent=2))
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        output = args.output_dir / f"{args.mode}_{stamp}"
        output.mkdir(parents=True)
        model = None
        policy = "scripted" if args.mode in ("preflight", "baseline") else args.policy
        if policy == "head":
            saved = json.loads((run_dir / "contract.json").read_text())
            actual = experiment.contract(env)
            if saved != actual:
                mismatches = [key for key in saved if saved[key] != actual.get(key)]
                raise ValueError(f"Saved head/environment contract differs: {mismatches}")
            filename = {"best": "best_head.zip", "final": "head.zip", "initial": "initial_head.zip", "warmstart": "warmstart_head.zip"}[args.checkpoint]
            checkpoint = run_dir / filename
            if args.checkpoint == "best" and not checkpoint.is_file():
                checkpoint = run_dir / "head.zip"
            model = PPO.load(str(checkpoint), device=experiment.device)
            if model.observation_space != env.observation_space or model.action_space != env.action_space:
                raise ValueError("Saved head spaces do not match this environment")
        if args.video:
            import imageio.v2 as imageio
            writer = imageio.get_writer(str(output / "rollout.mp4"), fps=30, macro_block_size=1)
            env.backend.on_frame = lambda scene, skill, step: writer.append_data(scene.render(mode="rgb_array"))
        seeds = args.seeds or (settings.preflight_seeds if args.mode == "preflight" else settings.test_seeds if split == "test" else settings.validation_seeds)
        result = evaluate(env, model=model, seeds=seeds, random_actions=policy == "random")
        result.update(policy=policy, split=split, contract=experiment.contract(env), config=asdict(settings),
                      head_checkpoint=str(checkpoint) if policy == "head" else None)
        save_json(output / "results.json", result)
        print(f"Success rate: {result['success_rate']:.1%}. Results: {output}")
    finally:
        if writer is not None:
            writer.close()
        env.close()


if __name__ == "__main__":
    main()

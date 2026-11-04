"""Run validation in a fresh process so CALVIN always owns physics client 0."""

import json
from pathlib import Path
import subprocess
import sys


class SubprocessValidation:
    def __init__(self, experiment, run_dir):
        self.experiment, self.run_dir = experiment, Path(run_dir)

    def evaluate(self, model, tag):
        directory = self.run_dir / "validation"
        directory.mkdir(exist_ok=True)
        checkpoint = directory / "candidate.zip"
        model.save(str(checkpoint))
        output = directory / f"{tag}.json"
        root = Path(__file__).resolve().parents[2]
        log_path = directory / f"{tag}.log"
        try:
            with log_path.open("w") as log:
                subprocess.run([
                    sys.executable, "-m", "src.training.validation", "--run-dir", str(self.run_dir.resolve()),
                    "--checkpoint", str(checkpoint.resolve()), "--output", str(output.resolve()),
                    "--asset-root", str(self.experiment.asset_root),
                ], cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Validation process failed; inspect {log_path}") from error
        result = json.loads(output.read_text())
        if not result.get("base_frozen"):
            raise RuntimeError("Validation did not verify frozen base parameters")
        return result

    def close(self):
        pass  # Each synchronous child has already exited.


def main():
    import argparse
    from stable_baselines3 import PPO
    from src.training.composition import Settings, Experiment, evaluate, save_json
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "checkpoint", "output", "asset-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    settings = Settings(**json.loads((args.run_dir / "config.json").read_text()))
    experiment = Experiment(settings, args.run_dir / "tasks.json", args.asset_root,
                            catalog_path=args.run_dir / "catalog.json")
    env = experiment.make_env("train")
    try:
        if experiment.contract(env) != json.loads((args.run_dir / "contract.json").read_text()):
            raise ValueError("Validation process does not match the training environment contract")
        names = json.loads((args.run_dir / "qualified_tasks.json").read_text())["used_for_training"]
        env.tasks = tuple(task for task in env.tasks if task.name in names)
        model = PPO.load(str(args.checkpoint), device=experiment.device)
        result = evaluate(env, model=model, seeds=settings.validation_seeds)
        result["base_frozen"] = True
        save_json(args.output, result)
    finally:
        env.close()


if __name__ == "__main__":
    main()

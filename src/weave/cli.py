"""Command-line interface for the WEAVE research project."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import traceback
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs/env"
DEFAULT_RESULTS = ROOT / "results/phase0"
DEFAULT_DATA = ROOT / "data/robofactory"


def load_config() -> dict[str, Any]:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(CONFIGS), version_base="1.3"):
        config = OmegaConf.to_container(compose(config_name="phase0"), resolve=True)
    if config["replay"]["episodes_per_task"] < 50:
        raise ValueError(
            "The Phase 0 protocol requires at least 50 train episodes per task."
        )
    return config


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(
        [
            *(ROOT / "src/weave").rglob("*.py"),
            *CONFIGS.glob("*.*"),
            ROOT / "src/skill_composer/data.py",
            ROOT / "src/skill_composer/common.py",
            *sorted((ROOT / "scripts").glob("*.sh")),
        ]
    ):
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _write_result(
    path: Path,
    mode: str,
    config: dict[str, Any],
    result: dict[str, Any],
    seeds: list[int],
) -> None:
    stamp = ROOT / ".weave-source-revision"
    if stamp.is_file():
        git_sha = stamp.read_text().strip()
    else:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    if len(git_sha) not in (40, 64) or any(c not in "0123456789abcdef" for c in git_sha):
        raise ValueError("Invalid source Git revision. Run make sync from the source checkout.")
    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    payload = {
        "mode": mode,
        "git_sha": git_sha,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "implementation_sha256": _implementation_hash(),
        "benchmark_plan": {
            "tasks": config["benchmark"]["tasks"],
            "env_counts": config["benchmark"]["env_counts"],
        },
        "checkpoint_sha256": {},
        "seeds": seeds,
        "result": result,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(path)


def _audit(args: argparse.Namespace) -> int:
    from .phase0_data import inventory

    config = load_config()
    mode = args.mode
    result_id = mode
    context: dict[str, Any] = {}
    if mode == "benchmark":
        if args.task not in config["benchmark"]["tasks"] or args.num_envs < 1:
            raise ValueError(
                "Benchmark requires a configured two-Panda task and positive --num-envs."
            )
        if args.backend == "cpu" and args.num_envs != 1:
            raise ValueError("Native CPU backend accepts one environment per process.")
        result_id = f"benchmark-{args.task}-{args.backend}-{args.num_envs}"
        context = {
            "task": args.task,
            "num_envs": args.num_envs,
            "backend": args.backend,
        }

    seeds: list[int] = []
    try:
        if mode == "inventory":
            result = inventory(args.data_dir, config)
            seeds = sorted(
                {
                    seed
                    for task in result["tasks"].values()
                    for seed in task["episode_seeds"]
                }
            )
        else:
            from .phase0_sim import benchmark, replay

            if mode == "replay":
                result = replay(args.data_dir, config)
                seeds = sorted(
                    {
                        row["episode_seed"]
                        for task in result["tasks"].values()
                        for row in task.get("episodes", [])
                    }
                )
            else:
                benchmark_config = config["benchmark"]
                start_seed = benchmark_config["seed"]
                seeds = list(
                    range(start_seed, start_seed + benchmark_config["repeats"])
                )
                result = benchmark(config, args.task, args.num_envs, args.backend)
    except Exception as exc:  # noqa: BLE001 - retain evidence from failed simulator runs
        result = {**context, "error": f"{type(exc).__name__}: {exc}"}
        traceback.print_exc()

    _write_result(args.results_dir / f"{result_id}.json", mode, config, result, seeds)
    failed = "error" in result or any(
        "error" in task for task in result.get("tasks", {}).values()
    )
    if mode == "benchmark" and "contact_probe" in result:
        failed = failed or not result["contact_probe"]["passed"]
    return int(failed)


def main(argv: Optional[list[str]] = None) -> int:  # noqa: UP045 - Python 3.9 compatibility
    parser = argparse.ArgumentParser(prog="weave")
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser(
        "phase0-audit", help="Verify data, replay demos, or benchmark a scene."
    )
    audit.add_argument(
        "--mode", required=True, choices=("inventory", "replay", "benchmark")
    )
    audit.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    audit.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    audit.add_argument("--task")
    audit.add_argument("--num-envs", type=int, default=1)
    audit.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")

    download = commands.add_parser(
        "download-data", help="Fetch and verify the pinned demo files."
    )
    download.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)

    report = commands.add_parser(
        "report", help="Render a Markdown report from audit JSON."
    )
    report.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    report.add_argument("--output", type=Path)

    args = parser.parse_args(argv)
    if args.command == "phase0-audit":
        return _audit(args)
    if args.command == "download-data":
        from .phase0_data import verify_files

        verify_files(args.data_dir, download=True)
        return 0

    from .eval.report import phase0_report

    rendered = phase0_report(args.results_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

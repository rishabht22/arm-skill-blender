"""Run prespecified scaling rosters over one physically collected corpus."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .evaluate import TEST, read_lock
from .learn import checkpoint_sha
from .protocol import read_manifest, write_json


SEEDS = (17, 59, 101)
# Each pair is (selector width, physical expert episodes). The repeated axis
# intersection is fit once. Test counts follow protocol.TEST order.
PLANS = {
    "pilot": {
        "roster": ((64, 384), (16, 384), (256, 384), (64, 4), (64, 32)),
        "dev_episodes": 96, "epochs": 8, "tests": None,
    },
    "paired": {
        "roster": ((64, 512),),
        "dev_episodes": 128, "epochs": 8, "tests": (2048, 256, 256, 256, 256),
    },
    "full": {
        "roster": ((64, 128), (64, 2048), (16, 32768), (64, 32768), (256, 32768),
                   (1024, 32768), (64, 1_000_000)),
        "dev_episodes": 1024, "epochs": 6, "tests": (2048, 256, 256, 256, 256),
    },
}


def contiguous(directory):
    end = 0
    for path in sorted(directory.glob("*.jsonl.gz")):
        left, right = map(int, path.name.split(".", 1)[0].split("-"))
        if left != end:
            break
        end = right
    return end


def wait_for(directory, required, enabled):
    available = contiguous(directory)
    if available >= required:
        return
    if not enabled:
        raise ValueError(f"{directory}: {available} of {required} episode ranks available")
    print(f"Waiting for {required} episodes in {directory} ({available} completed in prefix)", flush=True)
    while contiguous(directory) < required:
        time.sleep(20)


def call(log, *args):
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {args[0]} {log}", flush=True)
    with log.open("ab") as stream:
        subprocess.run([sys.executable, "-m", "larger_scale", *map(str, args)],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", choices=PLANS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--dev-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--wait-for-shards", action="store_true")
    args = parser.parse_args()
    plan = PLANS[args.plan]
    _, _, manifest = read_manifest(args.manifest, "dev")
    declaration = {"manifest": manifest, "plan": args.plan, "seeds": SEEDS, **plan}
    args.out.mkdir(parents=True, exist_ok=True)
    declaration_path = args.out / "plan.json"
    if declaration_path.exists():
        if json.loads(declaration_path.read_bytes()) != json.loads(json.dumps(declaration)):
            raise ValueError("study plan conflicts with existing declaration")
    else:
        write_json(declaration_path, declaration)
    wait_for(args.dev_data, plan["dev_episodes"], args.wait_for_shards)
    paths = []
    for width, n in plan["roster"]:
        wait_for(args.train_data, n, args.wait_for_shards)
        for seed in SEEDS:
            path = args.out / "models" / f"w{width}-n{n}-s{seed}.pt"
            metadata = path.with_suffix(".json")
            if path.exists() and metadata.exists():
                saved = json.loads(metadata.read_bytes())
                if (saved["sha256"], saved["manifest"], saved["width"], saved["fit"]["episodes"],
                        saved["seed"], saved["dev"]["episodes"], saved["epoch_budget"]) != (
                        checkpoint_sha(path), manifest, width, n, seed, plan["dev_episodes"], plan["epochs"]):
                    raise ValueError(f"invalid existing checkpoint or study mismatch at {path}")
            else:
                call(path.with_suffix(".log"), "train", "--manifest", args.manifest, "--train-data",
                     args.train_data, "--dev-data", args.dev_data, "--episodes", n, "--dev-episodes",
                     plan["dev_episodes"], "--width", width, "--seed", seed, "--epochs",
                     plan["epochs"], "--batch", 64, "--threads", args.threads, "--device",
                     args.device, "--out", path)
            paths.extend(("--model", path))
    if args.fit_only:
        return
    if plan["tests"] is None:
        dest = args.out / "dev_eval"
        # Compare the full pilot scaling roster on development starts only.
        call(dest.with_suffix(".log"), "evaluate", "--manifest", args.manifest, "--split", "dev",
             "--count", 48, "--workers", args.workers, "--out", dest, *paths)
        call(args.out / "report.log", "report", "--out", args.out / "dev_report.json", dest)
        return
    lock_path = args.out / "test_lock.json"
    if not lock_path.exists():
        call(args.out / "lock.log", "lock", "--manifest", args.manifest, "--out", lock_path,
             "--counts", *plan["tests"], *paths)
    else:
        read_lock(lock_path, manifest)
    destinations = []
    for split, count in zip(TEST, plan["tests"]):
        dest = args.out / split
        call(dest.with_suffix(".log"), "evaluate", "--manifest", args.manifest, "--split", split,
             "--count", count, "--workers", args.workers, "--out", dest, "--lock", lock_path, *paths)
        destinations.append(dest)
    call(args.out / "report.log", "report", "--out", args.out / "test_report.json", *destinations)


if __name__ == "__main__":
    main()

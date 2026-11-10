"""Plot observed closed-loop model and data scaling from a paired report."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def plot(report, split, output, width, episodes):
    result = report["splits"][split]
    total = result["matched_starts"]
    runs = defaultdict(list)
    for checkpoint in result["checkpoints"]:
        key = checkpoint["width"], checkpoint["episodes"]
        method = result["methods"][f"learned-w{key[0]}-n{key[1]}"]
        runs[key].append(checkpoint | {
            "success": method["seed_successes"][checkpoint["sha256"]] / total,
        })
    panels = [
        ("Demonstration data", "Expert decision examples", "decisions",
         [group for (w, _), group in runs.items() if w == width]),
        ("Selector capacity", "Trainable parameters", "parameters",
         [group for (_, n), group in runs.items() if n == episodes]),
    ]
    if any(len(groups) < 2 for _, _, _, groups in panels):
        raise ValueError("need two measured points on both scaling axes")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), sharey=True, layout="constrained")
    expert = result["methods"]["expert"]["micro_success"]
    for ax, (title, xlabel, field, groups) in zip(axes, panels):
        groups.sort(key=lambda group: group[0][field])
        by_seed = defaultdict(list)
        for group in groups:
            for run in group:
                by_seed[run["seed"]].append((run[field], run["success"]))
        for i, (seed, points) in enumerate(sorted(by_seed.items())):
            x, y = zip(*points)
            ax.plot(x, y, "o-", linewidth=1.2, markersize=5, color=("#2a7886", "#994568", "#c87824")[i % 3],
                    alpha=.72, label=f"Training seed {seed}")
        ax.axhline(expert, linestyle="--", color="#283b52", linewidth=1, label="Physical expert")
        ticks = [g[0][field] for g in groups]
        ax.set_xscale("log")
        ax.set_xticks(ticks, [f"{n:,}" for n in ticks])
        ax.minorticks_off()
        ax.grid(axis="y", color="#dbe4ed", linewidth=.8)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylim(-.035, 1.06)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Closed-loop task completion")
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axes[1].legend(loc="lower right", fontsize=8, frameon=False)
    label = "development" if split == "dev" else split
    fig.suptitle(f"Aster scaling on {total:,} matched {label} starts", fontweight="bold")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=384)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_bytes())
    plot(report, args.split, args.out, args.width, args.episodes)


if __name__ == "__main__":
    main()

# Goal-driven skill composition

The policy receives desired physical outcomes and the observed Aster cell
state. For each carrier, the goal can require inspection, a specified number
of index cycles, return to feed, or sorting by measured grade. A goal can
also require inspection of one carrier before indexing another.

The policy scores all 48 skill-and-carrier pairs after each action. It
receives neither an expanded task graph nor the planner's action mask.
MuJoCo executes the selected skill using the frozen Aster controllers. The
episode ends when the observed outcomes satisfy the goal or the rollout
encounters a failed request. Sorting dispatches to release or reject using
the observed grade.

## Training and evaluation

A breadth-first planner predicts skill effects in a discrete representation
of the cell. It labels all shortest-path choices for each demonstrated
state. Training can also execute planner demonstrations in MuJoCo with
`--physical`. Reports separate symbolic states, physical demonstrations,
and measured decisions. Model selection uses the abstract development
suite. The physical benchmark compares each learned seed, the replanning
expert, and a fixed script planned from an all-feed starting state. All
methods use the same seeded physical reset.

The benchmark uses four suites with independent seeds.

| Suite | Change from training |
|---|---|
| `novel_goal` | Unseen pairs of outcome profiles |
| `more_carriers` | Three goal carriers where training uses one or two |
| `partial` | A carrier starts loaded with the enclosure closed or latched |
| `ordering` | A held-out goal combination where the first carrier must be sorted and its inspection must precede indexing the second carrier |

Each shard stores the scenario, starting-state fingerprint, executed skill
requests, and result. The report verifies coverage and matching fingerprints
before computing success, invalid requests, excess skills, and paired
differences. Multiple seeds see the same starts. The report counts distinct
goals separately from reset instances.

## Watch the trained policy

From `projects/single_arm_composition`, use the pinned Aster environment
and a checkpoint produced by this subproject. The viewer prints the goal
and the highest-scoring choices. MP4 export adds them above the workcell.
The first command should take about 5–15 seconds per rollout at 4× speed
on the Mac. The MP4 command should take roughly 5–20 seconds per rollout.

```sh
mjpython -m goal_driven show --model goal_driven/outputs/final/w64-s17.pt \
  --start feed --show --speed 4
mjpython -m goal_driven show --model goal_driven/outputs/final/w64-s17.pt \
  --start loaded --show --speed 4
```

```sh
python -m goal_driven show --model goal_driven/outputs/final/w64-s17.pt \
  --start loaded --video goal_driven/outputs/loaded.mp4 --speed 3
```

Use `--goal /path/to/goal.json` in place of `--start` to provide outcomes.
This example starts B loaded. The model chooses the steps to reach the
requested outcome.

```json
{
  "carriers": {
    "A": {"inspect": true, "destination": "sorted"},
    "B": {"cycles": 1, "destination": "feed"}
  },
  "read_before_index": ["A", "B"],
  "start": {"carrier": "B", "location": "process", "enclosure": "open"}
}
```

## Lab runs

Copy the new subproject from the Mac repo root. Estimated time is 10–60
seconds with the Berkeley VPN or campus Wi-Fi.

```sh
rsync -av --exclude='outputs/' --exclude='__pycache__/' \
  projects/single_arm_composition/goal_driven/ \
  rishabh@icon.ist.berkeley.edu:arm-skill-blender-aster/projects/single_arm_composition/goal_driven/
```

In a server tmux pane, use `nvtop` to find one idle GPU. Replace the example
GPU number with the one you selected. This uses the existing `aster`
environment and writes to your NAS directory. Estimated time is 1–5 minutes
for all three seeds, depending on startup and the server.

```sh
cd ~/arm-skill-blender-aster/projects/single_arm_composition &&
conda activate aster &&
export CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false &&
run="/ICONAS/datasets/$USER/aster-goal-driven" &&
(
  for seed in 17 59 101; do
    python -m goal_driven train --out "$run/w64-s$seed.pt" \
      --seed "$seed" --problems 1800 --physical 24 --device cuda || exit 1
  done
)
```

The physical benchmark uses server CPU workers. The example below runs
4,000 matched starts and 20,000 method rollouts. Estimated time is 20–50
minutes with four workers. Rerun with the same arguments to resume completed
shards. To distribute a suite across hosts, assign unique `--part` values
from zero through `--parts - 1` while keeping count, batch, checkpoints, and
output directory identical.

```sh
cd ~/arm-skill-blender-aster/projects/single_arm_composition &&
conda activate aster &&
export CUDA_VISIBLE_DEVICES="" &&
run="/ICONAS/datasets/$USER/aster-goal-driven" &&
(
  for suite in novel_goal more_carriers partial ordering; do
    python -m goal_driven evaluate --suite "$suite" --count 1000 --workers 4 \
      --model "$run/w64-s17.pt" --model "$run/w64-s59.pt" --model "$run/w64-s101.pt" \
      --out "$run/benchmark/$suite" || exit 1
  done
  python -m goal_driven report --out "$run/benchmark/report.json" \
    "$run"/benchmark/{novel_goal,more_carriers,partial,ordering}
)
```

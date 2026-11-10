# Arm Skill Blender research projects

Nine projects recovered from the 13 commits through `7658d01`. Each folder has
a short `SUMMARY.txt` for a demo conversation and a README with setup, execution,
and training commands. The commits identify the original versions.

| Project | What to show | Source commits |
| --- | --- | --- |
| [MuJoCo reach and lift](projects/mujoco_reach/) | UR5e and Jaco motion, independent reach policies, and the first joint lift pilot | `cd02d9b`–`e915f3e` |
| [CALVIN composition](projects/calvin_composition/) | A selector calling frozen HULC skills in a Panda play scene | `907ca27` |
| [MuJoCo manipulation](projects/mujoco_manipulation/) | A Panda stacks and relocates a cube using contact physics | `d07de72` |
| [Planar dispatch](projects/planar_dispatch/) | Mobile pusher and lifter collecting cubes in PyBullet | `e4d6606` |
| [Precision assembly](projects/precision_assembly/) | Two Panda stations schedule rods, gears, connectors, and nuts | `e4d6606` |
| [RoboFactory LiftBarrier](projects/robofactory_lift/) | Two Panda arms grasp and lift one bar together | `779a37a`, executor updates in `de1f3a4` |
| [RoboFactory multi-task cell](projects/robofactory_cell/) | Relay, stack, and lid-opening skills scheduled in a shared cell | `de1f3a4` |
| [Goal-conditioned workbench](projects/goal_composition/) | Direct versus relay routing and a three-block tower | `7658d01` |
| [WEAVE](projects/weave/) | Native replay audit and the GPU cell smoke test | `7d052b4`–`7658d01` |

[Single-arm skill composition](projects/single_arm_composition/) includes the original Panda workshop, an interlocked packing cell, and a connector service bench in robosuite.

## Run the RoboFactory demos

Use Python 3.10 or newer. From the repository root, install the three MuJoCo
RoboFactory projects together. They share the licensed Panda assets and executor
code, and keep their datasets, checkpoints, and tests in their own folders.

```sh
python3 -m venv .venv-mujoco
source .venv-mujoco/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

Each command below uses a saved checkpoint and works without the raw training
recordings. Run it from the corresponding project folder. Add `--viewer` for a
live window and use `mjpython` instead of `python` on macOS when opening a viewer.

```sh
(cd projects/robofactory_lift && python -m skill_composer baseline rollout --run outputs/demo)
(cd projects/robofactory_cell && python -m skill_composer rollout --task relay_stack --run outputs/demo)
(cd projects/goal_composition && python -m skill_composer compose rollout --layout relay --run outputs/demo)
(cd projects/goal_composition && python -m skill_composer compose tower rollout --controller guided --layout occupied-dock --run outputs/demo/tower)
```

There are short videos in the [LiftBarrier](projects/robofactory_lift/outputs/demo/preview.mp4)
and [multi-task cell](projects/robofactory_cell/outputs/demo/preview.mp4) folders,
as well as a [planar collection demo](projects/planar_dispatch/outputs/demo/preview.mp4).
Training and full evaluation use the data acquisition steps in each README.

## Run the other projects

Use the setup and test commands inside each historical project folder. Those
projects use their original per-project Python dependencies and module names.
The root test command runs the current RoboFactory and WEAVE suites. Native WEAVE
GPU runs use the separate [NVIDIA environment](projects/weave/docs/REMOTE_RUN.md).

Raw recordings, upstream vendor checkouts, and regenerated training runs are
local to each project. The repository includes the small demo checkpoints and
the historical WEAVE audit evidence.

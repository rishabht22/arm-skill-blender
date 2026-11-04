# Frozen-skill composition experiment

This project trains a skill selector over the released HULC_D_D controller in
CALVIN-D. The base controller stays frozen. It is a language-conditioned latent
plan policy, not a diffusion policy. The selector is a single-arm baseline with
privileged simulator state and a supplied goal graph. It does not yet study
decentralized coordination, learned task dependencies, interruption, or recovery.

## Setup and commands

Run from the project root on either macOS or Linux. Conda must be installed.

```bash
python -m experiments.setup
python -m experiments.run list
python -m experiments.run check
python -m experiments.run preflight --split train
python -m experiments.run train
python -m experiments.run evaluate --video
```

Setup creates a project-local Python 3.10 environment. The launcher chooses it
without requiring activation. Both machines use the same dependency versions,
config files, action order, scene, and entry points. Install each machine's own
environment; never copy an environment directory between operating systems.

On the shared server, select a GPU you are permitted to use with
`CUDA_VISIBLE_DEVICES` before starting. `auto` uses the selected visible GPU;
without a selection it uses CPU. It does not check occupancy or wait for a GPU.
`--device cpu` explicitly runs on CPU on either machine. Nothing queues jobs.

## Files and assets

| Location | Responsibility |
| --- | --- |
| `experiments/configs/composition_training.json` | Training budgets and optimizer settings |
| `experiments/configs/composition_tasks.json` | Active skills, train/test compositions, starting states, final checks |
| `experiments/configs/calvin_skills.json` | All 34 released task instructions |
| `src/policies/calvin.py` | Load official weights and original transforms |
| `src/policies/frozen_skills.py` | Inference-only option execution and freeze verification |
| `src/envs/calvin_portable.py` | Native CALVIN scene with selectable rendering |
| `src/envs/composition.py` | Shared training and evaluation environment |
| `src/coordination/tasks.py` | Task definitions and split validation |
| `src/training/composition.py` | Graph warm-start, PPO, checkpoints, and evaluation |
| `outputs/composition/` | Run snapshots, metrics, heads, baseline results, and videos |

The original 34 tasks share one model. The starter experiment activates 12
skills and has 11 single-skill training cases, 9 candidate training compositions,
and 8 held-out compositions. Add tasks
and active skill IDs in the task config; changing the action order or goal
encoding requires a new head. Old reaching checkpoints remain on disk but are
not used by this project.

Assets live in `data/calvin/`:

* `models/HULC_D_D/`: official checkpoint and its original Hydra configuration.
* `datasets/task_D_D/validation/`: original scene configuration, statistics, and
  matching MiniLM language embeddings.
* `datasets/task_D_D/training/statistics.yaml`: original normalization metadata.
* `vendor/`: pinned official CALVIN and HULC source trees.

Setup downloads the authors' released assets. It fetches only inference metadata
from the approximately 177 GB dataset ZIP using HTTP ranges. Demonstration
episodes are not required for this experiment. Source and weights are not
modified by the head training code. See `CALVIN_LICENSE.txt` for attribution.

## Environment fidelity

The loader uses the released D scene, Panda model, cameras, native relative
7-dimensional actions, control frequency, original validation transforms,
normalization statistics, language embeddings, and task-success oracle. Neither
the head nor the reward code changes the low-level action coordinates or scale.

Training and evaluation instantiate the same `CompositionEnv` and base-policy
loader. Episodes start from explicit poses adapted from the native evaluation
utility. Only the episode reset changes the scene. A skill boundary resets the
controller's latent plan, not the robot, objects, or scene.

The default renderer is PyBullet's software renderer (`tiny`) on both platforms,
so switching between CPU and CUDA does not silently change the image backend.
The original policy was evaluated with hardware rendering. Software-rendered
pixels can differ, even though the scene geometry and camera settings match.
This is why real frozen-policy baselines are required, not assumed.
Linux can use `--renderer egl` for comparison. That changes the saved experiment
contract and requires its own baseline; evaluation rejects renderer mismatches.
CPU architecture and physics builds can also introduce small numerical changes.
This is not claimed to reproduce the official 1000-sequence CALVIN results.

## What the head learns

Each action selects one active skill. The base model receives its corresponding
released language instruction and controls the arm until native success or the
360-frame limit. The head does not produce joint commands.

The head observes the physical robot/scene state, required skill indicators,
completed event indicators, a prerequisite adjacency matrix, and remaining
budgets. The starter tasks are chains represented as goal graphs. The supplied
graph exposes the desired ordering. This tests learned graph-conditioned skill
selection and physical handoffs, not discovery of an unknown plan.

An invalid selection receives a penalty and consumes a head decision without
moving the robot. No action mask supplies the correct answer to the actor.
Successful skills receive +1, full completion +5, failed options -1, invalid
selections -0.25, and executed frames cost 0.0001 each. The discount is per head
decision, not per simulation frame. A failed frozen option ends the episode.
Final predicates are checked again to catch outcomes lost after later skills.

The head first receives supervised warm-start updates using training graphs
only, then PPO on real frozen-skill rollouts. Warm-start uses synthetic progress
states and is not a replacement for simulator validation. No held-out task
graph is used for warm-start or checkpoint selection.

## Checks before expensive training

`check` loads the actual model, resets the real scene, executes ten frozen-policy
frames, checks finite observations/actions, and verifies unchanged base weights.
It does not optimize any policy. Ten frames do not establish task success.

`preflight --split train` runs a scripted selector through every training recipe.
Each recipe has a per-task success threshold, so easy successes cannot hide a
broken skill chain. `train` repeats this check before optimization. By default,
only chains with at least 80% success across the three qualification seeds enter
training. The full report and excluded tasks are saved in `qualified_tasks.json`;
at least two qualified tasks, including a multi-skill chain, are required. Warm-start and checkpoint validation
use the same qualified training set. Held-out tasks are not filtered to hide
failures. Set `qualify_training_tasks` to false for strict all-task gating.
Three seeds are still a small reliability sample; increase `preflight_seeds`
before a research run.

`--allow-weak-base` explicitly uses the unqualified chains for diagnostics and is
recorded in the run status. It cannot fix a broken low-level controller. Do not
interpret such a run's failures as evidence against the composition head alone.

To inspect one physical chain:

```bash
python -m experiments.run baseline --split train --task drawer_cycle --seeds 9000 --video
```

## Training and evaluation records

`total_timesteps` counts head selections. Each valid selection can execute up
to 360 simulation frames, so 4096 head steps can be much slower than 4096 physics
steps. PPO can round the requested budget up to a complete rollout batch; the
actual count is saved. A small MLP does not necessarily train faster on CUDA;
image-policy inference and rendering dominate many of these runs.

Every run saves configs, task/catalog snapshots, checkpoint/scene/statistics/
embedding hashes, dependency versions, a frozen-parameter digest, preflight
results, Monitor CSV, PPO progress CSV, periodic head checkpoints, a final head,
and the best head selected using training-task validation seeds only. A separate
validation process prevents evaluation from resetting or changing the training
scene. This also avoids native CALVIN calls that implicitly address physics
client 0. Its RNG changes are isolated from the training RNG. The base model's parameters are
excluded from the head optimizer and checked before and after optimization.

`evaluate` defaults to held-out compositions and the best saved head. The
warm-started head is included in best-checkpoint selection, so PPO is not assumed
to improve it. The final head is a fallback if a best checkpoint is missing. It loads the saved
configs and checks the environment contract. CPU/CUDA selection may change;
skill order, policy assets, renderer, and observation/reward definitions may not.
Initial and final heads can also be compared with `--checkpoint initial` and
`--checkpoint final`. Results include per-task success, valid-selection rate, conditional frozen-skill
success, low-level frame counts, failure reasons, and a full decision trace.

Compare against both baselines on the same split and seeds:

```bash
python -m experiments.run baseline --split test
python -m experiments.run evaluate --policy random --split test
python -m experiments.run evaluate --split test --video
```

To continue a run on either machine after copying its output directory:

```bash
python -m experiments.run train --resume outputs/composition/RUN --timesteps 4096
```

This restores the head and optimizer into a new run directory. It starts fresh
episodes/RNG, so continuation is not bit-identical to an uninterrupted run.
Periodic checkpoints can resume an interrupted run. Failed runs are never
silently selected as completed evaluation models.

## Tests

```bash
.venv-calvin/bin/python -m pytest
```

The unit suite checks Gymnasium termination/truncation, scene continuity,
selection penalties, failure propagation, persistent final conditions,
train/test separation, RNG preservation, frozen-weight mutation detection, the
training gate, head checkpoint round trips, device selection, and portability.
The tiny head-training test uses a fake simulator and never updates the released
robot controller. Real simulator checks and frozen-skill baselines are separate
from these unit tests. No test result guarantees successful policy learning.

Initial software-rendered checks exposed low-level failures and sensitivity to
episode history. Episode resets now reload the native world to clear physics
and controller caches. Skills within one episode still share a continuous scene.
Always use the current reset protocol's preflight results for qualification.
These diagnostics are not a reproduction of the official model's reported score.

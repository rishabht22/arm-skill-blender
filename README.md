# RoboFactory Skill Composition

This repository also contains a separate [goal-conditioned workbench experiment](projects/goal_composition/README.md), available through `python -m skill_composer compose`. It uses its own RoboFactory recordings, data directory, and checkpoints.

A multi-task imitation-learning coordinator for two Panda robot arms in MuJoCo.
It trains on **published RoboFactory recordings only** and executes new supplied
skill graphs with **up to 17 operations**, using reusable frozen manipulation
skills. No generated expert demonstrations or RL training are used.

## Watch it

The published data, prepared training set, trained policy, and previews are
already present in this checkout.

```bash
cd /Users/risha/Github/arm-skill-blender
source .venv-mujoco/bin/activate

# New task: transfer parts between arms, open a container, build a tower inside.
mjpython -m skill_composer rollout --task relay_container --parts 3 --viewer

# Longer 17-operation version.
mjpython -m skill_composer rollout --task relay_container --parts 4 --viewer

# Pipelined handoff + stacking, with both arms working in parallel.
mjpython -m skill_composer rollout --task relay_stack --parts 4 --viewer

# Each arm's frozen pick, place, stack, and open-lid skills, in isolation.
mjpython -m skill_composer skills --arm 0 --viewer
mjpython -m skill_composer skills --arm 1 --viewer
mjpython -m skill_composer skills --arm 1 --skill stack --viewer
```

Use `python` instead of `mjpython` for the viewer on Linux. Physics-only execution
omits `--viewer`. A graphics context is required for windows and GIFs.

```bash
# Pre-rendered previews (macOS).
open outputs/cell/individual_skills.gif
open outputs/cell/relay_container.gif
open outputs/cell/relay_stack.gif
open outputs/cell/coordination_comparison.gif

# Regenerate all galleries from actual evaluation episodes.
python -m skill_composer demo

# Render a new rollout; the movie is accelerated, while its HUD shows sim time.
python -m skill_composer rollout --task relay_container --parts 4 \
  --gif outputs/cell/relay_container.gif

# Original public recording images, clearly separate from our MuJoCo scene.
python -m skill_composer preview --task PassShoe --episode 0
open outputs/cell/published_PassShoe.gif
```

## Training data and learning

The training bank contains **750 original published episodes / 169,215 frames**
from five RoboFactory tasks:

| Published task | Episodes | Reusable behavior |
| --- | ---: | --- |
| PickMeat | 150 | Reach, grasp, lift |
| StackCube | 150 | Pick and place on a support |
| PassShoe | 150 | Cross-workspace staging and receiving |
| PlaceFood | 150 | Parallel object pickup and lid opening |
| TwoRobotsStackCube | 150 | Parallel pickup and ordered stacking |

Of these episodes, **525 train / 110 validate / 115 test**. Whole trajectories
stay in one partition. The original files total about **5.6 GB**, verified against
pinned upstream SHA256 hashes. [Provenance and annotation details](data/DATASET.md).

The coordinator is a shared permutation-equivariant neural actor. Each candidate
operation has its type, assigned arm, execution status, dependency progress,
target properties, and both arms' measured joint positions/velocities. The actor
scores pending operations and WAIT for each arm, with shared weights across
objects, candidate order, and arm roles. It has no task-name or node-index input.

Supervision comes from **weak semantic annotations of the published joint/gripper
commands**, informed by the official source task programs. Pick includes the
post-grasp lift; place includes release and clearance/retreat where the source
expert performs it. Goal graphs and dependencies are hand annotated. These are
not publisher-provided skill labels or automatically discovered plans.

The 54-dimensional candidate representation includes actual recorded
proprioception. Candidate shuffling, arm exchange, and proprioception dropout
augment those same recordings; they do not create or count extra demonstrations.
Training uses categorical behavior cloning with AdamW. Checkpoint selection uses
held-out source episodes and checks performance both with and without motor-state
inputs. The selected checkpoint matches all **289 held-out annotated decisions**;
these simple source graph templates make that an infrastructure measure, not a
physical success rate or evidence of broad general intelligence.

```bash
# Fresh environment: Python 3.10+; 3.12 tested.
python3 -m venv .venv-mujoco
source .venv-mujoco/bin/activate
python -m pip install -e '.[dev]'

# Download or verify the five ORIGINAL published task files.
python -m skill_composer download
python -m skill_composer prepare
python -m skill_composer train --epochs 200
python -m skill_composer evaluate
```

## What is new about the evaluation tasks

The training programs have at most four operations. The two evaluation families
combine those known skills into longer graphs, with two to four parts:

- **Relay stack:** supplier picks parts and stages them in separate shared docks;
  receiver fetches staged parts and builds a tower. Supplier and receiver can
  overlap work. Three parts require 12 operations; four require 16.
- **Relay container:** the supplier first stages all required parts, then opens
  and holds the container lid. The receiver can prefetch a part, wait for access,
  and stack the parts inside. Three parts require 13 operations; four require 17.

These full graphs are excluded from supervised training and checkpoint
selection. They are **new compositions of known operations**, supplied as goal
programs. Dependencies and arm roles are inputs. The head learns dispatch and
waiting; it does not infer a plan from a new natural-language instruction or
invent new physical skills. Evaluation scenes and controllers were developed
using this task family, so this is a structural transfer demonstration rather
than a preregistered unseen-benchmark claim.

Only executor availability, arm ownership, and completed/running status mask
candidate actions. **Unmet graph dependencies are observations, not policy
masks.** A bad dispatch is rejected by the executor and counted; there is no
expert scheduler fallback for the learned controller.

## Fair comparisons

```bash
# Same scenes, objects, seeds, and frozen skills for every controller.
python -m skill_composer compare --seeds 20 --seed-start 3000 --parts 3

# Individual-controller inspection.
mjpython -m skill_composer rollout --task relay_stack --parts 3 \
  --controller serial --viewer
mjpython -m skill_composer rollout --task relay_stack --parts 3 \
  --controller greedy --viewer

# Either arm alone on the shared task (the other arm stays parked).
mjpython -m skill_composer rollout --task relay_stack --controller arm0 --viewer
```

The report compares the learned head with a serial coordinator and a strong
hand-written greedy graph coordinator. Serial performs one operation at a time;
greedy dispatches ready operations for both arms. Every controller uses identical
IK skills, grasp models, placements, and task criteria.

Physical success requires every operation to finish, no tracking failure, and
all final objects within 2.5 cm of their target centers. **Strict success** also
requires zero rejected dispatches and zero sampled arm-contact frames. Both are
reported; tower instability and contacts are not hidden. Timings are simulated
completion times on successful episodes. Performance against serial demonstrates
parallel execution, while the greedy control tests whether learning adds an
advantage. This version does not establish superiority over that strong control.

## Measured results

Final three-part evaluation: **20 fresh seeds per task and controller**
(seeds 3000–3019). This report contains 300 physical episodes across five tasks
and three controllers. These figures use the same frozen controller and goal
criteria for all comparisons.

| New composition | Controller | Physical success | Strict success | Mean sim time |
| --- | --- | ---: | ---: | ---: |
| relay_container | learned | 100% | 100% | 32.67 s |
| relay_container | serial | 100% | 55% | 35.61 s |
| relay_container | greedy | 100% | 100% | 32.48 s |
| relay_stack | learned | 100% | 80% | 23.58 s |
| relay_stack | serial | 100% | 55% | 33.11 s |
| relay_stack | greedy | 100% | 95% | 22.89 s |

The learned coordinator reduces relay-stack completion time by **28.8%**
versus serial execution. The dependency-aware greedy coordinator is slightly
faster and has a higher strict success rate on that task. This is a demonstration
of learned skill recomposition and parallel dispatch, not an improvement over
hand-written graph scheduling. Four-part variants are reported separately in
`outputs/cell/variant_evaluation_parts4.json` and remain more demanding.

## Skills and simulator limits

Each frozen skill is an object/target-conditioned closed-loop controller with
Cartesian IK waypoints, gripper control, measured joint-target convergence,
termination, and failure reporting. `stack` is the place skill with a support
height target. The arm controllers are scripted, not separately pretrained neural
policies. Only the composition actor is optimized.

Props are dynamic MuJoCo bodies under gravity and collide with the workbench,
fixtures, and other props. **Grasping uses proximity-gated idealized welds**, not
validated friction contacts. Robot-to-prop contact is simplified, joint-space
interpolation is not a complete collision planner, and brief arm contacts can
still occur. This is a procedural MuJoCo manipulation cell, not a validated port
of RoboFactory's original ManiSkill environments or a sim-to-real result. No
NVIDIA SimReady objects are included.

## Important files

| Path | Contents |
| --- | --- |
| `src/skill_composer/cell/dataset.py` | Verified published downloads, temporal annotations, source splits |
| `src/skill_composer/cell/program.py` | Shared goal-program and candidate observation contract |
| `src/skill_composer/cell/policy.py` | Shared actor, behavior cloning, held-out evaluation |
| `src/skill_composer/cell/world.py` | Dynamic cell, frozen IK executors, physical preconditions |
| `src/skill_composer/cell/experiment.py` | Viewer, rendering, matched closed-loop evaluation |
| `src/skill_composer/robot.py` | Licensed Panda scene construction |
| `data/robofactory/` | Full original five-task HDF5/JSON files and checksum manifest |
| `data/processed/multitask.*` | Training observations, annotations, and source episode audit |
| `outputs/cell/policy.pt` | Trained composition actor |
| `outputs/cell/training.json` | Training loss and source validation history |
| `outputs/cell/imitation_evaluation.json` | Original-recording held-out imitation metrics |
| `outputs/cell/closed_loop_evaluation.json` | Matched physical results, strict metrics, and dispatch traces |
| `outputs/cell/*.gif` | Individual skills, new compositions, and original-data preview |

Data, trained outputs, and the Python environment remain local and Git-ignored.
The downloader and training commands reproduce them. The committed v1 lift
baseline remains accessible via `python -m skill_composer baseline ...`; it is
not used to train the new multi-task coordinator.

```bash
python -m pytest -q
```

Sources: [RoboFactory paper](https://arxiv.org/abs/2503.16408),
[official task planners](https://github.com/MARS-EAI/RoboFactory/tree/main/robofactory/planner/solutions),
[published dataset](https://huggingface.co/datasets/FACEONG/RoboFactory_Dataset),
[MuJoCo Menagerie Panda](https://github.com/google-deepmind/mujoco_menagerie/tree/98a70c569c4c1b6da467a20012f766ee1f8dc521/franka_emika_panda).
The Panda model license and modifications are recorded in `assets/panda/`.

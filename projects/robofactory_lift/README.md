# Arm Skill Composer

A minimal hierarchical imitation-learning experiment using **150 published
RoboFactory LiftBarrier demonstrations**. A shared neural head selects the next
skill for two arms; the individual arm executors stay frozen.

```text
Published demonstrations → skill annotations → behavior cloning → composition head
                                                                     ↓
Current arm states + skill completion signals → next skill for each arm → MuJoCo
```

## Run

Set up the shared editable environment using the [repository README](../../README.md).
From this project folder, use the bundled checkpoint:

```bash
# Watch the trained head coordinate the two arms (macOS).
mjpython -m skill_composer baseline rollout --run outputs/demo --viewer

# Inspect either arm's individual frozen skills, at one third speed.
mjpython -m skill_composer baseline skills --arm 0 --run outputs/demo --viewer
mjpython -m skill_composer baseline skills --arm 1 --run outputs/demo --viewer

# Play the saved comparison video on macOS.
open outputs/demo/preview.mp4
```

On Linux use `python` for the viewer. Physics-only runs omit `--viewer` and
`--gif`; rendering requires a working graphics context. `open` is macOS-specific.

## Data and training

```bash
# Fetch or verify the original public files (~988 MB).
python -m skill_composer baseline download

# Infer skill boundaries and split whole trajectories.
python -m skill_composer baseline prepare

# Train only the composition head, then evaluate the held-out episodes.
python -m skill_composer baseline train --epochs 100
python -m skill_composer baseline evaluate
```

The source contains 15,077 frames. Offline heuristics annotate
**approach → grasp → lift → hold**, producing 673 decision events. The head acts
when either executor reports completion; ongoing skills are non-preemptive.
Its 48 inputs are both arms' measured joint positions/velocities, active skill
IDs, and completion flags. A shared two-layer MLP outputs four skill logits per
arm. Training uses class-weighted cross-entropy and AdamW.

Whole episodes are split into **105 training / 22 validation / 23 test**.
Normalization and frozen executor target calibration use only training episodes;
checkpoint selection uses validation episodes. Recorded actions create offline
labels, but actions, future observations, episode IDs, clocks, and success flags
are not head inputs. Online skill termination is scripted, not learned.

The saved head achieves **95.1% joint next-skill accuracy** on held-out annotated
events. A fixed-sequence baseline scores **100%**. These are weak-label imitation
scores, not physical task success rates. [Data provenance and annotation rules](data/DATASET.md).

## Compare individual skills and coordination

```bash
# Nominal scene: six controllers use identical frozen skills and initial state.
python -m skill_composer baseline compare --run outputs/demo --output outputs/baseline/comparison \
  --gif outputs/baseline/comparison/comparison.gif
open outputs/baseline/comparison/individual_skills.gif
open outputs/baseline/comparison/comparison.gif

# Apply the same 1.5-second arm-1 motor-start delay in every trial.
python -m skill_composer baseline compare --run outputs/demo --delay-arm1 1.5 \
  --output outputs/baseline/comparison_delayed \
  --gif outputs/baseline/comparison_delayed/comparison.gif
open outputs/baseline/comparison_delayed/comparison.gif
```

Omit `--gif` to print results and save traces without rendering.

| Controller | Nominal | Arm 1 delayed 1.5 s |
| --- | --- | --- |
| Either arm alone | Fails shared task | Fails shared task |
| Independent local skill queues | Passes | Fails |
| Learned head without guard | Passes | Fails |
| Learned head with temporal guard | Passes | Passes |
| Fixed sequence with the same guard | Passes | Passes |

Each solo arm completes its local skills, but cannot lift the bar level alone.
The shared goal requires both grasps, a height gain above 0.15 m, handle height
difference below 0.04 m, and no sampled arm-arm contacts for at least 0.5 seconds.
The temporal guard blocks lift/hold until both arms grasp the bar. The fixed
coordinator also succeeds, so the delayed-case improvement comes from this
explicit synchronization rule; a benefit from learning has not been established.
These are deterministic development examples, not aggregate benchmark results.

For a single episode, `rollout --no-guard` runs the raw head and `rollout --oracle`
runs the fixed coordinator. `--randomization 0.01 --seed 3` adds a small object
position perturbation. All rollouts, skill inspections, and comparisons share
one execution implementation.

## Files

| Path | Purpose |
| --- | --- |
| `src/skill_composer/data.py`, `download.py` | Published data, annotations, episode splits |
| `src/skill_composer/model.py`, `train.py` | Composition head, behavior cloning, evaluation |
| `src/skill_composer/skills.py`, `sim.py` | Frozen skill interface and MuJoCo physics |
| `src/skill_composer/execution.py`, `comparison.py` | Shared execution, viewer, controlled comparisons |
| `src/skill_composer/preview.py` | Real-demo prediction preview |
| `data/raw/` | Full original HDF5/JSON files and checksum manifest |
| `data/processed/` | Training arrays, annotation audit, frozen target calibration |
| `outputs/baseline/` | Checkpoint, frozen skills, training/evaluation records, demo GIF |
| `outputs/baseline/comparison*/` | Comparison traces and representative animations |
| `assets/panda/` | Necessary robot model, collision meshes, license, provenance |
| `tests/` | Data alignment, split isolation, executor and comparison checks |

Raw data, processed arrays, outputs, and the Python environment are kept locally
and ignored by Git. The commands above recreate the data and training artifacts. The small demo
checkpoint and calibration in `outputs/demo/` are included.
`head.pt` and `skills.json` are required for execution; checksums prevent loading
a mismatched frozen skill library. `rollout.json` records the latest standalone
episode. Each comparison stores its complete trace in `results.json`.

```bash
python -m pytest -q
```

## Limits and sources

This is a **separate MuJoCo proxy**, not a port of RoboFactory's ManiSkill
benchmark. The frozen scripts use median training-demo endpoint configurations;
they are replaceable placeholders, not independently pretrained grasp policies.
The barrier and supports are calibrated to those targets. Proximity-gated point
attachments approximate grasps; this does not establish friction-based grasping,
compositional generalization, or sim-to-real transfer. A branching task and
stronger object-conditioned skills are needed to test a learned coordination
advantage. NVIDIA SimReady objects and a USD importer are not included.

- [RoboFactory paper](https://arxiv.org/abs/2503.16408),
  [official implementation](https://github.com/MARS-EAI/RoboFactory),
  [published LiftBarrier data](https://huggingface.co/datasets/FACEONG/RoboFactory_Dataset/tree/main/LiftBarrier) (publisher marks MIT).
- [MuJoCo Menagerie Panda](https://github.com/google-deepmind/mujoco_menagerie/tree/98a70c569c4c1b6da467a20012f766ee1f8dc521/franka_emika_panda)
  (Apache 2.0). Model changes and licensing are recorded in `assets/panda/` and `NOTICE`.

# Single-arm skill composition

Can a model learn which reusable skill to call, and on which object or target,
from demonstrations of whole tasks? Start with one Panda arm in robosuite and
use simulator state for object poses, gripper state, and goal checks.

## Multi-scene Panda workshop

The workshop has 35 callable primitives and 15 scripted task sequences across
three single-Panda workcells. Each primitive issues one gripper command or
follows one continuous Cartesian path. The scripted tasks sequence those
calls, and the isolated trials start from a prepared state and record only
the selected call for playback.

| Workcell | Objects and fixtures | Example primitives |
| --- | --- | --- |
| Tabletop | Three graspable blocks, a tower, a roller, a puck, a barrier, and placement stations | `yaw_in_hand`, `topple`, `roll_cylinder`, `slide_in_channel`, `clear_obstacle` |
| Fixtures | Drawer, button, slider, latch, dial, gate, and toggle | `pull_drawer`, `press_button`, `lift_latch`, `swing_gate` |
| Assembly | Keyed peg and socket, rail-guided key, press-fit pin, and a captive screw plug | `insert_peg`, `extract_peg`, `slide_key_in`, `twist_in_socket` |

Use the existing `.venv-single-arm` from the repository root, or create a
Python 3.12 environment and install
[`individual_skill_generation/requirements.txt`](individual_skill_generation/requirements.txt).
From the repository root with the environment active:

```sh
cd projects/single_arm_composition
python -m workshop list
python -m workshop skill insert_peg
mjpython -m workshop skill insert_peg --live
mjpython -m workshop task sort_three --live
python -m workshop check
```

Use `mjpython` for the viewer on macOS and `python` on Linux. The skill
command prints the number of control steps and whether the physical
completion check passed. Use `--live` to watch the controller step
through setup and execution, or `--view` to replay the resulting clip.
Use `--seed` to vary the reset and `--speed 0.5` to slow the viewer.
The task command prints each call and verifies the final goal. The check
command exercises the entire catalog and exits with a failure status if
any run fails. Use `mjpython -m workshop check --live` to watch the full
suite, with one window per workcell.

### Generate composition data

```sh
python -m workshop generate --trials-per-skill 2 --episodes-per-task 2
```

This writes `data/raw/panda_workshop.h5`. Use `--out` to write another
file. Standalone trials cover all 35 primitives and go in `trials`.
Full-task episodes go in `train`, `validation`, and `recombination`.
The recombination split holds out `sort_three` (new object-to-station
pairings), `stack_blue_on_red` (a placement that depends on a previous
placement), `dial_latch_drawer` (a new sequence of fixture goals), and
`fit_key_and_peg` (both assembly goals in one episode). The fixture
layouts are fixed, with seeded variation in the Panda reset and free
object positions.

Each episode includes its reset seed, setup actions, and success or
failure. For full tasks, `goal` stores object placement or fixture
conditions sorted by object name. Each ordered call stores the primitive
ID, arguments, and outcome, plus actions, simulator states, and
observations. Row zero is the observation before the call. The
`scenes` group stores the object and joint names for each observation
array. For a future selector, use the first observation of each call
with the episode goal and supervise `STOP` from the final observation of
a successful full task. The CLI generates scripted rollouts only.

### Train the selector

From the repository root, use the PyTorch environment for the training
command. Generate `data/raw/panda_workshop.h5` first if it does not exist.

```sh
cd projects/single_arm_composition
../../.venv-mujoco/bin/python -m workshop.train --epochs 100
```

The command fits one small behavior-cloning model per workcell and saves
`outputs/workshop_selector.pt`. It uses the validation split to select a
checkpoint and reports next-call accuracy on the held-out recombination
split. The reported accuracy uses recorded observations at skill boundaries.
For a learning curve, use `--demos-per-task 2`, `4`, or `8`, with a
distinct `--out` path for each run. Add `--dry-run` to check data and label
coverage before fitting.

### Evaluate full rollouts

With the workshop simulator dependencies and PyTorch in the active
environment, run this from the single-arm project directory:

```sh
python -m workshop.evaluate --episodes 5 --seed 1001 --trace
```

The evaluator defaults to the four held-out tasks. At each skill
boundary it passes the current simulator observation and task goal to
the selector, then executes the predicted call. A successful rollout
requires the selector to issue `STOP` after satisfying the goal. The
scripted expert runs from the same seeded reset for comparison.

On macOS, with this repository's separate simulator and training
environments, run one held-out test live from the single-arm project
directory:

```sh
PYTHONPATH="../../.venv-single-arm/lib/python3.12/site-packages:../../.venv-mujoco/lib/python3.12/site-packages" \
  ../../.venv-single-arm/bin/mjpython -m workshop.evaluate --task fit_key_and_peg --episodes 1 --live
```

The window updates during execution and the terminal prints each
selected primitive before it runs. Close the window to stop the test.
Use `--view` in place of `--live` to replay a single finished rollout.

## Watch the original single-cube skills

Run this from the repository root with the existing macOS environment:

```sh
(cd projects/single_arm_composition && ../../.venv-single-arm/bin/mjpython -m individual_skill_generation.viewer)
```

One MuJoCo window shows an isolated clip for each primitive. It jumps to
the start of each call and prints the primitive name in the terminal.
It uses the composition dataset by default and can also play examples
from the standalone dataset. Use `--speed 0.5` for half-speed playback. On Linux, use `python`
from the activated skill environment in place of `mjpython`.

To play just one example, pass its skill name:

```sh
(cd projects/single_arm_composition && ../../.venv-single-arm/bin/mjpython -m individual_skill_generation.viewer --skill push_along)
```

Use `--help` to list the supported skill names. The clip starts at the saved
state for that call, so a push begins with the hand already behind the cube.

## Generate composition demonstrations

```sh
# From projects/single_arm_composition, with the skill environment active.
python -m composition.generate
```

The expert uses the following vocabulary. Each motion holds the current
gripper command and sets one pose target. Gripper calls hold the arm pose.

| Primitive | Command |
| --- | --- |
| `hover_over` | Move above the cube or the requested destination |
| `descend_to_grasp` | Descend to the cube with the jaws open |
| `raise_by` | Move vertically up by a requested distance |
| `lower_to` | Move vertically down to a requested height |
| `move_behind` | Move above the staging point opposite the push direction |
| `descend_behind` | Descend to the height used for pushing |
| `push_along` | Push along the requested tabletop axis direction |
| `open_gripper` | Open the jaws |
| `close_gripper` | Close the jaws |

The goal and initial state change the sequence. A lift can start with an
open gripper, a closed gripper, or an already-grasped cube. A push can start
with the hand already staged behind the cube. A place includes moving to the
destination, lowering, releasing, and clearing the cube. Both push directions
and both sides of the placement workspace are sampled.

| Split | Tasks | Episodes attempted by default |
| --- | --- | ---: |
| Train | Lift, place, push | 180 |
| Validation | Lift, place, push | 45 |
| Test | Lift, place at farther targets, push | 60 |
| Recombination | Push, then pick up the same cube and place it | 20 |

`data/raw/panda_composition.h5` stores split/task/episode groups. Each episode
contains a numerical goal, reset seed, setup actions, ordered skill calls,
and overall success. Each call includes its target pose and observations,
actions, and states. The first observation of each call is the selector input
at that decision. The final observation of a successful episode can be
labeled STOP. Use complete successful episodes for imitation and keep the
failure records for diagnosis. Setup actions, expert sequence, and success
flags are dataset metadata. The farther placement band and push-then-place
episodes are reserved for evaluation.

This dataset uses one cube, simulator state, and pushes along either direction
of one tabletop axis. A learned selector has yet to be trained and evaluated.

## Generate individual skill trials

From this directory, using Python 3.12:

```sh
python3.12 -m venv .venv-skills
source .venv-skills/bin/activate
python -m pip install -r individual_skill_generation/requirements.txt
python -m individual_skill_generation.generate
python -m individual_skill_generation.replay data/raw/panda_semantic_skills.h5 grasp/00000 carry/00000
```

Generation is headless. It attempts 100 trials per context by default. Use
`--contexts grasp carry` to select contexts, `--per-context` to change trial
counts, `--seed` to change resets, and `--out` to write a fresh file. The
generator refuses to overwrite existing datasets.

Standalone trials use the same nine primitives as the composition expert.
Each trial varies the scene, goal arguments, or starting condition for one
call. For example, the carry context records `hover_over` with a placement
destination and a cube already held. Any actions used to create that
starting state are stored separately. Moves
check pose convergence. Grasp checks opposing finger contact. Carrying checks
that the cube stays in the grasp and moves with the hand. Release checks that
the cube returns to table height, and push checks displacement in the commanded
direction. The sampler varies reset poses, cube yaw, and target poses within a
small workspace. The model seed fixes the cube geometry within a dataset.

### Dataset format

Each `trials/<context>/<index>` HDF5 group contains:

- Attributes for the skill ID, seed, target pose, success, and completion options.
- `actions` with one row per 20 Hz control step. The seven components are world XYZ in meters, rotation vector in radians, and gripper command (−1 open, +1 close).
- `observations` and `states` with one more row than `actions`. Row zero is the state before the skill, and row `i + 1` follows action `i`. Observations include arm joints, finger positions, end-effector pose, cube pose, and grasp contact.
- `reset_state` and `setup_actions` for replay. Setup actions are stored separately from the skill demonstration.

The file records environment versions, model XML, and per-context outcome
counts. Skill failures are recorded as trajectories with `success=false`.
Setup failures are counted with their seeds and reasons because no skill call
ran. Raw datasets are local and ignored by Git.

These trials cover free-space motion and simple tabletop interactions with one
cube. Doors, insertion, tool use, and cluttered motion would need their own
fixtures and completion checks.

## Where to get the individual skills

| Source | What it provides | Fit for this project |
| --- | --- | --- |
| [RoboFactory code](https://github.com/MARS-EAI/RoboFactory) and [published data](https://huggingface.co/datasets/FACEONG/RoboFactory_Dataset/tree/main) | Scripted ManiSkill task generation using motion primitives, with recordings organized by task. `PickMeat` and `StackCube` are useful reference trajectories. | The published task folders do not provide a standalone open/close/reach dataset. Extracting segments would require labeling and a simulator/action adapter for robosuite. The [existing repo audit](../robofactory_cell/data/DATASET.md) documents these recordings and its inferred labels. |
| [robosuite controllers](https://robosuite.ai/docs/modules/controllers.html) and [PickPlaceCan example](https://robosuite.ai/docs/_sources/demos) | A simulated arm, Cartesian end-effector control, gripper control, and a single-object pick-and-place environment. | Implement small closed-loop scripts and record their rollouts directly. This gives both the skills and exact call boundaries in the same simulator. |

Motion primitives resolve their own controller targets. For example,
`hover_over` uses the cube position or the task's destination plus clearance.
`move_behind` uses the cube position and the requested push direction.
The selector can choose among these named operations with task-relative
arguments. The controller handles the Cartesian pose internally.

Collect standalone trials across valid initial arm poses, gripper states, and
target poses. Use the scripts as frozen executors for the first experiment.
Their observations and actions can also become training data if learned
low-level skills are needed later.

## Where to get whole-task demonstrations

| Source | What to use | Label granularity |
| --- | --- | --- |
| Scripted rollouts in robosuite | Build a pick-and-place expert by calling the task primitives. Record labels as it executes. Then add a scene with two objects and alternate which object and destination the goal requests. | Exact primitive IDs, arguments, and timing from the script. |
| [robomimic Can PH](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html) | 200 proficient-human pick-and-place episodes. Start with the low-dimensional release for an independent check on segmentation. | Map gripper transitions and motion phases to the project vocabulary. Audit labels against replay and check the recorded environment and controller versions first. |
| [MimicGen releases](https://mimicgen.github.io/docs/datasets/mimicgen_corl_2023.html) and [subtask annotations](https://mimicgen.github.io/docs/tutorials/subtask_termination_signals.html) | Inspect the StackThree source demos for a longer sequence with two grasps and two placements. Generated datasets are an option for more scene variation. | The source annotation scheme marks object-centric subtask ends. A grasp segment can contain approach and closure. Inspect each release's HDF5 fields and derive finer labels if needed. |

Keep standalone skill trials and full-task episodes separate. In full-task
episodes, log the goal, reset seed, state at each decision, skill ID and
arguments, start/end steps, outcome, and final task success. Save per-step
observations and actions for replay. Record moves relative to named objects,
destination regions, or the current end-effector pose so they can be resolved
against the live state at execution time.

## First experiment

1. Execute each primitive on its own, then compose them by hand to move one
   can into a bin. Verify a physical grasp and that the released can remains
   in the bin after settling.
2. Use that rollout to verify the recorder. For selector training, add goals
   that require different decisions. For example, use the same scene to ask
   for different object/bin pairs, and include a verified pre-grasped reset so
   the expert can begin with transfer. A fixed sequence from one reset type
   mainly tests whether a model can memorize the order.
3. At each skill boundary, train a small selector to predict the next skill
   and its target (or `STOP`) from the current state and goal. Keep the low-level
   scripts frozen. Split by whole episode and reserve some object/destination
   pairings for a separate recombination test.
4. Compare closed-loop task success with a fixed sequence and the scripted
   state-aware expert, using the same skill executors. Report decision accuracy
   separately from physical success, invalid calls, and timeouts.

The next milestone is a learned selector evaluated in closed-loop rollouts
against the scripted expert and a fixed sequence.

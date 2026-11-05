# Two robots with released manipulation experts

The first experiment uses the released pushing and lifting networks from
[Spatial Intention Maps for Multi-Agent Mobile Manipulation](https://github.com/jimmyyhwu/spatial-intention-maps)
(ICRA 2021). This matches our revised starting constraint: one different,
pretrained manipulation expert per robot, with a new coordination head above
them. It does not yet give each robot multiple independent experts.

## What is available

| Source | Available starting material | Fit for this project |
| --- | --- | --- |
| [Spatial Intention Maps](https://github.com/jimmyyhwu/spatial-intention-maps) | Released single-agent pushing and lifting checkpoints; heterogeneous-team checkpoints also advertised | Best initial fit: both experts use one simulator and the same four-channel map format |
| [Diffusion Policy / Push-T](https://github.com/real-stanford/diffusion_policy#-evaluate-pre-trained-checkpoints) | Public demonstrations and pretrained pushing checkpoints | Strong single-robot manipulation reference; does not supply a complementary robot skill library |
| [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) | 2D physics; transport, ball-passage, navigation and other scenarios | Good future training backend; a compatible released multi-skill expert bank was not verified |
| [BabyAI / Minari](https://minari.farama.org/datasets/minigrid/BabyAI-UnlockPickup/optimal-v0/) | Optimal unlock/pickup demonstration datasets | Useful discrete composition reference, with different manipulation mechanics |
| [RoboFactory](https://github.com/MARS-EAI/RoboFactory) | Multi-robot manipulation benchmark and data-generation tooling | Relevant later; not the lightweight 2D starting point |

An additional composition reference is
[PACTS](https://planpacts.github.io/), which includes PushBarrier. Its project
page did not provide a verified downloadable policy release in this search.

The selected source uses 2D overhead maps and planar mobile navigation in
PyBullet, with lifting modeled in 3D. It is a tabletop-style, 2.5D environment.

## Runnable starting task

Two mobile robots collect yellow cubes into a red corner receptacle in a
1 m × 0.5 m room. Robot 0 can call only the pushing network; robot 1 can call
only the lifting network. Their expert catalogs are disjoint. This is an API
restriction, not a claim that the lifting robot cannot incidentally push through
contact.

The smoke test has one cube. Training and the small comparison suite use two.
Start with the source geometry and target so that changed geometry and new
goal conditioning do not obscure policy compatibility.

A centralized head chooses **push**, **lift**, or **both**. A decision dispatches
the selected experts' greedy native spatial actions, then runs physics until
those actions finish. The other robot remains physically present. Both experts
retain their own partial-observation maps. The head sees simulator object and
robot positions, heading, carrying state, progress and remaining decision budget.

There are no learned action mixtures, rewritten expert targets, task-solving
teacher actions, or online expert updates. This is an initial dispatch-learning
problem; it is not yet composition over several skills on each robot.

## Run

On this machine the dependencies, source, pathfinding extension and weights
have already been installed. From the repository root:

```bash
.venv-mujoco/bin/python -m experiments.planar --episodes 1 --seed 0 --cubes 1 --max-decisions 20 --gif outputs/planar/demo.gif --output outputs/planar/demo.json

.venv-mujoco/bin/python -m experiments.planar --train --train-episodes 64 --cubes 2 --max-decisions 40 --output outputs/planar/pilot/training.json

.venv-mujoco/bin/python -m experiments.planar --policy suite --checkpoint outputs/planar/pilot/head.pt --episodes 5 --cubes 2 --max-decisions 40 --output outputs/planar/pilot_comparison.json
```

To set up another machine, first install the project's base requirements, then:

```bash
# The forced stdio include fixes an old bundled-zlib macro collision on new macOS SDKs.
CFLAGS='-include stdio.h' .venv-mujoco/bin/python -m pip install -r requirements-planar.txt
.venv-mujoco/bin/python -m experiments.planar_setup
```

The upstream software originally targeted Python 3.7 and PyTorch 1.2 on Ubuntu.
This adapter was exercised on this Mac with Python 3.12, PyTorch 2.15, and
PyBullet 3.2.7. Other platforms need their own rollout check. The generated
pathfinding extension is specific to the local Python/platform.

## Freezing, training and measurement

The released networks produce dense Q maps over native spatial actions. The
source's movement controller executes those commands. This is not a neural
joint-action controller. The native lifting implementation uses a fixed
attachment and pose adjustments during pickup/drop; the experiment inherits
that abstraction.

Weights are loaded strictly, in eval mode, with gradients disabled. Their
SHA-256 values and the source files are checked. Parameter and buffer hashes
are rechecked after training batches and evaluations. The optimizer owns only
the 1,764-parameter coordination head.

The head uses PPO with a critic, clipped policy updates, an entropy term, and
discounting by elapsed simulation time. Reward combines delivered objects,
distance progress, execution time and a small decision cost. A dispatch that
makes no progress still consumes a decision. The 40-decision horizon limits
stalls. The first 64-episode run is a pipeline pilot, not a tuned result.

Training uses seeds 0–63. Comparisons use seeds 20000–20004, with identical
initial states across policies. These five test seeds are now development
evidence; use fresh seeds for any subsequent tuned model's final test. The
checkpoint is the final training checkpoint, not selected against test scores.

Compare against pushing alone, lifting alone, both experts always active, and
alternating dispatch. Retain successes, objects delivered, simulated execution
time, decision counts, collision events, recovery resets and complete dispatch
traces. Compare speed alongside success; fast failed attempts are not efficient
solutions. Simulator wall time is separate from robot simulation time.

Native out-of-bounds object recovery is counted and ends our trial as a failure;
it cannot rescue a reported success. Native removal after delivery remains the
task's collection semantics. The source collision statistics count action-level
events, not unique physical contact pairs.

## What this establishes, and the next task

An empty collection room does not require both abilities: either expert might
finish alone. Its purpose is to check transfer, expose interference, and test
whether dispatch improves over the best fixed baseline. Success here does not
establish complementary skill composition.

The proposed next task is **push-to-pick delivery**: the pusher stages an object
at a transfer bay; the lifter retrieves it and places it past a low barrier.
Multiple objects create choices about staging order, bay occupancy and parallel
work. A later second gate or blocked bay can make the schedule depend on state.

That task needs qualification of the frozen experts under new geometry and
targets. The released single-agent policies are trained for a corner collection
goal; arbitrary intermediate goals cannot be assumed to work. If transfer
fails, record the failure and decide whether to relax the released-weights-only
constraint before training new primitives. Do not relabel one expert's internal
movement phases as several independently pretrained skills.

## First measured pilot

The one-cube visual smoke test (seed 0, both experts) succeeded in five dispatch
decisions and 4.57 simulated seconds. It is an illustrative rollout, not a
held-out performance result. The saved animation is `outputs/planar/demo.gif`.

After 64 training episodes, the five paired two-cube tests gave:

| Dispatch policy | Successful episodes | Objects collected across all five trials |
| --- | --- | --- |
| Pushing only | 1/5 | 4/10 |
| Lifting only | 3/5 | 6/10 |
| Both always | 0/5 | 2/10 |
| Alternating | 0/5 | 2/10 |
| Learned head | 0/5 | 1/10 |

All trials had zero native recovery resets. Failed trials remain in the report.
The learned head did not improve coordination in this pilot; running the
released experts simultaneously also transferred poorly. Frequent robot
collision events show interference in the shared room. Five seeds cannot
support a general performance ranking.

Before adding handoff geometry, the next experiment should improve and qualify
coordination in this room, with fresh evaluation seeds. The current checkpoint
is retained as a failed pilot rather than presented as a successful controller.
The raw paired results are in `outputs/planar/pilot_comparison.json`; the compact
summary is `outputs/planar/pilot_summary.json`. The full project test suite passed
14 tests, including three tests for this experiment.

## Provenance and files

Source revision: `336e03a53a24192d4d7f92f06eb4c13d9f254333`.
Original MIT license and source remain under
`data/vendor/spatial_intention_maps/`. Runtime compatibility changes remove
unused real-robot imports and update the moved scikit-image import; the native
source is unchanged on disk. The dispatch wrapper changes the action boundary
to support the three coordination choices.

| Expert | Published run | Weight SHA-256 |
| --- | --- | --- |
| Pushing | `20201214T092813073846-pushing_1-small_empty-base` | `a12280f17498ceca7b66aea718302d967d59d6cea3649c5050a499d5b97f8f9f` |
| Lifting | `20201217T171254022070-lifting_1-small_empty-base` | `dec69554b8e35cb114ee25236b6588645e82f6c02aa9df7044ef1454cf350a41` |

| File | Purpose |
| --- | --- |
| `src/planar/experts.py` | Released networks, integrity checks and compatibility imports |
| `src/planar/world.py` | Native task, synchronous dispatch, rewards and measurements |
| `src/planar/head.py` | PPO coordination head and checkpoint contract |
| `experiments/planar_setup.py` | Source/checkpoint acquisition and pathfinding build |
| `experiments/planar.py` | Training, paired baseline evaluation and rendered demos |
| `tests/test_planar.py` | Native rollout, accounting, frozen experts and checkpoint checks |

The existing assembly launcher and experiment remain separate.

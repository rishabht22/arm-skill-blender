# Precision assembly and efficient skill composition

The active project is a two-station production cell in one MuJoCo world. Two
Panda arms execute released, learned assembly behaviors. A small coordination
head chooses which ready job to start and when the arms should work in parallel.
The earlier cube-stacking experiment is in [MuJoCo manipulation](../mujoco_manipulation/).

## Run now

From this project folder, install the runtime, qualify the frozen skills, and
train a head. Then watch a short batch from a terminal with a working display:

```bash
python -m experiments.setup
python -m experiments.run preflight --samples 5
python -m experiments.run train --problems 100 --epochs 40 --rounds 2
python -m experiments.run evaluate --episodes 1 --kits 2
```

Train a new head, then compare it against the scheduling baselines:

```bash
python -m experiments.run train
python -m experiments.run benchmark --headless --episodes 3 --kits 3
```

The launcher uses `mjpython` for Mac viewing commands. Physics and display-model
synchronization were tested headlessly. The native window must be opened from a
terminal with a working graphics connection.

The demo snapshot contains a head retrained with the original pilot budget and
the qualification profiles it expects. After setup, a fresh checkout can run
the bundled head with these commands.

```sh
mkdir -p data/assembly
cp outputs/demo/profiles.json data/assembly/profiles.json
python -m experiments.run evaluate --headless --episodes 1 --kits 2 --checkpoint outputs/demo/best_head.pt
```

Use `preflight --verify-only` to run qualification while keeping the active
profile. The loader checks that the active profile and checkpoint contract match.

## What changed

The base controller is the released [AssembleNIST expert](https://github.com/suniique/assembly-nist),
an independent implementation of AlphaNIST/SGS. It performs grasping,
reorientation, precision insertion, gear seating, connector mating and nut
threading through learned joint actions. Our adapter adds timed entry and
release/retreat motions; their time counts toward task completion.

Both arms reuse one frozen multi-task backbone, with disjoint allowed catalogs.
These are twelve assembly variants across five families, not twelve unrelated
networks or twelve completely different behavior families.

| Arm 0 | Arm 1 |
| --- | --- |
| R4: 4 mm round rod | Gs: small gear |
| R12: 12 mm round rod | Gm: medium gear |
| P4: 4 mm keyed rectangular peg | Gl: large gear |
| P12: 12 mm keyed rectangular peg | N8: M8 threaded nut |
| BNC: coaxial connector | N12: M12 threaded nut |
| USB: USB-A connector | RJ45: network connector |

The policy receives point clouds reconstructed from meshes and simulator poses,
plus proprioceptive history. It is state-based, not a camera or VLA policy.
The actual assembly is performed by the released neural network, without
scripted assembly phases or a teacher-action fallback.

## Task and environment

Each kit follows four work-order stages:

1. Arm 0 inserts a rod or keyed peg.
2. Arm 1 seats a gear.
3. Arm 0 mates a connector.
4. Arm 1 finishes a threaded-nut or connector component.

Several kits arrive at different times. Because each kit returns to both
stations, job order affects waiting and total batch completion time.

These are **kit work-order dependencies**, not physical handovers of the same
part between arms. Stations have separated workspaces. Shared-workspace contact
coordination and arm-to-arm collision avoidance are outside this benchmark.
The first coordination head is centralized.

Both active robots and workpieces run in one MuJoCo model. Controls run at
25 Hz; physics uses 0.5 ms steps. Native gains, limits and task geometry are
preserved. The combined timestep is finer than the source's coarse-task setting,
so expert transfer is tested in this scene rather than assumed.

A new workpiece is supplied at dispatch. Robot joints, velocities, the other
station's part, policy history and simulation clock are preserved when the
contact model changes. No robot teleports, welded grasps, action replays or
within-skill object resets are used. Room walls are absent; task fixtures remain.
A persistent display model copies poses from the contact simulation and never
drives its physics.

## Qualification and freezing

```bash
python -m experiments.run preflight --samples 5
python -m experiments.run freeze
```

Qualification executes each variant five times with its partner parked. The
source reset generator rejects invalid candidate poses before execution; all
executed trials and failures remain in the report. The native assembly goal is
1 mm. After release and withdrawal, the part must remain within 2 mm. Timeouts
and native safety violations fail the attempt. Every variant must pass at least
80% of its attempts before training is allowed.

Initial qualification passed **59/60** attempts. P12 passed 4/5; the other variants
passed 5/5. This is a small qualification sample, not a reliability guarantee.
The expert stays in eval mode with gradients disabled. Parameter and checkpoint
hashes are checked after physical runs. The head optimizer never owns the expert.

The pinned upstream revision is
`eb5d15e004591d380c0d521da67e025ebee37703`. The released checkpoint is
`e8192_it12500.pt`, SHA-256
`eb2cd4e9aacda2d26d23c059570522559e9fcd812a64f0aef60daea83408581e`.
All upstream files are downloaded locally with Git-blob and SHA-256 checks.
Their original source and notices remain in `data/vendor/assembly_nist/`.

For Mac compatibility, a generated C++ source replaces the x86-only FP16
conversion with a portable equivalent, checked against all 65,536 encodings.
The plugin registration macro is adapted to MuJoCo 3.3. Original upstream
files and learned weights are unchanged.

## How the new head trains

The method is **lookahead-planning distillation with DAgger**, not PPO.

1. Use measured skill times and variability from physical qualification.
2. Generate work orders with different skills, release times and batch sizes.
3. Search expected-duration schedules to label useful dispatch decisions.
4. Fit a small set-based dispatch network.
5. Collect states visited by that network, relabel them with lookahead, and fit
   again. Validation selects the saved checkpoint.

Training uses a cheap duration surrogate, not full robot physics. The head sees
job status, readiness, expected durations, uncertainty and remaining work. It
cannot inspect sampled future completion times. Action masks enforce resources,
release and precedence constraints; they do not identify the best action.

The initial head has **16,514 parameters**. It used 100 planning problems per
round, 40 fitting epochs and two DAgger rounds: 3,220 labeled decisions and about
68 seconds of training locally. The default is 180 problems. Training includes
two to six kits, or eight to twenty-four operations. The default physics test
uses three kits. This is generalization within one production template.

```bash
python -m experiments.run train --problems 180 --epochs 40 --rounds 2
python -m experiments.run evaluate --checkpoint outputs/assembly/RUN/best_head.pt --headless --kits 4
```

Training seeds start at 0, validation at 10000, held-out physics testing at
20000. The `evaluate --proxy` diagnostic is explicitly labeled as a duration
surrogate. It must not be used as evidence of robot speed or success.

## Measure efficiency in physics

```bash
python -m experiments.run benchmark --headless --episodes 10 --kits 3
python -m experiments.run baseline --scheduler johnson --headless --episodes 3
```

Every scheduler receives the same work orders and incoming part states. The
standard suite compares serial execution, parallel FIFO, critical-path priority,
an aggregate Johnson-style heuristic, and the learned head. Johnson's rule has
no optimality guarantee for this four-stage problem. Shortest-processing-time
and lookahead baselines are also available through `baseline --scheduler`.

Reports retain success rates, simulated completion time, utilization, job
traces, failures, dispatch compute time and paired comparisons. Failed batches
receive a 180 s penalty in the aggregate metric, not a fast completion. Simulator
wall time and simulated robot execution time are reported separately.

Initial three-batch physics pilot, twelve operations per batch; all five
schedulers succeeded on all three batches:

| Scheduler | Mean simulated completion time |
| --- | ---: |
| Serial | 54.56 s |
| Parallel FIFO | 35.76 s |
| Critical-path | 37.44 s |
| Aggregate Johnson heuristic | 34.36 s |
| Learned head | 35.57 s |

The head was about 35% faster than serial execution, approximately tied with
FIFO, and slower than the best heuristic. It has **not established an advantage
over strong classical scheduling**. Three batches do not support a broad
statistical claim. The larger benchmark is intended to expose that distinction.

## Files and settings

| Location | Purpose |
| --- | --- |
| `experiments/configs/assembly.json` | Training budgets and evaluation defaults |
| `src/assembly/experts.py` | Released backbone and disjoint task catalogs |
| `src/assembly/world.py` | Shared contact model and state preservation |
| `src/assembly/service.py` | Timed entry/exit and neural skill execution |
| `src/assembly/profiles.py` | Qualification, measured durations, frozen manifest |
| `src/assembly/scheduling.py` | Work orders, constraints, surrogate and physics task |
| `src/assembly/planner.py` | Classical baselines and bounded lookahead |
| `src/assembly/head.py` | Variable-batch head and DAgger training |
| `src/assembly/benchmark.py` | Paired native-physics comparisons |
| `src/assembly/visualization.py` | Persistent display synchronized from physics |
| `data/assembly/` | Qualification, frozen manifest and cached incoming states |
| `outputs/assembly/RUN/` | Head weights, validation, duration model and contract |
| `outputs/assembly/benchmark_RUN/` | Raw episodes and comparison summary |

## Another Mac or the lab server

```bash
python -m experiments.setup
```

Setup installs the local runtime, downloads about 176 MB of pinned source,
models and assets, and builds the contact plugin. A C++17 compiler is required:
`clang++` on Mac or `g++` on Linux. Everything runs on CPU. No GPU or GPU-waiting
process is required. Mac was tested here; Linux builds must run local verification.

Copy `data/assembly/profiles.json`, `data/assembly/frozen.json` and the desired
`outputs/assembly/RUN/` folder to use the same trained head. Verify the new
machine without replacing the trained duration model:

```bash
python -m experiments.run preflight --samples 5 --verify-only
python -m experiments.run evaluate --checkpoint outputs/assembly/RUN/best_head.pt --headless
```

Normal `preflight` refreshes profiles. Changed physics, catalogs, experts,
duration estimates, features or planner code require matching contracts and,
when appropriate, new qualification and training. Never copy virtualenvs or
compiled plugins between operating systems.

```bash
.venv-mujoco/bin/python -m pytest
```

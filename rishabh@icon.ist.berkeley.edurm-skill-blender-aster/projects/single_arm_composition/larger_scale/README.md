# Large-scale single-arm skill composition

The research question is whether one arm can reliably compose a fixed
library of atomic skills into tasks whose dependency structures were
absent from training. That result will inform later experiments on
whether composing individual skills can produce new multi-arm behavior.

## Workcell

Build a new MuJoCo laboratory sample-handling cell with one Panda.
Use mechanically functional fixtures such as keyed carriers, racks,
an indexed processing station, a latched enclosure, and separate
inspection and rejection stations. Include clearance constraints,
interlocks, and meaningful partial states. Derive task success from
physical state and verified events. Verify each primitive in isolation
and test that invalid mechanical sequences fail.

## Experimental protocol

- Generate tasks as dependency graphs over the same frozen primitive
  library. Require skill coverage in training for structural test cases.
- Deduplicate canonical task structures and keep whole task families
  disjoint across train, validation, and test. Reserve a structural test
  with familiar objects and argument ranges. Use separate splits for
  novel bindings, fixture layouts, longer horizons, and recovery starts.
- Target tens of thousands of distinct task instances, a million or
  more expert episodes, and thousands of paired closed-loop test starts.
  Count distinct task structures and layouts separately from reset seeds.
- Run model and data scaling curves with multiple independent training
  seeds and track actual parameters, decision examples, and throughput.
  Compare against retrieval or fixed plans, a state-aware symbolic
  baseline, and a feasibility expert using the same skill controllers.
- Match starting states across methods. Report task success, subgoal
  progress, failure modes, and uncertainty across task families and
  training seeds. Log every attempted rollout with its manifest and
  checkpoint provenance. Keep the sealed test split out of tuning.
- Provide reproducible commands for distributed generation, training,
  closed-loop evaluation, and matched viewing of a live policy and the
  recorded expert episode.

The implementation and reproduction commands follow. See [RESULTS.md](RESULTS.md)
for executed runs and pending targets.

## Aster sample preparation workcell

A Menagerie Panda serves four feed nests and raised recovery cradles, an
optical reader, an indexed process instrument, and separate release and
rejection racks. Each carrier has a keyed stem. The reader requires seating
contact and a ray hit on that carrier's stem before it returns the carrier's
assigned assay grade. The process instrument has a sliding shield, a latch
bolt and strike, a spring-return plunger, and a two-position rotor. A measured
plunger stroke triggers the index drive only when position switches report a
closed shield and an engaged latch. Sorting uses the measured assay result
and the task's assigned destination.

Thirteen frozen controllers execute Cartesian moves with joint-space servo
control. The pads must make opposed contact before a compliant weld captures
the measured grasp transform. The weld releases at the end of the motion.
Carrier seating requires depth, orientation, and floor contact. Latch travel,
panel motion, and rotor motion have separate checks. One recentering pass uses
the measured grasp transform when the first seating attempt stalls. Recovery
starts place one or two rotated carriers in raised cradles, requiring physical
extraction, rotation, and reinsertion into a feed nest.

## Composition protocol

Each graph contains four carrier routes. An inspection transaction contains
three atomic actions, and a processing transaction contains seven. Route
templates vary the order and number of processing visits. Cross-route edges
link the end of a transaction to the beginning of a transaction on another
carrier. The reader and rotor each have one physical slot, so even a
topologically eligible action can wait for an occupied station. The last
action routes each carrier to its assigned release or reject nest.

The generator reduces transitive cross-edges and canonicalizes job labels
before hashing each DAG. Families use the quotient graph on the four carrier
routes. The family hash determines the split, including across manifest
generations. The structural slice uses familiar layout and assignment ranges,
route templates and horizons. The binding slice changes the carrier and
destination permutations. The layout slice relocates the reader and output
racks. The horizon slice adds process visits beyond the fitting range. The
recovery slice starts two carriers in cradles. Train has examples with one
cradle recovery. Each test slice has its own graph families, so comparisons
between slices also reflect their different sampled families.

Reset seeds are derived from the manifest, split, task ID and repetition.
Each method receives the same seeded start. Task IDs include graph, layout,
carrier and goal assignments, assay grades, and recovery configuration.
Additional reset seeds increase the episode count independently of those
task-instance counts. The hash seal verifies manifest contents. The test lock
records checkpoint hashes and test counts before the evaluation CLI opens a
test slice. This is a procedural seal backed by integrity checks.

The learned selector scores unfinished goal nodes using the goal DAG,
completion ledger, measured fixture state, locations, assay results,
positions, and binding information. The feature encoder exposes the count
of unfinished predecessor nodes. The model learns to use that input and the
physical observations to choose an action. Its inference mask removes
completed nodes and padding. A premature request is scored as a violation
before the scheduler actuates the arm. The expert labels all admissible
actions at each measured decision and chooses one by transaction continuity
and travel distance. The symbolic baseline chooses the first admissible node
in graph order. The retrieval baseline chooses a train DAG with the same
route templates and replays its fixed topological order, with a lane-order
fallback. The expert can retry a failed action. Every baseline uses the same
frozen controllers.

Task success requires every graph event, correct physical output seating,
the expected index counts and assay reads, and an open, unlocked shield.
Rollouts also record subgoal progress, dependency violations, failed
primitives, and recoveries. The report checks matching start fingerprints and
resamples families, graphs within families, and model seeds for the paired
95% intervals.

## Reproduce

From the repository root, create a Python 3.12 environment and install the
subproject pins. Then run commands from `projects/single_arm_composition`.

```sh
python3.12 -m venv .venv-aster
. .venv-aster/bin/activate
pip install -r projects/single_arm_composition/larger_scale/requirements.txt pytest
cd projects/single_arm_composition
python -m pytest -q larger_scale/tests
python -m larger_scale check --layouts 128 --repetitions 2 --workers 8 \
  --out larger_scale/outputs/gate.json
```

Generate a fresh sealed manifest and begin the large collection. Use
`--tier pilot` for a separate development-only gate.

```sh
run=larger_scale/outputs/large
python -m larger_scale manifest --tier large --out "$run/manifest"
python -m larger_scale audit "$run/manifest"
python -m larger_scale collect --manifest "$run/manifest" \
  --out "$run/train" --count 1000000 --workers 8 --batch 32
python -m larger_scale collect --manifest "$run/manifest" --split dev \
  --out "$run/dev" --count 4096 --workers 4 --batch 32
```

For distributed collection, give each host the same sealed manifest,
controller source, pins, count and batch size. Assign one unique `--part`
from zero through `--parts - 1` per host. Each part writes whole gzip shards
with atomic replacement. Gather the uniquely named data shards, statistics,
and part contracts into the same corpus directory. Repeating a part command
resumes completed shards. Evaluation supports the same partition flags
with a shared test lock.

The paired plan fits three independent model seeds on a 512-episode prefix
and locks 3,072 matched starts. The full plan fits the data curve at width
64 and the model curve at 32,768 episodes. It fits three seeds per point and
evaluates the entire roster under a second fixed lock. Start the plans once
their corpus prefixes exist, or use `--wait-for-shards` while collection is
in progress.

```sh
python -m larger_scale.study --plan paired --manifest "$run/manifest" \
  --train-data "$run/train" --dev-data "$run/dev" --out "$run/paired" \
  --workers 8 --wait-for-shards
python -m larger_scale.study --plan full --manifest "$run/manifest" \
  --train-data "$run/train" --dev-data "$run/dev" --out "$run/full" \
  --workers 8 --wait-for-shards
```

`python -m larger_scale --help` lists the individual `train`, `lock`,
`evaluate`, and `report` commands. Both evaluation and training log their
exact checkpoint and corpus provenance. Final reports require every rank in
the declared slice. Use `report --partial` for completed matched shards and
`corpus-status` for physical episode counts while collection runs. The pilot
plan runs closed-loop development comparisons without opening test.

Use the same task and reset for live learned execution, expert capture, and
recorded expert replay. Set `task` to a task ID from an evaluation shard or
from `paired/replay_task.txt` in the executed run.

```sh
task=$(cat "$run/paired/replay_task.txt")
model="$run/paired/models/w64-n512-s17.pt"
for mode in --record-demo --real-demo; do
  python -m larger_scale view --manifest "$run/manifest" --split structural \
    --lock "$run/paired/test_lock.json" --task "$task" --reset 0 \
    --recordings "$run/paired/recordings" "$mode"
done
python -m larger_scale view --manifest "$run/manifest" --split structural \
  --lock "$run/paired/test_lock.json" --task "$task" --reset 0 \
  --recordings "$run/paired/recordings" --model "$model"
```

Append `--show` for the interactive viewer and use `mjpython` on macOS.
Append `--video output.mp4` for an MP4 with a working graphics context and
FFmpeg. Without either flag, the commands verify the rollouts headlessly.
The recording stores actuator commands, equality state, physics ticks, and
provenance. The replay verifies the initial and final physical digests.


To render the completed development scaling points, install Matplotlib
and run the standalone plotter.

```sh
pip install matplotlib==3.10.0
python -m larger_scale.curves larger_scale/outputs/pilot/dev_report.json \
  --out larger_scale/outputs/pilot/scaling.svg
```

## Resume on another machine

Copy the working source tree and `larger_scale/outputs/large` to the
compute host. Git omits the output directory, so copy it separately if
moving the source via Git. Keep the existing manifest, episode shards,
and checkpoints together. Install the pinned packages in a fresh local
environment on that host.

Run `collect` with the existing count and batch size to resume physical
collection. It uses CPU workers. Run the full study with `--device cuda
--fit-only --wait-for-shards` on the GPU host to resume fitting while
collection proceeds. The saved development corpus covers the full plan's
development requirement. The plan skips finished model seeds and starts
each new tier after its train prefix is available.

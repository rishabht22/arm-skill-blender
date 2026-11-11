# Goal-driven results

Three width-64 selectors had 10,625 parameters each. Training used 8,343
distinct symbolic goal-state examples from 487 unique goal and start pairs.
Each fit also executed the same 24 physical planner demonstrations.
All 24 completed and supplied 188 measured decisions per fit. Those
decisions duplicated goal-state examples already covered by the symbolic
data. Each fit ran 28 epochs on CPU. The selected checkpoint for each
seed completed all 48 abstract development starts.

## Closed-loop MuJoCo benchmark

Five methods ran from each of 1,024 matched starts. The three learned
seeds, the replanning expert, and the fixed script used the same workcell
and physical reset. The fixed script planned from an all-feed state for
each goal and replayed that sequence in the actual cell.

| Held-out suite | Starts | Distinct goals | Seed 17 | Seed 59 | Seed 101 | Replanning expert | Fixed feed plan |
|---|---:|---:|---:|---:|---:|---:|---:|
| Novel goal combinations | 256 | 36 | 255 | 255 | 255 | 255 | 124 |
| Three goal carriers | 256 | 227 | 253 | 253 | 253 | 253 | 115 |
| Partial starts | 256 | 85 | 256 | 256 | 256 | 256 | 0 |
| Ordering constraints | 256 | 36 | 256 | 256 | 256 | 256 | 256 |
| Total | 1,024 | | 1,020 | 1,020 | 1,020 | 1,020 | 495 |

The learned policies completed 3,060 of 3,072 rollouts with zero invalid
requests. All learned failures were physical seating failures. The
replanning expert had the same aggregate completion count as each seed.
In the three-carrier suite, each seed completed one start the planner
failed and failed one start the planner completed. There were 2,109
successful learned rollouts with an action order different from the
replanning expert on the same successful start. Mean excess skill calls
over the abstract planner reference were zero in every learned slice.

## Live examples

The same outcome goal asks the Panda to index B once, return it to feed,
and leave the enclosure open and unlocked. Seed 17 chose `load B` first
from the feed rack and `close B` first from a loaded, open reset. The two
physical rollouts completed in seven and six selected skills respectively.
The same checkpoint also completed a custom two-carrier JSON goal that
required inspecting A before indexing an already-loaded B.

The local outputs contain the [paired benchmark report](outputs/final/report.json)
and two annotated clips for the [feed start](outputs/final/goal-feed.mp4)
and the [loaded start](outputs/final/goal-loaded.mp4).

## Lab GPU checkpoints on 4,000 matched starts

The user trained three CUDA checkpoints and supplied the summary of the
paired MuJoCo benchmark. The report is on the lab NAS at
`aster-goal-driven/benchmark/report.json`. The benchmark used 1,000
starts per suite and executed 20,000 method rollouts. The suite generator
extends the local evaluation prefixes, so the local and lab results have
overlapping starts.

| Held-out suite | Matched starts | Seed 17 | Seed 59 | Seed 101 | Replanning expert | Fixed feed plan |
|---|---:|---:|---:|---:|---:|---:|
| Novel goal combinations | 1,000 | 993 | 993 | 993 | 991 | 489 |
| Three goal carriers | 1,000 | 989 | 990 | 990 | 986 | 476 |
| Partial starts | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 0 |
| Ordering constraints | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |
| Total | 4,000 | 3,982 | 3,983 | 3,983 | 3,977 | 1,965 |

The learned selectors completed 11,948 of 12,000 rollouts with zero invalid
requests. They recorded 51 seating failures and one index interlock or
stall. The replanning expert recorded 23 seating failures. The fixed plan
made 2,033 invalid requests, including a failed precondition on every
partial start. The completion counts describe this benchmark and its
shared starts.

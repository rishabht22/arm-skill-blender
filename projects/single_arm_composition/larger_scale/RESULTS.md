# Aster experiment record

## Physical gate

The deterministic controller passed 1,024 single-carrier trials over 128
layouts with two reset repetitions per carrier. All 13 primitives passed.
The trial log includes 1,024 successful calls for each nonsort primitive and
512 each for release and reject. A separate regression starts each primitive
from a fresh copy of its physical input state.

All eight counterexamples passed. A measured button stroke of more than
10 mm left the rotor stationary with the shield open or unlatched. The
engaged bolt stopped the shield against the strike after approximately 8 mm
of travel. In the paired drop test, an aligned carrier reached the socket
floor while the reversed carrier stopped approximately 4.8 mm above the
aligned depth. The gate also exercised reversed arm insertion, closed-shield
insertion, sorting before inspection, and rejection of a wrong route.

[Physical gate JSON](outputs/gate.json)

## Scale declared in the large manifest

| Split | Canonical graphs | Task instances | Layouts |
|---|---:|---:|---:|
| Train | 24,000 | 72,000 | 24 |
| Development | 2,400 | 7,200 | 24 |
| Structural | 2,400 | 7,200 | 24 |
| Binding | 1,200 | 3,600 | 24 |
| Layout | 1,200 | 3,600 | 104 |
| Horizon | 1,200 | 3,600 | 24 |
| Recovery | 1,200 | 3,600 | 24 |
| Total | 33,600 | 100,800 | 128 distinct |

The large collection targets 1,000,000 train episodes and 4,096 development
episodes. The paired plan declares 2,048 structural starts and 256 per
transfer slice. It evaluates three 12,737-parameter models and three
baselines at each matched start. The full curve uses 128, 2,048, 32,768 and
1,000,000 physically collected episodes at fixed width, and widths 16, 64,
256 and 1,024 at 32,768 episodes. Each curve point has three training seeds.
These are protocol targets.

[Manifest audit](outputs/large/manifest_audit.json)

## Engineering runs and reproducibility correction

The first controller keyed its inverse-kinematics cache by rounded yaw but
solved the first request for a key at its unrounded yaw. The initial physics
fingerprints matched across methods, but running another method first could
change the final fingerprint. The corrected controller uses the rounded
value both in the key and in the solve. A regression compares cold and warm
runs after executing a different policy. The corrected run uses freshly
generated manifests and retrains every checkpoint under the same model and
evaluation protocol. Some graph families can recur across the two manifests
within the same split. The correction was driven by fingerprint divergence.

The first runs, including the full pilot scaling fits and partially
evaluated locked comparisons, are preserved under
`outputs/engineering_v1`. Their controller provenance is separate from the
corrected run. In that engineering pilot, 384 episodes supplied 19,951
decisions over 264 graphs. The 4-episode condition supplied 197 decisions
and reached 64.5% to 75.3% development admissible-action top-1 across seeds.
The 32- and 384-episode width-64 conditions reached 100% on that decision
metric. Widths 16, 64 and 256 at 384 episodes had 1,649, 12,737 and 149,249
parameters respectively and all reached 100% on that metric. Closed-loop
completion is scored separately.

## Corrected pilot scaling

The pilot evaluated 15 learned checkpoints and three baselines on 48
matched development starts from 37 graphs and 16 families. The models used
training seeds 17, 59 and 101. The symbolic planner and physical expert
each completed 48 of 48 starts. Retrieval completed 9 of 48.

| Selector width | Parameters | Train episodes | Expert decisions | Closed-loop successes across 3 seeds |
|---:|---:|---:|---:|---:|
| 64 | 12,737 | 4 | 233 | 0 / 144 |
| 64 | 12,737 | 32 | 1,630 | 144 / 144 |
| 64 | 12,737 | 384 | 20,246 | 144 / 144 |
| 16 | 1,649 | 384 | 20,246 | 144 / 144 |
| 256 | 149,249 | 384 | 20,246 | 144 / 144 |

The smallest data tier had 143 dependency violations and one failed skill
request. Every seed reached 48 of 48 at 32 episodes and above. At fixed
384 episodes, all three measured widths reached the development ceiling.
The eight-epoch budget was fixed across the pilot points, so larger corpora
also supplied more optimizer updates. Development decisions selected
checkpoints. This pilot used the development split throughout.

[Paired pilot report](outputs/pilot/dev_report.json) · [Scaling figure](outputs/pilot/scaling.svg)

## Verified matched replay

Structural rank zero, task `2d0e7d94ea8a1c03906fb56d`, was selected by
rank before inspecting its result. The seed-17 live model and recorded
expert completed the task from the same reset. Both headless executions
match their corresponding paired evaluation records at the initial and
final physics fingerprints. The expert recording also reproduced its own
final fingerprint when played from actuator commands.

[Learned rollout](outputs/large/paired/learned.mp4) · [Recorded expert](outputs/large/paired/expert.mp4)

## Corrected large run at handoff

The laptop workers were stopped on request. The durable train corpus
contains 12,544 physical episodes, of which 12,465 completed successfully.
It contains 669,688 decision examples across 10,515 canonical graphs and
24 layouts. The development corpus contains 3,104 completed episodes and
167,418 decisions. The full scaling plan completed three model seeds at each
of the 128- and 2,048-episode tiers. Its next tier requires a 32,768-episode
train prefix.

| Locked slice | Starts completed / planned | Learned successes across 3 seeds | Symbolic | Physical expert | Retrieval |
|---|---:|---:|---:|---:|---:|
| Structural | 1,840 / 2,048 | 5,520 / 5,520 | 1,840 / 1,840 | 1,840 / 1,840 | 512 / 1,840 |
| Binding | 16 / 256 | 48 / 48 | 16 / 16 | 16 / 16 | 1 / 16 |
| Layout | 16 / 256 | 48 / 48 | 16 / 16 | 16 / 16 | 3 / 16 |
| Horizon | 16 / 256 | 48 / 48 | 16 / 16 | 16 / 16 | 1 / 16 |
| Recovery | 16 / 256 | 45 / 48 | 15 / 16 | 15 / 16 | 1 / 16 |

These are 11,424 paired method rollouts on 1,904 completed starts. The
structural results span 1,402 graphs and 34 families. The learned selector
used a different successful action order from the expert in 5,230 of its
5,520 structural rollouts. There were zero learned dependency violations
on these completed starts. The three learned recovery failures occurred
on the same reset and at the same physical reseating step as the expert
and symbolic failures. The transfer slices are small interim samples.

[Corpus handoff snapshot](outputs/large/handoff_corpus_status.json) ·
[Partial paired report](outputs/large/paired/progress.json)

## Lab GPU training and locked three-seed evaluation

The lab server trained width-64 selectors with 12,737 parameters on the
12,544-episode original MuJoCo corpus. The user verified six CUDA epochs
for seed 17 and supplied the combined locked report summary for seeds 17,
59, and 101. The report is on the lab NAS at
`aster-large/extra_gpu/three-seed-n12544/report.json`.

| Locked slice | Matched starts | Seed 17 | Seed 59 | Seed 101 | Physical expert | Symbolic | Retrieval |
|---|---:|---:|---:|---:|---:|---:|---:|
| Structural | 128 | 128 | 128 | 128 | 128 | 128 | 38 |
| Binding | 32 | 32 | 32 | 32 | 32 | 32 | 2 |
| Layout | 32 | 32 | 32 | 32 | 32 | 32 | 7 |
| Horizon | 32 | 32 | 32 | 32 | 32 | 32 | 5 |
| Recovery | 32 | 32 | 32 | 32 | 32 | 32 | 2 |
| Total | 256 | 256 | 256 | 256 | 256 | 256 | 54 |

All 768 learned rollouts succeeded with zero dependency violations. The
three seeds shared the same 256 starts. The evaluation reused starts from
the existing manifest, so the results establish reproducibility across
training seeds on this suite. A matched comparison across training data
tiers is still needed to estimate the effect of corpus size.

## Matched comparison across training data tiers

The user supplied the lab server report summary for nine width-64
checkpoints. Each tier used seeds 17, 59, and 101, and every checkpoint
ran on the same 256 starts from the existing locked suite. The report is
on the lab NAS at `aster-large/extra_gpu/matched-data-scaling/report.json`.

| Locked slice | Matched starts | 128 episodes | 2,048 episodes | 12,544 episodes | Physical expert | Symbolic | Retrieval |
|---|---:|---:|---:|---:|---:|---:|---:|
| Structural | 128 | 384 / 384 | 384 / 384 | 384 / 384 | 128 / 128 | 128 / 128 | 38 / 128 |
| Binding | 32 | 96 / 96 | 96 / 96 | 96 / 96 | 32 / 32 | 32 / 32 | 2 / 32 |
| Layout | 32 | 96 / 96 | 96 / 96 | 96 / 96 | 32 / 32 | 32 / 32 | 7 / 32 |
| Horizon | 32 | 96 / 96 | 96 / 96 | 96 / 96 | 32 / 32 | 32 / 32 | 5 / 32 |
| Recovery | 32 | 96 / 96 | 96 / 96 | 96 / 96 | 32 / 32 | 32 / 32 | 2 / 32 |
| Total | 256 | 768 / 768 | 768 / 768 | 768 / 768 | 256 / 256 | 256 / 256 | 54 / 256 |

Each data tier reached the completion ceiling with zero learned dependency
violations. The smallest tested tier, 128 episodes, was sufficient for
every seed on this suite. The two smaller tiers were trained on CPU and the
12,544-episode tier was trained on CUDA, so training device also varied.

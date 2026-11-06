# WEAVE cell

WEAVE uses the RoboFactory two-Panda tabletop layout and its native GPU
backend. Both arms use Panda joint-delta position control. The cell returns
proprioception and arm-relative TCP, partner, object, and fixture poses. World
poses and active-object masks support graph execution. Skill observations and
physical goal predicates will be specified with the skill implementation.

## Reset distributions

[The Phase 1 config](../configs/env/phase1.yaml) defines the reset ranges,
geometry, and smoke-test batch size. Every lane instantiates the same training
cubes and YCB bodies. Each reset independently draws a permutation of tabletop
slots, per-object position jitter and yaw, and a uniformly sampled active-object
count. Unused objects begin in dedicated off-table parking positions and remain
ordinary dynamic bodies. The active count is available to the graph generator
as a scene capacity. Held-out assets belong to the sealed evaluation workflow.

Dock and stack markers jitter independently in the table plane. The handoff
marker stays at the center. The pot translates within its designated region and
its hinge starts at a sampled angle. Each Panda starts from the native posture
with independent uniform arm-joint noise. RoboFactory's reset mask limits
writes and metric resets to the selected lanes.

## Fixture and physics

WEAVE reuses the pinned PlaceFood meshes and rebases the lid around a passive
revolute joint. Its pivot is near the back edge of the lid collision mesh.
Shifting the child shape origins by the inverse pivot offset preserves the
closed geometry. The pot is anchored to the table. Object grasping relies on
friction and finger contact. The native PlaceFood scene retains the sliding
lid for comparison.

The monitor records peak pairwise force and thresholded contact frames per
control step. Its channels cover arm against arm, each arm against the table,
and each arm against active non-target objects. Contact is sampled at control
boundaries, so a transient between samples can be missed. A footprint is the
union over control samples of the TCP and the world AABBs of the selected
wrist and hand collision bounds.

## Validation evidence

`make phase1-smoke` runs a seeded random-policy rollout, verifies repeatable
full resets and isolated partial resets, and checks contact-force isolation
with a deliberate overlap in one lane at reset. It also checks the footprint
accumulator and scans the source and runtime trace for pose writes and grasp
constraints. The output includes the source and config hashes, seeds, and
instrumented throughput. See [ICON Lab setup](REMOTE_RUN.md#phase-1-smoke-test).

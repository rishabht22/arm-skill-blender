# Phase 0 audit

G0 passed. The ICON Lab measurements support branch A. Use native RoboFactory
GPU infrastructure with WEAVE wrappers. All demo tasks failed the BC precheck.
See the [generated report](../results/phase0/iconlab/report.md) and stamped JSON.

## Simulator and environment branch

The upstream requirements pin ManiSkill `3.0.0b12`, whose distribution metadata
pins SAPIEN `3.0.0b1`. RoboFactory's quick start describes a Python 3.9 Conda
environment and a Vulkan runtime. WEAVE uses an isolated Python 3.11 environment
to satisfy the existing repository's Python minimum while keeping the upstream
simulator versions. The Python source in WEAVE uses 3.9-compatible syntax.
[RoboFactory requirements](https://github.com/MARS-EAI/RoboFactory/blob/37426a4c67cd307dbff8009c95080975105e0b51/robofactory/requirements.txt#L1),
[RoboFactory quick start](https://github.com/MARS-EAI/RoboFactory/blob/37426a4c67cd307dbff8009c95080975105e0b51/README.md#L16-L49),
[ManiSkill release metadata](https://pypi.org/pypi/mani-skill/3.0.0b12/json).

RoboFactory's base environment selects CUDA for parallel simulation and
rejects more than one environment on the CPU backend in the same process. It
constructs separate Panda articulations and wraps them in ManiSkill's
`MultiAgent`. `MultiAgent` routes observations and action dictionaries by
agent UID and index. This makes the native tabletop tasks candidates for branch
A. The tabletop initializer broadcasts sampled object translations across a
batch, and PlaceFood edits the whole meat pose array after calling the scene
initializer. Partial reset and per-environment diversity will need explicit
validation even if the throughput runs succeed.
[RoboFactory backend](https://github.com/MARS-EAI/RoboFactory/blob/37426a4c67cd307dbff8009c95080975105e0b51/robofactory/utils/envs/sapien_env.py#L231-L271),
[agent construction](https://github.com/MARS-EAI/RoboFactory/blob/37426a4c67cd307dbff8009c95080975105e0b51/robofactory/utils/envs/sapien_env.py#L399-L430),
[table resets](https://github.com/MARS-EAI/RoboFactory/blob/37426a4c67cd307dbff8009c95080975105e0b51/robofactory/utils/scenes/scene_builder.py#L101-L150),
[PlaceFood reset](https://github.com/MARS-EAI/RoboFactory/blob/37426a4c67cd307dbff8009c95080975105e0b51/robofactory/tasks/place_food.py#L108-L116).

## Demonstrations and replay

The local HDF5 inventory verified the pinned file sizes and SHA256 values.
The released recordings are RGB observations from CPU simulation with absolute
joint targets and a normalized gripper command. Single-arm recordings store
`actions` directly. Two-arm recordings store separate arrays under
`actions/panda-0` and `actions/panda-1`. The observations include joint
positions, joint velocities, RGB frames, and camera calibration. Some tasks
include TCP poses. The inspected groups contain no `env_states` or object pose
trajectories. The recordings carry the original environment IDs and
`commit_info` from ManiSkill's recording wrapper. Current RoboFactory task
registrations are mapped explicitly in the audit config. Exact reproduction
depends on the reset distribution and physics, so matching a task name and
controller is only the starting point.
[Verified demo manifest](../configs/env/robofactory_demos.json),
[audit config](../configs/env/phase0.yaml),
[ManiSkill recording wrapper](https://pypi.org/project/mani-skill/3.0.0b12/).

The default native control rate comes from `SimConfig` in the pinned
distribution. PassShoe and PlaceFood override the physics frequency. The replay
runner records the effective physics and control rates for each instantiated
environment, samples the existing v1 training partition, and seeds both global
NumPy and Torch before applying the published reset seed. It plays each command
sequence twice in state-observation mode and captures simulator object poses
and pot joint state. It records the official final success predicate,
joint-position error against the recording, and repeat drift in joint and object
state. Object poses from a replay describe that simulated execution. The
published data provide no object-pose trace to compare them against.
[Pinned ManiSkill release](https://pypi.org/project/mani-skill/3.0.0b12/),
[PassShoe physics config](https://github.com/MARS-EAI/RoboFactory/blob/37426a4c67cd307dbff8009c95080975105e0b51/robofactory/tasks/pass_shoe.py#L84-L91),
[PlaceFood physics config](https://github.com/MARS-EAI/RoboFactory/blob/37426a4c67cd307dbff8009c95080975105e0b51/robofactory/tasks/place_food.py#L85-L92),
[replay runner](../src/weave/phase0_sim.py).

A separate open-loop run tests delta commands derived from recorded pre-action
joint positions. The normalized delta is clipped to the pinned Panda controller's
range. The results retain clipping counts and raw and converted success rates.
Native-scene replay is a precheck for BC. Any different WEAVE scene used for BC
must pass its own replay validation. Demo validation and test trajectories
remain reserved for final evaluation.
[Conversion and partition selection](../src/weave/phase0_data.py).

## Contacts and throughput

The pinned ManiSkill implementation in `mani_skill/envs/scene.py:683–743`
queries contacts for actor or link pairs and divides impulse by the physics time
step. The GPU branch uses SAPIEN's per-pair CUDA impulse query. Panda's grasp
check uses force and direction at both fingers
(`mani_skill/agents/robots/panda/panda.py:236–262`). The native two-arm loader
creates the Pandas in the same physics scene. The audit probes contact across
all link pairs with a deliberate arm overlap in one environment at reset and
checks the other lanes for isolation. This confirms that the API returns
meaningful forces across environments before the safety monitor relies on it.
[ManiSkill release](https://pypi.org/project/mani-skill/3.0.0b12/),
[contact probe](../src/weave/phase0_sim.py).

The benchmark launches each native task and batch size in its own process. It
times synchronized state-observation control steps, records repeated samples and
their median, and measures NVIDIA's process memory after warmup. Incremental
MiB per environment subtracts the pre-scene CUDA process baseline and divides
by the batch size. Fixed PhysX allocations affect that quotient, so the raw
process readings are retained. The contact probe runs outside the timed loop.
The timings cover the native scenes and their built-in success checks. WEAVE's
training loop will need a separate throughput measurement once its safety
logging and policy inference are integrated.
[Benchmark config](../configs/env/phase0.yaml).

## Geometry and asset reuse

The native YAML provides opposing Panda base poses, the table surface, the
central workspace, primitive cubes, and an asset-backed table with a box
collider. These are candidates for reuse in the WEAVE cell. The PlaceFood pot
URDF's moving joint is prismatic. Use the original asset and official success
predicate for the native comparison. For the WEAVE hinged-lid task, preserve
the geometry where practical and validate a revolute articulation with contact
manipulation before using it in training. Treat replay from PlaceFood as a
potential source of BC only if it validates under the changed task and controller.
Long Pipeline Delivery uses four Panda arms and is outside the specified
two-arm native evaluation.
[Two-arm table and bases](https://github.com/MARS-EAI/RoboFactory/blob/37426a4c67cd307dbff8009c95080975105e0b51/robofactory/configs/table/two_robots_stack_cube.yaml#L56-L90),
[PlaceFood asset reference](https://github.com/MARS-EAI/RoboFactory/blob/37426a4c67cd307dbff8009c95080975105e0b51/robofactory/configs/table/place_food.yaml#L22-L35),
[pot URDF](https://huggingface.co/datasets/sparklexfantasy/RoboFactory_asset/blob/58ad250efb3de75f956c852ba8ad50e7ca30409f/objects/pot_annotated/mobility.urdf#L78),
[Long Pipeline Delivery](https://github.com/MARS-EAI/RoboFactory/blob/37426a4c67cd307dbff8009c95080975105e0b51/robofactory/tasks/long_pipeline_delivery.py#L25-L36).

## Gate decision

Native GPU batching and contact probes passed at every configured size.
Phase 1 will validate per-environment randomization and partial resets in the
WEAVE wrapper. Start skill development with RL-only training.

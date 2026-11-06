# Published multi-task training data

The default v2 coordinator trains exclusively on original files from
[FACEONG/RoboFactory_Dataset](https://huggingface.co/datasets/FACEONG/RoboFactory_Dataset/tree/63f8a85),
pinned revision `63f8a85`. All five task folders contain 150 episodes. The 750
unique source episodes split into 525 training, 110 validation, and 115 test
trajectories. The source files contain 169,215 recorded action steps.

| Task | HDF5 bytes | SHA256 |
| --- | ---: | --- |
| PickMeat | 230847995 | `7ede3079f213921c31df299ee1f76fd9db1c2220fcdbba3a49942ff1a318e2a5` |
| StackCube | 266445234 | `57d6b7beac139fbc6a4349274de7c5ab778ef04aca529c5d69a5e9692cf9228b` |
| PassShoe | 1869703974 | `13d8264fa8809216a0db271e8e5933239910d43d062c3412d6d8d6a8c705fb72` |
| PlaceFood | 1590607246 | `1606c85f5fc91bc35a58674dfb5e3a91bfbaf1be7028ef5f61a72bc121e79de2` |
| TwoRobotsStackCube | 1682515074 | `76641c6a8bb944fb5eacfec51bd5174f34553d11851aae89383044f323130510` |

The files are kept intact under `data/robofactory/`; JSON metadata, sizes, and
both file checksums are recorded in `manifest.json`. `python -m skill_composer
download` verifies existing files before reuse. Files are Git-ignored because
of their size. The publisher marks the dataset MIT.

## What is derived from the recordings

`cell/dataset.py` reads the recorded seven-joint motor targets, normalized
open/close commands, measured nine-joint positions, and measured nine-joint
velocities. It does not read a locally generated expert rollout. The source
images are used for the original-recording preview, not policy training.

The publisher does not provide the semantic labels used here. We annotate source
task goal graphs using the official expert programs and infer operation timing
from gripper ramps and joint-target motion. Pick includes closing and the first
post-grasp lift. Place includes release and the source expert's clearance retreat
before a dependent receiver can proceed. The lid-opening boundary is the access
milestone before food placement. Labels remain weak, task-specific annotations.

At source operation boundaries, the dataset stores goal-graph progress, skill
availability, both arms' current measured joint states, and the next expert
operation or WAIT. No episode ID, future joint command, or task name enters the
policy observation. Source episode IDs remain in metadata for split auditing.
There are 1,884 decision events after annotation. Candidate permutation, abstract
arm exchange, and modality dropout are augmentation of existing source records;
they are not counted as extra demonstrations.

The long `relay_stack` and `relay_container` graphs are defined only for execution
and evaluation. They are not inserted into the supervised training data. Their
roles and dependencies are supplied, so transfer is graph-conditioned scheduling
and skill recomposition, not automatic discovery of unseen task plans. There are
only five original source program templates despite 750 continuous trajectories;
more recordings do not imply 750 different task structures.

No generated expert trajectories, synthetic planner labels, DAgger rollouts, or
reinforcement-learning rollouts are used for training in this version. Physics
evaluation rollouts are stored separately under `outputs/cell/`.

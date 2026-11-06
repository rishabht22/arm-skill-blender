# Goal-conditioned workbench

This experiment shares the Panda motion executors with the cube-in-container project. The new scene is an open workbench with one movable block, a stack support, and shared and private staging pads. It has its own `compose` commands, raw data, prepared dataset, and output directory.

The input is the desired placement, the block's location or holder, and a map of which arm can reach each site. A shared learned scorer selects an executable `pick(arm, site)` or `place(arm, site)` call. Each call runs to completion and the next decision uses the resulting scene state. At runtime the workbench measures the block and gripper state in MuJoCo and checks site reach with the executor's inverse kinematics. Reach and gripper state determine the legal calls. The scorer receives goal match, whether the acting arm can reach the requested goal, and whether another arm can reach the call's site. This gives it the ingredients for routing, with the sequence decided one call at a time. It receives no prewritten skill graph, arm assignment, or layout ID.

The scope is one payload and serial calls. Gripper state determines whether the available skill is pick or place, so this experiment isolates **arm selection, target selection, and when to introduce a transfer**. Motor control uses the shared scripted IK routines and idealized grasp attachments. These results concern routing over those executors. Pretrained arm weights are a separate milestone.

## Published data

The source is RoboFactory **LongPipelineDelivery**, a four-arm delivery task outside the five tasks used for the cube-in-container project. The final two recorded stages supply both a relay sequence and its observed direct-delivery suffix. We map source arms `panda-1` and `panda-0` to the two workbench roles and infer decision timestamps from recorded gripper and joint commands. Object location, destination meaning, and reach relations are task annotations based on the published expert script. The source annotations describe delivery to a region. Stacking in MuJoCo uses the same place call with a higher target pose.

The prepared data has **600 decisions from 150 source episodes**. There are 105 training, 22 validation, and 23 test episodes, and 450 decisions have multiple legal calls. The 600 rows share **four distinct abstract states**. This makes held-out source accuracy a pipeline check. Closed-loop behavior on the workbench is the composition test.

- [Published dataset revision](https://huggingface.co/datasets/FACEONG/RoboFactory_Dataset/tree/63f8a85/LongPipelineDelivery)
- [Published expert implementation](https://github.com/MARS-EAI/RoboFactory/blob/main/robofactory/planner/solutions/long_pipeline_delivery.py)

`data/raw/` holds the original files and `data/processed/decisions.json` contains the prepared examples, source checksums, episode splits, source arm IDs, and frame numbers. The download is approximately 4.05 GB. The HDF5 SHA256 is `b4e123df9bcdb5739c29e44336a992c869ee4d7f44c24c04ca368ecf62fe984c`. The downloader and preparer both verify the pinned files. Raw data, prepared examples, and outputs live locally and are Git-ignored.

## Run it

From the repository root, using the existing MuJoCo environment:

```bash
source .venv-mujoco/bin/activate

# For a fresh checkout. The verified raw and prepared files are already here.
python -m skill_composer compose download
python -m skill_composer compose prepare

# Run training yourself.
python -m skill_composer compose train --epochs 100
```

Training writes `outputs/composer.pt`, `outputs/training.json`, and `outputs/imitation_evaluation.json` relative to this project directory. To rescore the checkpoint on the held-out source episodes, run `python -m skill_composer compose evaluate`.

```bash
# Reference search can be viewed before training.
mjpython -m skill_composer compose rollout --layout relay --controller search --viewer

# After training, watch the learned policy make the choices.
mjpython -m skill_composer compose rollout --layout direct --viewer
mjpython -m skill_composer compose rollout --layout relay --viewer
mjpython -m skill_composer compose rollout --layout relay-swapped --viewer
mjpython -m skill_composer compose rollout --layout direct --goal source --viewer

# Evaluate the same seeds and goals with learned and shortest-plan controllers.
python -m skill_composer compose compare --seeds 5
```

Use `python` for headless runs and for the viewer on Linux. `compare` records full dispatch traces and contact counts in `outputs/comparison.json`. Each trace includes all legal alternatives at every choice. The private dock is a physically executable detour. The swapped layout exchanges the shared and private dock bindings, and changing the desired destination in the direct scene should change which arm picks the block. The shortest-plan controller provides a separate reference using the same scene interface and executors. It contributes no training examples.

## Three-block tower

The `compose tower` commands add a three-block goal with interchangeable color
order, shared and private docks, and an occupied-dock layout. The scene tracks
support relations and permits each arm to hold a different block. The composer
chooses pick, place, or stack calls with an arm, object, and destination. Stack
uses the existing scripted placement executor at the measured support height.

The separate `data/processed/tower_decisions.json` contains 1,500 decisions
from 300 published LongPipelineDelivery and ThreeRobotsStackCube episodes.
Simultaneous expert picks are projected to serial decisions with multiple valid
labels. Source goals and reachability are annotated. The training report records
optimizer steps, elapsed time, and checkpoint weight change.

```bash
python -m skill_composer compose tower train --epochs 100
python -m skill_composer compose tower compare --seeds 5
mjpython -m skill_composer compose tower rollout --layout occupied-dock \
  --order yellow red blue --controller guided --viewer
```

Use `compose tower download` and `compose tower prepare` to rebuild the training
inputs in a fresh checkout. The prepared data is already available locally.
The default `learned` controller selects one action at a time from model scores.
`guided` runs symbolic search with learned scores breaking ties. `search` uses
the same symbolic search and can run before training. The comparison separates
all three results. Tower checkpoints and reports go in `outputs/tower/`.

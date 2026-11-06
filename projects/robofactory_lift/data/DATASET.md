# Published data used by this project

Source: [FACEONG/RoboFactory_Dataset](https://huggingface.co/datasets/FACEONG/RoboFactory_Dataset/tree/main/LiftBarrier),
task `LiftBarrier`, upstream revision `63f8a85`. The publisher marks the dataset
MIT. The download is the original public data, not locally generated rollouts.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `LiftBarrier.h5` | 987909637 | `fcc09ee8f0f4bcbfe46e41c467121d8ef565f4f2ff8699526d8a4a2e8d672226` |
| `LiftBarrier.json` | 44275 | `91a44d5c59667337f74565f7897cd60a11aab35644865ca05ad52939ad24ec15` |

Place these two files in `data/raw/` if downloading manually, then run
`python -m skill_composer baseline download` to verify checksums without redownloading
correct existing files. There is no need to download the other task folders.

The HDF5 file has 150 `traj_N` groups. Each contains:

- `actions/panda-0` and `actions/panda-1`: `T × 8`, seven position targets and one
  normalized gripper command. Open is +1, closed is -1.
- `obs/agent/panda-N/qpos` and `qvel`: `(T+1) × 9`, seven arm and two finger joints.
- Three RGB camera streams, `(T+1) × 240 × 320 × 3`, and camera calibration arrays.
- Success, terminated, and truncated flags.

There are no skill labels or object poses in this release's LiftBarrier
observations. Its JSON calls the environment `PickBarrier`, while the current
official implementation registers `LiftBarrier-rf`; the loader identifies the
task by its pinned file provenance and checks the recorded control mode.

Only current proprioception and annotated executor status enter the head. RGB
frames are used for the preview. Commands and future sequence information are
used offline to infer skill boundaries. Success metadata filters the expert
episodes but is never a policy feature.

The annotation rules are task-specific:

1. Approach ends at the first expert gripper command below 0.999.
2. Grasp ends one step after the first command below -0.95.
3. Lift ends at the final seven-joint target change larger than 0.003 rad.
4. Hold occupies the remaining frames.

These are weak labels. They do not establish contact, grasp success, or the
semantic correctness of every motion. The preparation command saves all
boundaries for inspection and rejects episodes that violate these assumptions.

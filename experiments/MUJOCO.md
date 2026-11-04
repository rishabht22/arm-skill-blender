# MuJoCo manipulation baseline

Run from the project root:

```bash
python -m experiments.setup
python -m experiments.run demo
```

The launcher selects the MuJoCo environment and uses `mjpython` for the Mac
viewer. The installed local environment is `.venv-mujoco`. No lab GPU or policy
training is needed. On a server without a display, use `--headless`.

The Panda picks up the red cube, stacks it on the green cube, then picks it up
again and places it at a separate table position. The same scene continues
throughout. Cubes move through contact physics, not teleportation or welded
grasps. Random initial cube positions have a 14 cm clearance for the fingers.

```bash
python -m experiments.run demo --task stack
python -m experiments.run demo --headless --episodes 10
python -m experiments.run demo --headless --video
python -m experiments.run list
```

The reusable skills are reach, grasp, lift, carry, place on a block, place on
the table, release, and retreat. Their implementation is in
`src/mujoco_skills/controllers.py`. They are fixed feedback controllers, not
pretrained neural policies. This is a physical skill-composition baseline;
a learned MuJoCo selector is not connected yet. The previous CALVIN training
command is disabled by default to avoid starting another irrelevant run.

The ten-seed initial physical benchmark completed stack-and-relocate on 10/10
seeds. This is a small test, not a claim of reliability for every layout.
The separate physical tests run with:

```bash
.venv-mujoco/bin/python -m pytest tests/test_mujoco_skills.py
```

Videos require a working graphics connection. A Mac terminal normally has that
connection; a restricted automation session may not. Headless physics checks
do not require graphics. Outputs go under `outputs/mujoco_demo/`.

For released learned task policies, the official sources are:

* RoboMimic model zoo: https://robomimic.github.io/docs/model_zoo/robomimic_v0.1.html
* Diffusion Policy checkpoints: https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/
* Diffusion Policy loader: https://github.com/real-stanford/diffusion_policy

These include full-task experts such as Lift, Can, Square, Tool Hang, and
Transport. They are not interchangeable primitive policies. Their original
simulator/controller versions and observation normalization must be respected;
they have not been validated in this shared Stack scene. Do not assume that a
policy trained for one scene will work when inserted into another scene.

Meta-World also releases controller code for its MuJoCo task library:
https://github.com/Farama-Foundation/Metaworld/tree/main/metaworld/policies
Those experts are task-specific and also require scene compatibility checks.

Previous CALVIN checkpoints and logs remain available. Explicit legacy usage:

```bash
python -m experiments.run --calvin evaluate --video
```

# MuJoCo reach and lift

From this folder, install the original runtime in a Python 3.12 environment.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-training.txt
python -m pytest -q
python -m experiments.basic_policies --check
```

Run the motion preview with `mjpython -m experiments.basic_robot_motion` on
macOS. Use `python` on Linux with a display.

Train and evaluate independent reaching policies. The trainer stores the
policy, observation normalizer, settings, and logs in `outputs/`.

```sh
python -m experiments.basic_policies --arm ur5e --train
python -m experiments.basic_policies --arm jaco --train
python -m experiments.evaluate_policy --headless --episodes 10
```

The initial joint-control pilot is available through the original lift
environment and saved configuration.

```sh
python -m experiments.basic_policies --config experiments/configs/two_arm_lift.json --check
python -m experiments.basic_policies --config experiments/configs/two_arm_lift.json --train --output-dir outputs/two_arm_lift
```

Pass the resulting run directory to `experiments.evaluate_policy --run-dir`.

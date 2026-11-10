# Industrial connector service bench

One Panda services a panel with two interfaces. An asymmetric sensor rib
fits a notch in a recessed vertical socket. A two-sided swing keeper
captures the seated connector's shoulder. A separate power cartridge
has an offset flange and travels horizontally through a guide rail.
The old sensor can go to a quarantine nest while a replacement goes
into the same socket. The model uses rigid contact geometry and
simulator state for mechanical service workflows.

The workcell uses original geometry inspired by industrial connectors
with keyed mating and mechanical locks.
[HARTING connector locking systems](https://www.harting.com/en-CA/locking-systems-for-hoods-and-housings)

The 12 bench-specific skills each execute one path. Shared arm motions
and jaw commands come from the baseline workshop.

| Component | Bench skills |
| --- | --- |
| Keeper | `approach_keeper`, `swing_keeper` |
| Sensor or spare | `descend_to_sensor`, `align_sensor`, `seat_sensor`, `extract_sensor` |
| Cartridge | `align_cartridge`, `feed_cartridge`, `seat_cartridge_stop`, `withdraw_cartridge` |
| Inspection and return nests | `orient_for_inspection`, `seat_in_nest` |

The cartridge stop is checked by contact. Inspection jobs specify the
orientation of the sensor key after it is parked on a nest. The expert sequences the skills. Goals check supported
placement, mating depth and orientation, and the keeper angle. The
headless suite also checks two physical counterexamples. A reversed
connector must hit the wrong socket wall and fail to seat. A connector
under the closed keeper must resist the same extraction that succeeds
with the keeper open.

## Watch and test

Run from `projects/single_arm_composition` with the existing MuJoCo 3.3.0
environment.

```sh
../../.venv-single-arm/bin/python -m service_bench list
../../.venv-single-arm/bin/python -m service_bench check
../../.venv-single-arm/bin/mjpython -m service_bench skill seat_sensor --live
../../.venv-single-arm/bin/mjpython -m service_bench task recommission_panel --keeper locked --live
```

The recommission job opens the keeper, removes the installed sensor,
places it in quarantine, installs the spare and the power cartridge,
and locks the keeper. Use `mjpython` on macOS for the live viewer and
`python` on Linux. The task command runs the expert by default. With a
trained bench checkpoint, pass `--checkpoint` to run the learned
selector live against the same job goal.

## Demonstrations and training

```sh
../../.venv-single-arm/bin/python -m service_bench generate
```

The new export writes `service_bench/data/raw/panda_service_composition.h5`.
It separates isolated skill trials from three full-task splits. Train
and validation share job types with distinct seeds. Test holds out six
job combinations, including reverse replacement, inspection followed by
replacement, and removal of power while retaining the locked sensor.
Test episodes are excluded from fitting and checkpoint selection. Use
`--out` to create another export. The first export remains available as
`panda_service.h5`.

```sh
../../.venv-mujoco/bin/python -m baseline.workshop.train \
  --dataset service_bench/data/raw/panda_service_composition.h5 \
  --out service_bench/outputs/service_composition_selector.pt --epochs 100
```

## Compare a model and a recorded test demo

List the episode IDs:

```sh
../../.venv-single-arm/bin/python -m service_bench tests
```

Replay one stored expert demo in MuJoCo:

```sh
../../.venv-single-arm/bin/mjpython -m service_bench test test_1 --real-demo
```

Run the learned selector live from the same test start:

```sh
PYTHONPATH="../../.venv-single-arm/lib/python3.12/site-packages:../../.venv-mujoco/lib/python3.12/site-packages" \
  ../../.venv-single-arm/bin/mjpython -m service_bench test test_1 --model
```

The expert viewer streams the recorded simulator frames. Model mode
recreates and checks the saved start, then gives the policy the current
observations and saved goal at each skill boundary. The terminal prints
each chosen skill and reports each goal condition after the model stops.
A model can pass with any valid ordering. The checkpoint must come from
the selected dataset. Use `--checkpoint` or `--dataset` to override the
defaults, `--headless` for terminal output only, and `--speed 0.5` to
slow the viewer. A path such as `recommission_panel/00000` also works
in place of the numeric alias.

### Guarded model comparison

Add `--guarded` to `test ... --model` to try a hybrid controller with
the same checkpoint. It focuses the selector on one pending component
at a time, prioritizes removing a sensor that occupies the port,
checks grasp and approach preconditions, and gates `STOP` on the
physical goal checks. The network still ranks the individual calls.
Use plain `--model` for the unassisted selector result.

The trainer fits a small classifier at the saved skill boundaries.
The simulator generates the demonstrations separately, and the
existing controllers execute each selected skill. Next-call accuracy
uses recorded expert states. The live test measures task completion
after the selector's own decisions change the simulator state.

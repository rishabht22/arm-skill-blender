# Interlocked packing cell

One Panda arm loads and unloads colored parts in a two-slot sliding
drawer, moves parts to inspection stations, and secures the cargo.
A fixed canopy covers the drawer when closed. A physical bolt blocks a
tab on the drawer until the bolt is raised. The drawer skill checks that
supported parts stay on its moving floor.

The cell adds five independently testable motions to the shared Panda
skills. `set_bolt` translates the grasped bolt. `slide_drawer` moves the
drawer and its contents. `center_over_slot`, `seat_in_slot`, and
`lift_from_slot` each execute one transfer, seating, or extraction
motion. The job expert sequences these calls and checks support contacts,
relative slot placement, and final fixture positions.

Run these commands from `projects/single_arm_composition`.

```sh
../../.venv-single-arm/bin/python -m packing_cell list
../../.venv-single-arm/bin/python -m packing_cell check
../../.venv-single-arm/bin/mjpython -m packing_cell skill seat_in_slot --live
../../.venv-single-arm/bin/mjpython -m packing_cell task ship_red_blue --mode locked --live
```

Use `mjpython` for live tests on macOS and `python` on Linux. A job can
start locked, unlocked, or open. The suite checks that the locked drawer
cannot open, then runs the five isolated motions and 14 jobs.

## Demonstrations and evaluation

```sh
../../.venv-single-arm/bin/python -m packing_cell generate
```

The export defaults to `packing_cell/data/raw/panda_packing.h5`. Use
`--out` for a new filename. The train and validation splits contain the
same 10 job types with distinct seeds. The four held-out jobs combine
known skills into different cargo assignments, a new final closure
condition, a two-part unload, and a same-slot exchange. Starts cycle
through the three access modes. Seeded variation includes object poses,
the order of independent moves, and occasional prefilled slots.

The same trainer can read the new HDF5 file. The arguments below keep
the packing checkpoint in its own subproject.

```sh
../../.venv-mujoco/bin/python -m baseline.workshop.train \
  --dataset packing_cell/data/raw/panda_packing.h5 \
  --out packing_cell/outputs/packing_selector.pt --epochs 100

PYTHONPATH="../../.venv-single-arm/lib/python3.12/site-packages:../../.venv-mujoco/lib/python3.12/site-packages" \
  ../../.venv-single-arm/bin/python -m packing_cell evaluate \
  --checkpoint packing_cell/outputs/packing_selector.pt --episodes 5
```

The evaluator compares the selector and scripted expert on identical
fresh starts, using the held-out jobs by default. On macOS, run the
following to watch the trained selector execute one held-out shipping job:

```sh
PYTHONPATH="../../.venv-single-arm/lib/python3.12/site-packages:../../.venv-mujoco/lib/python3.12/site-packages" \
  ../../.venv-single-arm/bin/mjpython -m packing_cell evaluate \
  --task ship_green_blue --episodes 1 --seed 21001 --live
```

The viewer shows the learned policy, and the terminal prints each call
as the selector makes it. The expert comparison runs separately without
a viewer. Omit `--task` to watch all four held-out jobs, or add
`--speed 0.5` to slow the live view.

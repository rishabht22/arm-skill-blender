# Single-arm skill composition

The experiments use one Panda arm in robosuite and MuJoCo 3.3.0, with
simulator state as the input to the skill selector.

- [Baseline workshop](baseline/README.md) contains the original single-cube
  experiments, three-workcell skill library, trained selector, and paired
  benchmark. Existing local recordings and outputs are under
  `baseline/data/raw` and `baseline/outputs`.
- [Interlocked packing cell](packing_cell/README.md) puts movable parts,
  a latch, a cargo drawer, and inspection stations in one workcell.
- [Connector service bench](service_bench/README.md) adds a polarized sensor
  socket, a physical hold-down keeper, and a horizontal power-cartridge rail.
- [Large-scale composition benchmark](larger_scale/README.md) defines the
  next experiment, with a new workcell and held-out task structures.

Run Python modules from this directory with the repository's existing virtual
environments. The baseline commands now start with `python -m baseline.workshop`.

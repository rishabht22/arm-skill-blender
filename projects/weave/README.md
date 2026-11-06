# WEAVE

WEAVE studies coordination in a native RoboFactory two-Panda GPU cell.
[Progress](PROGRESS.md) records the gates, and the
[Phase 0 audit](docs/phase0_audit.md) explains the replay and throughput protocol.

The [historical lab report](results/phase0/iconlab/report.md) includes the replay
precheck, benchmark results, and contact probes that informed backend selection.
To regenerate its Markdown from the archived measurements, install the
repository's `weave` extra and run this command from this folder.

```sh
python -m weave.cli report --results-dir results/phase0/iconlab
```

Use [the remote run guide](docs/REMOTE_RUN.md) for inventory, native replay, and
the Phase 1 cell smoke test. Phase 1 checks random-action rollout integrity.
The learned skill stages are pending.

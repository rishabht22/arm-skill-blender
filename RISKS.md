# WEAVE risks

| Risk | Owner | Mitigation |
| --- | --- | --- |
| The available SSH target was described as macOS. The pinned simulator requires Linux x86-64 with NVIDIA for the GPU audit. | WEAVE engineering | Check the OS and GPU from inside the SSH session. Run the pinned audit on a supported execution host and use the measured rates to choose the branch and compute budget. |
| Released demo metadata and current native code may reconstruct different trajectories. | WEAVE engineering | Compare raw replay to recorded joint traces, measure repeatability, and gate BC on success under the actual environment and controller. |
| Native table reset code shares sampled positions and may mishandle partial reset. | WEAVE engineering | Verify lane isolation and per-environment randomization before training. |
| The PlaceFood pot uses a sliding joint while WEAVE calls for a hinged lid. | WEAVE engineering | Keep the native comparison and WEAVE variant separate. Check physical validity and replay transfer before using demos. |
| Collision metrics could undercount contacts if link queries fail across lanes. | WEAVE engineering | Use a positive arm collision probe with independent unmodified lanes and validate the contact monitor before skill training. |
| WEAVE's hinged mesh and instrumentation have yet to run on the target GPU. | WEAVE engineering | Review the Phase 1 smoke trace and measure instrumented throughput before opening G1. |

# WEAVE decisions

- Use the pinned RoboFactory release candidate and its published ManiSkill and
  SAPIEN versions for the native audit. The recording metadata identify a
  different set of ManiSkill commits. Native open-loop replay measures the
  effect of version and scene differences on demo usability.
- Preserve the v1 partition algorithm and recorded episode order. Phase 0
  samples only its training episodes. Reserve demo validation and test data
  and object set B for the sealed final evaluation.
- Retain the original PlaceFood prismatic lid for the official task comparison.
  The WEAVE task requires a revolute lid. Reuse compatible asset geometry and
  validate the articulation and any BC transfer under the new contact dynamics.
- Test absolute joint replay and normalized joint-delta conversion on the same
  sampled episodes. The audit applies a raw success floor before marking BC as
  a candidate. Learned rollouts require the planner and IK restrictions in
  the proposal, including when choosing the simulator controller.
- Use an explicit retry action for the proposed silent-failure experiment.
  A claimed-done node stays masked for normal dispatch. Retry eligibility
  and its test policy must avoid revealing the physical success label.
- Keep environment branch selection open until the native replay, throughput,
  and contact reports are available. The lab GPU is the execution host for
  the audit, training, and evaluation. Repository tools own the data handling
  and artifact generation.

## Phase 0 environment selection

Select branch A based on the ICON Lab report. The native tabletop scenes
successfully ran GPU batches at every requested size and passed every contact
probe. Use the native RoboFactory environment infrastructure with WEAVE wrappers
and verify partial resets and per-lane randomization during cell development.

All native demo tasks failed the behavior-cloning precheck. Keep those demos
out of BC unless a corrected replay and controller conversion pass validation.
Prepare the initial training path for RL-only skills. The native throughput
measurements are planning inputs and must be remeasured with WEAVE instrumentation.

The reviewed report identifies Git `de1f3a492066485ddb1f78c1dee287dc2a5c4722`
and implementation `c7ce8eb5fca818d70e04f48b88c795c27db83f09b4298b0541c1f478c7ac100c`.
The stamped JSON and generated report were imported under `results/phase0/iconlab` and verified.

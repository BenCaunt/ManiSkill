# Native GPU controller and reset lifecycle

The source-bound lifecycle suite passes 23 cases: all eleven original arm
control modes on GPU Fill, six additional GPU task/control combinations, and
six CPU regressions. Each case starts at seed 101, advances two nonzero control
steps, saves dictionary and flat checkpoints, then compares one further step
against dictionary, flat and rebuilt-scene reset replays. Reset uses seed 17
before restoring the saved state. These are actual native tasks and controllers;
the probe uses proprioceptive state to construct nearby absolute targets.

| Task | Additional GPU control coverage |
|---|---|
| Excavate | Aligned end-effector pose delta |
| Hang | Aligned and target-relative end-effector pose deltas |
| Pour | Absolute end-effector pose |
| Write | Original unnormalized demonstration pose delta |
| Pinch | Target-relative end-effector position delta |

All native inverse-kinematics requests succeed. The independent host verifies
actual source, image and trace identities, recomputes the prescribed actions,
checks physical step accounting and robot motion, and compares saved flat
states with their dictionary representation. It then checks all recorded
particle, material, drive, controller and task checkpoint fields exactly.
All 51 GPU and 18 CPU reset trials pass those exact value checks. Excavate and
Pour restore a saved particle count different from the fresh reset recipe;
the saved counts are 11,056 and 8,534 respectively. The rebuilt-scene cases
verify replacement of the native physics-system object.

The joint-position replay limit is 1e-3, retained from the earlier CPU
controller lifecycle probe. GPU replay differences reach 2.0663e-5 and CPU
differences reach 7.6958e-5 in native joint coordinates. This is a lifecycle
stability gate, not a new reference-parity limit. Particle and joint-velocity
differences are recorded descriptively here: GPU maxima reach 0.0004908 m and
0.0007344 respectively; CPU maxima reach 0.0003410 m and 0.003475. The GPU
end-effector controller recomputes drive targets after the action with up to
7.1526e-7 difference; restored drive targets themselves are exact.

The separate [short task replay suite](gpu-tasks.md) still fails its original
strict gates for Fill, Excavate and Pour. No threshold was widened or replaced
by this lifecycle suite. Full episode stability, manipulation success,
reference physics parity and multi-environment task behavior remain unproven.

The worker completed all 23 cases and wrote its archive successfully, then its
shell wrapper exited 1 while trying to display a log path cleared by `read` at
EOF. That reporting failure is preserved separately. Saving the final log path
inside the loop fixes the wrapper; no physical case was rerun to conceal it.

Archive SHA-256:
`8c3e4673bc70ca7bdd9fa1fccf8e3940ebcf8b12230bff2eebb8e21b112d3c85`.
The frozen protocol, host verdict and wrapper failure are in
[`gpu-lifecycle-results.json`](gpu-lifecycle-results.json). The complete numeric
records remain in the outer workspace at
`artifacts/softbody/lambda/evidence-gpu-lifecycle-v1`. The host checker is
`softbody_lab/gpu_lifecycle_checks.py`; twelve synthetic corruption tests verify
the checker and do not count as physics evidence.

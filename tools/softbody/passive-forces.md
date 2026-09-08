# Native GPU robot compensation

`FixedBasePassiveForces` exports the actual native articulation to Pinocchio
after the reference masses, principal inertias, COM frames and joint frames
have been applied. Zero-acceleration inverse dynamics supplies gravity and
Coriolis/centrifugal compensation. Because the native exporter places the fixed
root at identity, the implementation first expresses gravity in the actual
root's frame. It does not normalize reported poses or change physical state.

`GPUPassiveForces` reads live native GPU positions and velocities and returns
the full system-shaped generalized-force array. `LegacyPassiveForceMixin`
supplies this baseline to the MPM world, which combines controller compensation
with particle reactions before applying forces once. The existing CPU agent
path retains SAPIEN's native passive-force calculation.

The current model requires a fixed root and one-DOF revolute or prismatic
joints. Construct it after GPU initialization and reset application. Rebuild
after changing root orientation, inertial parameters, joint frames, topology
or buffer allocation. Root orientation and index changes fail explicitly;
external mass/frame changes are not automatically detected. Per-link gravity
disabling is rejected. This remains a CPU-transfer correctness path.

## Independent experiment

`probe_passive_forces.py` uses seven collision-free fixed-base rigs: three
branched mechanisms, two original Hang Pandas and two original Fill buckets.
Roots are rotated, gravity is non-axis-aligned, and initial joint velocities
are nonzero. Original robot URDFs and separately exported inertial/frame
parameters are supplied externally; the diagnostic removes only visuals and
collision geometry. No robot mesh is bundled in this probe.

Three fresh processes evaluate 175 identical sampled force states and run
60 compensated physical steps of 2 ms. The CPU-native oracle does not import
the candidate force implementation. Host checks compare actual recipes,
initial states, source identities and recorded forces; constant-velocity and
linear-position predictions are computed independently from initial state.
The GPU rollout uses the real GPU buffer binding throughout.

| Version 6 check | Maximum absolute error | Frozen limit |
|---|---:|---:|
| Gravity force against native CPU | 4.7448e-5 | 2e-4 |
| Coriolis force against native CPU | 2.0894e-5 | 2e-4 |
| Combined passive force against native CPU | 4.9699e-5 | 2e-4 |
| GPU joint position against native CPU | 7.2318e-7 | 2e-5 |
| GPU joint velocity against native CPU | 9.7379e-6 | 2e-4 |
| GPU velocity drift from the constant-velocity prediction | 9.7752e-6 | 2e-4 |

Revolute coordinates use radians and torques; prismatic coordinates use metres
and forces. These mixed-coordinate maxima apply only to the identified rigs.
The force experiment passes; it does not establish collision dynamics, MPM
task parity or batched environment behavior. The task mixin was added after
this experiment and is covered separately by the actual task runtime suite.

The outer verification workspace retains all failed setup attempts, frozen
protocols, numeric traces and the independent checker. Version 1 failed on
an unsupported per-link sleep setting; version 2 assumed the wrong native
URDF-parser return shape; version 3 failed local preflight on a mismatched
robot parameter pack; version 4 was rejected before execution because its
archive included a generated directory. Versions 5 and 6 pass unchanged
numeric gates. None of these failures was removed from the evidence.

Version 6 archive SHA-256:
`61305320d5c07417b0b6ab0bbd8f04462fac0d6b22748c55fbb2c6fdafecb7ef`.
Version 5 archive SHA-256:
`6c415703de244a32d6c88476d1842dc764625f00444fa3b747c997cc32401d3e`.
The compact frozen protocols and host verdicts are in
[`passive-force-results.json`](passive-force-results.json).

# GPU rigid-body coupling investigation

The fork has a native GPU wrench projector, an explicit shared-world MPM
scheduler and fixed-base robot compensation. The six task prototypes now have
a single-scene GPU integration path. Actual task runtime and checkpoint replay
are recorded separately from full task parity; batched task lifecycle remains
incomplete. See the fork's `tools/softbody/gpu-tasks.md` for task-level results.

SAPIEN 3.0.3 on the existing A10 uses PhysX
`105.1-physx-5.3.1.patch0`. Its installed API exposes rigid-dynamic force and
torque application plus articulation generalized-force application (`qf`). It
does not expose direct GPU force application to an articulation link. This
matches the maintainers' [API discussion](https://github.com/haosulab/SAPIEN/discussions/257).
The API inventory was checked inside the actual immutable worker image.

For a fixed-base articulation, an external link wrench can instead be projected
into joint forces. For a revolute ancestor with world axis `a`, pivot `o`, link
COM `c`, force `F` and torque `T` about that COM, its contribution is
`a · (T + (c - o) × F)`. A prismatic ancestor receives `a · F`.
Contributions must be summed over descendant links and added to the controller's
existing generalized forces. This requires actual current GPU poses and correct
topology; writing a link row of the rigid-body force buffer is insufficient.

## Verified force-only experiment

`probe_gpu_forces.py` runs six collision-free one-DOF inertial-link scenes:
sliders and hinges, rotated roots, positive and negative loads, and two idle
scenes. Each loaded scene also receives a separate baseline joint force. Five
physics steps apply the load, followed by fifteen unforced steps, at 2 ms.
There is no MPM, robot controller, gravity, contact geometry, or rendering.

The worker runs three fresh processes: CPU physical link wrenches, CPU projected
joint forces, and GPU projected joint forces. `gpu_force_checks.py` independently
recomputes first-step velocities from the authored mass/inertia and force inputs,
then compares all trajectories and checks the idle scenes. The GPU test uses six
scenes in a shared native `PhysxGpuSystem` with explicit spatial offsets.

Version 1 failed before GPU simulation because it attempted state initialization
before GPU buffers existed. That failure and its completed CPU runs are retained.
Version 2 initializes GPU state after `gpu_init`, and explicitly enables PCM as
required by GPU PhysX. These collision-free rigs have no contact model affected
by PCM. Input physics and the predeclared numeric limits did not change.

Version 2 passed all checks:

| Check | Maximum absolute error | Declared limit |
|---|---:|---:|
| GPU first-step velocity against host analytic calculation | 7.3312e-8 | 1e-5 |
| CPU projected velocity against CPU physical wrench | 1.1921e-7 | 1e-5 |
| GPU projected velocity against CPU physical wrench | 5.3644e-7 | 1e-5 |
| GPU joint position against CPU physical wrench | 1.3039e-8 | 1e-6 |
| Idle scene position and velocity | 0 | 0 |

Prismatic coordinates use metres and metres/second; revolute coordinates use
radians and radians/second. These maxima combine the six explicitly identified
diagnostic coordinates; they are not task-level error bounds.

Local evidence: `artifacts/softbody/lambda/evidence-gpu-force-v2`, with frozen
protocol `gpu-force-v2-protocol.json` and host verdict `gpu-force-v2-verdict.json`.
The result archive SHA-256 is
`332b8bdf9bb04e2436489fb7d838b14de2b116f0847c9faa98c9106480596aff`.
The probe SHA-256 is
`2fb57ffba8b518a57dfd713f28df4f45a4891d134f1cb94f287c27a1c8572300`.

The optional GPU library came from SAPIEN's
[version-matched official release](https://github.com/sapien-sim/physx-precompiled/releases/tag/105.1-physx-5.3.1.patch0).
That release supplies no publisher digest. The first HTTPS download was pinned
and subsequent use verifies the recorded binary SHA-256
`4c582a16509a71faf5592fe9708586dfcc7ab61ae932eabc1ddd81f290818706`.
The archive SHA-256 is
`167a01aad7381afef963b89169968c289e7b653880a7a823c116d87ee5c00fc6`.
The source URL, asset ID and retrieval time are saved in
`artifacts/softbody/lambda/physx-gpu-manifest.json`. This runtime was added as a
read-only mount for the probe; the existing candidate image was not changed.

## Branched articulated forces

`gpu_wrenches.py` captures fixed-base articulation topology, including fixed
descendants and sibling branches. Current native GPU link poses supply world
COMs, axes and pivots. The adapter adds projected link wrenches to an explicit
complete baseline force buffer. It never writes robot positions or velocities.

`probe_gpu_articulations.py` tested six independent three-DOF trees with rotated
roots, nonidentity joint/COM frames, initial joint motion, fixed descendants,
tip-only/tool-only loads, a baseline-only tree and an idle tree. Fresh processes
used CPU physical link forces, CPU projected forces and GPU projected forces.
The CPU physical reference does not import the projector. Host verification
independently recomputed virtual power from native velocities and actual forces.

The `gpu-articulation-v1` record passed: GPU/CPU joint position difference was
9.3133e-10 (limit 2e-6), velocity difference 3.2783e-7 (limit 5e-5), and virtual
power residual 6.7954e-7 W (limit 2e-5 W). Unrelated-branch force, baseline-only
force error and idle motion were exactly zero. This is an inertial force probe,
not a robot manipulation or gravity-compensation test.

## Shared GPU world and actual MPM contact

`MPMGPUWorld` in `gpu_coupling.py` fetches native GPU rigid state, integrates each
attached MPM model, accumulates all reactions, applies combined free-body and
joint forces, and provides paired `prepare_step`/`complete_step` hooks around
the caller's single shared PhysX step. Its `step` convenience method owns that
step itself. Each model's state buffers rotate only after completion. A failure invalidates the
world and couplers. CPU transfers make this a correctness implementation;
throughput has not been measured.

Create the world after `gpu_init` and reset initialization, then register fresh
models. The caller must rebuild after topology, model-parameter or GPU-buffer
allocation changes. The current path does not implement partial reset or detect
all externally modified model parameters.

```python
world = MPMGPUWorld(system)
coupler = world.add_model(scene, model, states, bodies, mpm_dt=0.0005)
# Register other independent models before the first step.
details = world.step(
    before_physx=update_controller_targets,
    baseline_qf=current_complete_controller_force_buffer,
)
```

The baseline can be a callable evaluated after the controller callback. Supply
the entire system's joint-force array, including unrelated articulations; any
independent free-body forces likewise belong in `baseline_force` and
`baseline_torque`. Forces are cleared and accumulated afresh each step.
Fixed-base one-DOF revolute/prismatic joints are supported. Floating roots and
other joint types require additional implementation.

`probe_gpu_coupling.py` runs 100 rigid steps (400 MPM steps) in four scenes:
64-particle impacts on a free plate, slider and hinge, plus an idle neighbor.
The CPU comparison uses one independent PhysX system per scene. The GPU version
uses one system with scene offsets of 0, 4, 8 and 12 metres. All physical masses,
inertias, dimensions, materials and time steps are unchanged between modes.

Version 2 exposed missing slider/hinge contacts because the adapter subtracted
scene offsets twice. SAPIEN 3.0.3's [native fetch kernels](https://github.com/haosulab/SAPIEN/blob/3.0.3/src/physx/physx_system.cu)
already remove those offsets (free bodies at line 33, links at line 96).
Version 3 uses the native scene-local positions directly, restoring actual
contacts. The failed v2 record and original limits remain available.

| GPU v3 check | Maximum error | Original limit |
|---|---:|---:|
| Free-system linear momentum | 1.0389e-7 kg m/s | 2e-6 |
| Free-body measured impulse | 3.2783e-10 kg m/s | 2e-7 |
| Slider measured impulse | 3.5391e-10 kg m/s | 2e-7 |
| Hinge measured angular impulse | 7.5009e-11 kg m²/s | 2e-8 |
| Idle rigid velocity | 0 | 0 |
| Idle particle displacement | 1.1176e-8 m | 0 — fails |
| Idle particle velocity | 1.7683e-6 m/s | 0 — fails |

All three loaded bodies move 6–9 mm with nonzero measured MPM reaction forces.
The **overall verdict remains failed** because both CPU and GPU idle particles
exceed the exact-zero gates. CPU/GPU trajectories are descriptive only; their
maximum particle position difference is 3.2336e-6 m.

The source-audited host verifier checks the frozen input archive, actual probe,
request inventory, returned implementation files, worker image and trace hashes.
Seven verifier tests cover analytic inputs and reject forged inventories, changed images, falsely
reported native velocities, vacuous zero-force comparisons, shared idle drift
and nonfinite traces. The full host suite now passes 159 tests and 14 subtests,
including nine passive-force and ten task-runtime verifier checks.

Local evidence is `artifacts/softbody/lambda/evidence-gpu-coupling-v3`, with
`gpu-coupling-v3-protocol-source-audited.json` and
`gpu-coupling-v3-verdict-source-audited.json`. Archive SHA-256:
`38380d07b68287de562d2909bf8960a056d37a9c4b73694d8bb14f9c0bbdaf9a`.

## Original solver at rest

`probe_mpm_rest.py` runs the same stationary particle recipe with no SAPIEN,
collision shapes or coupling adapter. The pinned ManiSkill2 and candidate MPM
integration/model/simulator files are byte-identical. In separate immutable
containers, both produce byte-identical NPZ traces after 400 MPM steps:
1.1176e-8 m maximum position drift and 1.8392e-6 m/s maximum particle velocity.

The raw identity-matrix SVD yields a rotation with small float32 residuals.
The original elastic stress expression consequently produces a peak nonzero
stress of 7.1526e-4 Pa. Host float32 evaluation from the recorded rotation
reproduces the recorded stress exactly. This establishes an intrinsic numerical
contribution to rest drift; it neither calibrates acceptance thresholds nor
proves complete multi-scene isolation. Original zero-limit failures remain.

Evidence: `mpm-rest-v1-protocol.json`, `mpm-rest-v1-audit.json`, and
`evidence-mpm-rest-v1` under the same outer-workspace artifact directory.
Archive SHA-256:
`07cec1d84ff71c4497b796bfdab310ebb4942e7bf5f489e48a68343cbd79d696`.

## External stepping and robot compensation

Version 4 alternates the world's convenience `step` with the paired external
hooks around a real native GPU step. It also checks duplicate prepare/complete
rejection. Contact and impulse gates still pass: the GPU free-system momentum
residual is 1.0358e-7 kg m/s, with free-body, slider and hinge impulse residuals
3.0548e-10, 4.0792e-10 and 6.8197e-11 respectively in their declared units.
The unchanged exact-zero idle-particle gates still fail: displacement reaches
1.1176e-8 m and GPU velocity reaches 1.5934e-6 m/s. Overall verdict: failed.
Archive SHA-256:
`9111974491a708ec2aa827b2adf35333f2a3d777575444e8f941b24fe1ab8540`.

`FixedBasePassiveForces` uses the native articulation's exported Pinocchio
model, with world gravity expressed in the actual fixed root's frame.
`GPUPassiveForces` reads actual GPU joint state and returns a complete force
buffer for combination with MPM reactions. Seven collision-free mechanisms,
including the original Panda and bucket inertial recipes, pass 175 static
force comparisons and 60 online compensation steps against an independent
native-CPU oracle. See `tools/softbody/passive-forces.md` in the fork for
limitations, numeric limits and frozen evidence identities.

## Remaining implementation and evidence

Full GPU task episodes, broader controls, partial reset/checkpoints,
heterogeneous topology, GPU-buffer invalidation, batch isolation and performance
remain unverified. The short task suite retains replay failures for Fill,
Excavate and Pour. GPU PCM contact differences need reference comparisons.
Full reference parity and the existing unsuccessful task demonstrations remain
separate unresolved requirements.

The force probe uses the same GPU lock as the supervisor. Do not run older manual
simulation runners concurrently with either process.

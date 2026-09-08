# Experimental MPM bridge

This branch preserves ManiSkill 2 v0.5.3's MPM solver and adds a SAPIEN 3
coupling adapter and `MPMBaseEnv`. The tested configuration uses one CPU PhysX
scene with either CPU or CUDA MPM. This is an editable-checkout prototype.
`Fill-v0`, `Excavate-v0`, and `Hang-v0` now run with their legacy robot, material initialization, SDF contacts,
success/reward equations, and particle sphere visuals. Full reference parity is
still unverified. The other three task ports, GPU PhysX batching, and distributable
wheel packaging remain under development.

The copied runtime has separate terms in
[`warp_maniskill/LICENSE.md`](../../warp_maniskill/LICENSE.md). Those terms
include a non-commercial research/evaluation restriction; the repository's
Apache license does not replace them. Original and modified file hashes are
in [`PROVENANCE.json`](../../warp_maniskill/PROVENANCE.json).

## Run the probes

Use a ManiSkill development environment with SAPIEN 3.0.3. Build the bundled
runtime first; Linux CUDA builds need a CUDA 11.8 toolchain. Do not import a
different installed Warp before importing this integration.

```sh
python warp_maniskill/build_lib.py
python tools/softbody/probe_coupling.py --device cpu --output /tmp/mpm-contact-001
python tools/softbody/probe_coupling.py --device cuda --articulation hinge --output /tmp/mpm-hinge-001
python tools/softbody/probe_environment.py --device cuda --output /tmp/mpm-env-001
```

Each output directory must be new. Reports include simulation parameters and
raw numeric trajectories. The independent analytical gates cover free-body
linear momentum, reaction impulse, slider generalized impulse, hinge angular
impulse including the lever arm, and ballistic motion through ManiSkill's
control/simulation hooks. Joint probes account for the fixed base's constraint
forces; free-system linear momentum is not their acceptance criterion.

The lifecycle probe checks batched state observations, dictionary and flat
state replay through reset, seeded reset, scene reconfiguration, and rejection
of state assignment outside reset. Its particles are pressureless diagnostic
particles, not a validated material. No successful manipulation policy is
claimed by these probes.

## Task integration

The Fill prototype requires the assets from the pinned ManiSkill 2 v0.5.3 checkout.
They are not included here and are not downloaded implicitly. Their separate
CC-BY-NC-4.0 terms remain applicable. Robot inertias and joint frames are explicitly
preserved from the reference loader output, including the mesh-cooked bucket and
the missing-inertia default for the empty fixed link. These are reference
simulation inputs, not new physical measurements. The legacy PGS solver mode
is also explicit instead of inheriting ManiSkill 3's TGS default.
The beaker uses an open triangle-mesh rigid collider and a legacy visual SDF for
MPM. The bucket's rigid collider remains the legacy convex hull, separate from
the open SDF used by particles.

```sh
export MANISKILL_LEGACY_ASSET_DIR=/path/to/ManiSkill2/mani_skill2/assets
python tools/softbody/probe_fill.py --output /tmp/fill-001
python tools/softbody/probe_fill_lifecycle.py --output /tmp/fill-lifecycle-001
python tools/softbody/probe_fill.py --env-id Excavate-v0 --output /tmp/excavate-001
python tools/softbody/probe_fill_lifecycle.py --env-id Excavate-v0 --output /tmp/excavate-lifecycle-001
```

Import `mani_skill.envs.softbody.fill` explicitly to register `Fill-v0` with Gym.
Import `mani_skill.envs.softbody.excavate` for `Excavate-v0`. Both reuse
`LegacyBucketEnv` and `LegacyPandaBucket`; Excavate retains the original Perlin
terrain recipe, four wall colliders, target particle count, and reward/success
equations. Its seed-101 initialization has 11,056 particles, 4.146 kg total mass,
and a target of 1,113 lifted particles. These are benchmark simulation inputs.
The current mesh extraction needs a SAPIEN renderer even for state-only Fill
rollouts. The recorded task runs use Linux, CUDA MPM, and CPU PhysX. Cameras use
visual-only sphere entities at actual particle positions, with no physics or
state-registry component. This supports color/depth/segmentation but has per-particle
CPU update overhead; it is not yet a batched rendering implementation.

The Fill probe checks particle count/mass, outward ground normal, finite state,
and broad robot stability bounds. It records every state for external comparison
and always reports `parity_validated: false`: completing this probe is not a
reference-parity verdict. The lifecycle probe uses nonzero actions and a
target-relative controller, restores dictionary and flat checkpoints, and checks
reconfiguration, controller memory, drive targets, and actual particle camera
pixels. It reports particle replay differences separately from joint/target checks.

Subclass `mani_skill.envs.softbody.base_env.MPMBaseEnv`, load rigid bodies using
ordinary ManiSkill scene construction, and call `rebuild_mpm(builder, bodies)`
inside `_initialize_episode` after resetting rigid poses. Use
`register_collision_body` for supported metric primitive colliders or
`geometry.register_visual_body` for triangle-mesh SDFs. The latter preserves the
legacy grid sampling and face-normal convention. Unsupported shapes fail
explicitly; open containers must retain their collision cavity.
Override `_configure_mpm_model` for contact settings. Task simulation hooks must
call their superclass so coupling runs once per PhysX timestep.

`MPMCoupler.prepare_step` integrates MPM, averages its substep wrenches, and
applies world-frame force and torque about each body's COM. ManiSkill then steps
PhysX. `complete_step` publishes the final MPM buffer. Runtime stepping never
assigns robot or object poses. GPU MPM currently transfers state/wrenches through
the CPU; this is not a GPU PhysX adapter or a throughput claim.

Reset replay supports a changed particle count for uniform-color particle recipes,
with fixed rigid/task topology and the same solid/fluid state layout. Dictionary
checkpoints determine the count from their particle arrays; flat checkpoints use
the current field schema and vector length. Counts must be positive and at most
65,536 (or the task's smaller observation capacity). Heterogeneous visual colors,
changed rigid models and changed goal-array shapes require a separate checkpoint
contract. This does not introduce resampling: every saved particle and material
value is restored at its original index.
Reset builds the model before restoring state, including after reconfiguration.
Checkpoints contain all five particle material arrays, controller memory, and native drive targets; restoring them
does not invoke a controller reset afterward. The current schema requires these
fields for robot checkpoints. Flat checkpoints preserve float64 task inputs;
state observations keep their normal observation dtype. Camera observations are
recomputed after restore.
The lifecycle probe changes the reset recipe's density before restoring a checkpoint
and checks that the saved masses and material arrays replace the new recipe.
`probe_checkpoint_topology.py` additionally changes a pressureless recipe between
27 and 64 particles, tests dictionary/flat/rebuilt-scene resets on CPU and CUDA,
and compares subsequent free fall. The old implementation fails at its MPM shape
check; the fixed implementation restores values exactly and gives zero measured
position difference in all twelve replay trials. Malformed flat layouts and
unsupported material types are rejected. Pour's lifecycle probe also changes
the reset seed so its fresh particle count differs from the saved checkpoint.
The September 8 Pour trial restores 8,534 saved particles after a seed-1 reset
creates 8,032. Particle/material/drive/task values restore exactly. Subsequent
one-control replay differs by at most 0.000142 m in particle position,
1.23e-10 m3 in current volume and 3.13e-5 rad in joint position; this is lifecycle
evidence, not reference parity. Fill, Excavate and Hang lifecycle regressions pass.

`mani_skill.envs.softbody.capture.CaptureAdapter` exposes both bucket tasks and Hang to the
independent `softbody_lab` recorder. Version 2 fixtures use explicit physical actor, fixed-root,
joint, controller, task, and particle fields for portable reset. Each engine's
actual native state buffer is also recorded, but its serialization is not assumed
to match another engine's. The adapter does not import ManiSkill 2 or reference
trajectories. Replay assignments go through the native reset path.

The September 8 portable Fill check used three independent reference replays of
one seed-101, one-action fixture. Input hashes matched and all trajectory/reward
limits passed. The overall strict verdict **failed**: derived initial position
and quaternion component differences were 3.58e-7 m and 1.79e-7, above the existing
1e-8 limits. These limits were not widened. This result establishes working
capture/replay, not full reference parity or successful manipulation.

The corresponding Excavate fixture also matched its portable input hash and all
one-step trajectory/reward limits. Its strict verdict failed the same two initial
derived-pose gates (3.71e-7 m and 1.79e-7). A separate 20-control no-op capture
completed with finite states, matching reference joint positions, and zero spilled
particles. It is a settling diagnostic, not a calibrated full-episode parity pass
or proof that the robot can scoop material successfully.

The official seed-1, 231-control Excavate demonstration was subsequently replayed
in both engines with identical portable input hashes (12,115 particles,
4.543125 kg, target 772). The reference succeeds at step 230 and finishes with
791 lifted particles and one spilled. The port lifts 763 but spills 31, failing
the unchanged requirement of fewer than 20 spilled particles; the amount and
settling checks pass. Independent NumPy recomputation agrees with every recorded
success label. Maximum joint error is 0.000452 rad and COM separation 0.001628 m,
while particle-identity separation reaches 0.268717 m. This unsuccessful transfer
is retained for diagnosis; reference repeatability and full-episode aggregate
parity are not established. The source demonstration is episode 0 at dataset
revision `0c367447d26e4e2de13fbf5e5d2ab09a258187da`, with HDF5 SHA-256
`4a6baa93d40d82cedf54ee7ae28add84d84aa90373f1f5fdb622fe3f54387b8b`.

Excavate reward membership uses the first actual rigid collision hull, as in the
legacy evaluator, separately from the open visual SDF used for particle contacts.
That hull is cooked by the native PhysX version; exact cross-version hull and
reward membership equivalence under manipulation remain to be verified.

## Hang and the numeric reference pack

Hang preserves the original 3,636-particle rope, 0.0698112 kg mass, five evaluation
indices, randomized rod, and ten recorded grasp starts. Its native Panda uses
the legacy URDF and exported SAPIEN 2 mass/COM/inertia/joint frames. Twelve bodies
have MPM collision geometry: nine mesh SDFs, two fingers with primitive colliders,
and the rod. The recorder additionally observes the collision-free links, for all
13 Panda links plus the rod.

Create the numeric pack inside the pinned ManiSkill 2 reference environment
(SAPIEN 2.2.2, NumPy 1.23.5, legacy Warp built with CUDA). The reference checkout
must be clean at `493be36121a9dd06071a57172274babe617b789f`, with its original
assets and license files; its `HangEnv.sdf` cache must be writable if stale.

```sh
python /path/to/this/fork/tools/softbody/export_hang_assets.py \
  --source /path/to/ManiSkill2 --output /path/to/new-hang-pack
```

The verified pack's `export.json` SHA-256 is
`bd9a93ad23798da1b34bbb6b64f4bd2470a4740cbfbbec519ee5b584720db431`.
The exporter reproduced this exact manifest independently. All NPZ hashes are
pinned by it; the pack also includes provenance and original notices. Run the
native task in the separate ManiSkill 3 environment:

```sh
export MANISKILL_LEGACY_MPM_DATA=/path/to/new-hang-pack
export MANISKILL_LEGACY_ASSET_DIR=/path/to/ManiSkill2/mani_skill2/assets
python tools/softbody/probe_hang.py --output /tmp/hang-001
```

Import `mani_skill.envs.softbody.hang` to register `Hang-v0`. The reset restores
the recorded grasp after ManiSkill's controller reset, preserving the legacy
nominal controller memory and native drive targets. Explicit user checkpoints
take precedence. Particle/robot assignments occur only during reset.

The original recorded joint acceleration is retained as diagnostic data. In the
tested SAPIEN 3.0.3 runtime, setting `qacc` does not restore that native readback;
the port reports the actual value instead. Acceleration is not in the original
environment's serialized state, observation, or PD controller input. This is an
observable compatibility limitation, not a claim of identical native APIs.

The seed-101 native run completes 20 zero-action controls and dictionary/flat
checkpoint replay with finite states. Maximum particle replay difference was
8.94e-8 m; joint differences were at most 2.39e-7 after the gripper correction.
The zero-action run does not solve the task.
Three independent one-control reference replays calibrate the portable fixture.
Its exact input hash matches, but **strict parity fails**: initial derived pose
and velocity gates fail, and trajectory differences include 3.51e-6 m particle
positions versus a 1e-6 m limit, and 7.82e-5 joint velocity versus 1e-5.
The initial implementation had an unintended SAPIEN 3 fixed tendon between the
fingers. A separate gravity-free, zero-drive two-slider URDF probe showed that
the new loader moved asymmetric fingers together, while SAPIEN 2 left them
stationary. Disabling `build_mimic_joints` reproduced the reference probe; the
task now uses that setting and retains the original paired PD drive targets.
Against the same frozen fixture and calibration, joint-velocity error fell from
7.82e-5 to 2.76e-7 and particle-position error to 3.58e-7 m. The strict verdict
still fails four initialization gates and two trajectory gates: particle
velocity 1.63e-5 versus 1e-5, and affine velocity matrix 0.001017 versus 0.001.
The original failure remains recorded. Reward, mass, drive targets, rod state,
and evaluation indices meet their limits. The limits were retained.
Broader physical manipulation coverage, additional seeds,
full-episode aggregate gates, and camera/checkpoint coverage beyond the current
probe still require verification.

## Physical demonstration replays

The official dataset revision `0c367447d26e4e2de13fbf5e5d2ab09a258187da` contains
recorded actions and one initial state per episode. Its first Fill and Hang
demonstrations were freshly replayed in the pinned reference engine, then exported
as portable initialization/material/controller/action fixtures for the port.
Reference outcomes were excluded from the candidate containers. Every subsequent
state came from drive targets and physical contacts; no state correction occurred
between actions. These are recorded reference controls, not a vision policy.

| Task / seed | Controls | First success, reference / port | Final result |
| --- | ---: | ---: | --- |
| Fill / 1 | 176 | 156 / 161 | Both successful; reference contains 701/704 particles, port 637/704 |
| Hang / 3 | 338 | 237 / 237 | Both successful with released fingers and rope supported above ground |
| Pour / 1 | 299 | 297 / 297 | Both successful; reference contains 3,842 particles with 20 spilled, port 3,957 with 1 spilled |

An external NumPy evaluator recomputed the legacy outcome predicates from every
recorded state and matched all reference and candidate labels. Counts and masses
stay constant and all recorded values are finite. The port's Fill final contained
mass is 0.122304 kg versus the reference's 0.134592 kg; both report zero spill
under the original spill definition (which does not count particles still in the
bucket). Passing the task threshold does not establish matching transfer quality.

No full-episode parity protocol has been calibrated. Maximum particle-identity
position differences over these episodes are 0.1573 m for Fill and 0.0920 m for
Hang; maximum COM distances are 0.0132 m and 0.0171 m. Both successful actions
and these substantial trajectory differences remain part of the evidence.

## Native Pour prototype

Import `mani_skill.envs.softbody.pour` to register `Pour-v0`. The task keeps the
original fluid recipe, contact parameters, sampled viscosity/density, rewards,
success predicates and IK draw order. Linux SAPIEN 3.0.3 supplies the tested
Pinocchio runtime. Fresh seed-101 particles match the reference exactly (8,534
particles, 0.245670289 kg); fresh IK joint positions differ by up to 1.92e-5 rad.
Portable replay restores the explicitly recorded independent robot inputs.

The trusted SAPIEN 2.2.2 reference exporter and optional local preparation tool
produce restricted external model data. Neither is a runtime dependency:

```sh
# Inside the pinned reference environment; its SDF cache must be writable.
python tools/softbody/export_pour_assets.py --source /path/to/ManiSkill2 --output /tmp/pour-pack
# In a separate preparation environment with coacd==1.0.14, trimesh, scipy, rtree.
python tools/softbody/prepare_pour_collision.py --pack /tmp/pour-pack --output /tmp/pour-pack/collision
# Inside the native ManiSkill 3 environment.
export MANISKILL_LEGACY_MPM_DATA=/tmp/pour-pack
export MANISKILL_LEGACY_ASSET_DIR=/path/to/ManiSkill2/mani_skill2/assets
python tools/softbody/probe_pour.py --output /tmp/pour-native-001
```

The model manifest SHA256 is
`54d4c40bdfe3184788d5e1f1841c8806611cff1249f0aad8b5c9a150ab982710`;
the bottle collision NPZ SHA256 is
`6f2f689d65dcbb036e407b2350a4aad10d5ce47f8f17f24551628d4bf9fc593e`.
Candidate runs receive only `export.json`, the two `body-*.npz` files, the
collision pack and original notices/provenance, plus original mesh assets.
Separate initial-readback diagnostics stay with the verifier.

Both containers retain the original open fluid SDFs. Their rigid collisions use
an open bottle decomposition and an open beaker triangle mesh instead of the
original closed hulls. Bottle mass, COM and inertia remain the original explicit
simulation inputs. All 128 native cooked bottle components pass a 6 mm radius
clearance probe through the cavity. CoACD hit its hull limit and warned that the
requested 0.5 mm concavity threshold was not attained. This is a documented
geometry approximation, not verified exterior-contact equivalence. The reward's
geometric targets retain the exported reference bounds.

The 20-control native diagnostic stays finite. Dictionary, flat and rebuilt-scene
checkpoints restore current fluid volume separately from rest volume, including
the actual volume-correction buffer and both target heights. Negative current
volumes are rejected. Subsequent replay differences reach 0.000272 m in particle
position and 1.91e-10 m3 in volume; these are lifecycle measurements, not reference
parity limits. The legacy native serializer omitted the second fill height; the
portable contract records both actual heights explicitly.

Three independent reference replays match the exact portable reset inputs after
reset-only compensation for a SAPIEN 2 body/COM translation roundoff. The original
15-nanometer reset failure remains recorded. Calibration still fails: reference
affine-velocity variability reaches 1.05245 versus the predeclared ceiling 1.0.
No Pour parity protocol was promoted. Candidate one-control differences include
0.0001583 m particle position, 5.57e-10 m3 current volume and 0.002069 rigid
velocity. Full-episode aggregate behavior needs separate calibration.

The official seed-1 Pour demonstration completes physically in both engines and
passes every original success predicate from control 297 through 299. An external
NumPy evaluator recomputes the predicates from every frame and matches all labels.
Final contained mass is 0.0737596 kg in the port versus 0.0716159 kg in the
reference. Counts and masses remain constant and all recorded values are finite.
Maximum full-episode particle-identity separation is 0.6659 m, COM separation
0.003425 m, current-volume difference 9.72e-7 m3, joint position difference
0.004738 rad and joint velocity difference 0.1154 rad/s. Success in this one
recorded demonstration does not establish trajectory or real-fluid fidelity.

Pour's state observations pad particle arrays to 16,384 entries and expose the
live `particle_count`, so changing the sampled fluid height does not change the
observation space. Padded rows are inactive zeros and never become physical
particles. Checkpoints retain only actual particles and both fill heights at
their original precision; observations follow native float32 conventions. Dictionary
and flat observation spaces were checked at seeds 1 and 101 with 8,032 and 8,534
live particles respectively. Existing Fill, Excavate and Hang lifecycle probes
also pass after the shared fluid-state changes.

The pinned collision pack was prepared on macOS ARM64 with CoACD 1.0.14,
NumPy 2.2.6, SciPy 1.18.1, trimesh 5.1.0 and rtree 1.4.1. Cross-platform
bitwise reproducibility of decomposition is not established; the runtime checks
the fixed pack checksum, and the GPU worker separately checks its cooked cavity.

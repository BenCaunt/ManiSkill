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

Reset replay currently assumes the same particle topology.
Reset builds the model before restoring state, including after reconfiguration.
Checkpoints contain all five particle material arrays, controller memory, and native drive targets; restoring them
does not invoke a controller reset afterward. The current schema requires these
fields for robot checkpoints. Flat checkpoints preserve float64 task inputs;
state observations keep their normal observation dtype. Camera observations are
recomputed after restore.
The lifecycle probe changes the reset recipe's density before restoring a checkpoint
and checks that the saved masses and material arrays replace the new recipe.

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
8.94e-8 m; joint differences were below 1.87e-9. It does not solve the task.
Three independent one-control reference replays calibrate the portable fixture.
Its exact input hash matches, but **strict parity fails**: initial derived pose
and velocity gates fail, and trajectory differences include 3.51e-6 m particle
positions versus a 1e-6 m limit, and 7.82e-5 joint velocity versus 1e-5.
Reward, mass, drive targets, rod state, and evaluation indices meet their limits.
The limits were retained. Successful physical manipulation, other seeds,
full-episode aggregate gates, and camera/checkpoint coverage beyond the current
probe still require verification.

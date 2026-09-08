# Experimental MPM bridge

This branch preserves ManiSkill 2 v0.5.3's MPM solver and adds a SAPIEN 3
coupling adapter and `MPMBaseEnv`. The tested configuration uses one CPU PhysX
scene with either CPU or CUDA MPM. This is an editable-checkout prototype.
`Fill-v0` now runs with its legacy robot, material initialization, SDF contacts,
success/reward equations, and particle sphere visuals. Full reference parity is
still unverified. The other five task ports, GPU PhysX batching, and distributable
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
```

Import `mani_skill.envs.softbody.fill` explicitly to register `Fill-v0` with Gym.
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

Reset replay currently assumes the same particle topology and material model.
Reset builds the model before restoring state, including after reconfiguration.
Checkpoints contain controller memory and native drive targets; restoring them
does not invoke a controller reset afterward. The current schema requires these
fields for robot checkpoints. Camera observations are recomputed after restore.

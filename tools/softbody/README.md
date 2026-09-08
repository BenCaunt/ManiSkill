# Experimental MPM bridge

This branch preserves ManiSkill 2 v0.5.3's MPM solver and adds a SAPIEN 3
coupling adapter and `MPMBaseEnv`. The tested configuration uses one CPU PhysX
scene with either CPU or CUDA MPM. This is an editable-checkout prototype.
The six legacy tasks, GPU PhysX batching, particle rendering, and distributable
wheel packaging are still under development.

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

Subclass `mani_skill.envs.softbody.base_env.MPMBaseEnv`, load rigid bodies using
ordinary ManiSkill scene construction, and call `rebuild_mpm(builder, bodies)`
inside `_initialize_episode` after resetting rigid poses. Use
`register_collision_body` for supported metric primitive colliders. Unsupported
meshes fail explicitly; open containers must retain their collision cavity.
Override `_configure_mpm_model` for contact settings. Task simulation hooks must
call their superclass so coupling runs once per PhysX timestep.

`MPMCoupler.prepare_step` integrates MPM, averages its substep wrenches, and
applies world-frame force and torque about each body's COM. ManiSkill then steps
PhysX. `complete_step` publishes the final MPM buffer. Runtime stepping never
assigns robot or object poses. GPU MPM currently transfers state/wrenches through
the CPU; this is not a GPU PhysX adapter or a throughput claim.

Reset replay currently assumes the same particle topology and material model.
Reconfiguration must build the model before restoring state. Visual observations
are not advertised until particle rendering is connected.

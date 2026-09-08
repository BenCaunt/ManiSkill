# Experimental shared-world Fill tasks

`Fill-v0` now supports multiple environments in one native SAPIEN GPU physics
system. Each scene owns its original particle model, state buffers, colliders,
controller state and visual pool. One `MPMGPUWorld` schedules all MPM models
before ManiSkill advances the shared PhysX system once. This is a correctness
implementation: MPM models still execute separately and transfer body state and
wrenches through CPU memory. No throughput improvement is claimed.

The tested two-environment configuration uses seeds 101 and 17 with different
nonzero joint-target actions. Each control performs 25 shared 2 ms rigid steps,
with four 0.5 ms MPM substeps per model per rigid step. Each initial model has
704 real particles. Separate one-environment runs use the same seeds, actions
and physical inputs. The frozen v2 suite completes all three cases, but its
overall verdict remains **FAIL** because particle trajectories exceed the
original 10 micrometre comparison limit. Full reference parity is unverified.

## Use

Build the bundled Warp runtime and supply the separately licensed legacy
assets as described in [README.md](README.md). On the tested Linux CUDA/SAPIEN
3.0.3 environment:

```python
import gymnasium as gym
import torch
import mani_skill.envs.softbody.fill

env = gym.make(
    "Fill-v0", num_envs=2, sim_backend="physx_cuda",
    control_mode="pd_joint_target_delta_pos", obs_mode="state_dict",
    mpm_batch_particle_capacity=2048,
).unwrapped
obs, info = env.reset(seed=[101, 17])
action = torch.tensor([
    [.02, -.04, .01, .03, -.01, .03, -.02],
    [-.01, .02, -.005, -.015, .005, -.015, .01],
], device=env.device)
obs, reward, terminated, truncated, info = env.step(action)
checkpoint = env.get_state_dict()

def select(value, index):
    return ({k: select(v, index) for k, v in value.items()}
            if isinstance(value, dict) else value[index:index+1].clone())

env.reset(seed=[33], options={
    "env_idx": torch.tensor([1], device=env.device),
    "reset_to_env_states": {"env_states": select(checkpoint, 1)},
})
env.close()
```

Particle fields have a fixed leading shape `(num_envs, capacity, ...)`.
`mpm_meta.count` has shape `(num_envs, 1)` and `mpm_meta.mask` identifies the
live prefix. Inactive rows are zero. Counts and exposed values come from actual
solver particles; capacity does not add particles to the physics model. Fill's
default capacity is 4096. The existing single-environment layout and CPU default
remain unchanged; multiple environments require CUDA MPM and GPU PhysX.

Dictionary and flat checkpoints preserve particles, per-particle material,
controller memory, drive targets, native rigid state and task targets. A partial
checkpoint contains rows for only the selected environments. Restoring a changed
particle count rebuilds the selected MPM model within the declared capacity and
preserves untouched models' current state-buffer order. Uniform particle colors
and the existing solid/fluid layout are required. The 704-to-352 test is a reset
diagnostic, not a modified benchmark episode or a successful manipulation.

Reconfiguration requires a full reset. A failed shared physics world also requires
a full reset; resetting one scene cannot silently revive a failed neighbour.
Reset cannot interrupt a pending shared step. Invalid partial reconfiguration is
rejected before changing a live world.

## Reset and rendering evidence

The v1 run found that native root poses changed slightly in environments that
were not reset, even though their particles and controller targets stayed fixed.
The generic reset loop reapplied every root pose and introduced extra rounding.
The scene adapter now uses native indexed rigid/root apply calls for selected
environments. Controller reset retains the selection mask until its target
memory has been updated. Explicit checkpoint selection also reaches native apply.

SAPIEN 3.0.3's indexed joint-buffer methods ignore the supplied indices and use
the internal index-buffer prefix. The adapter therefore keeps full joint-buffer
calls with untouched rows preserved. See the pinned
[native implementation](https://github.com/haosulab/SAPIEN/blob/3.0.3/src/physx/physx_system.cpp).
The tests establish exact exposed state isolation; they do not prove that every
internal PhysX solver cache survives a partial reset unchanged.

All seven v2 checkpoint comparisons pass exactly: three untouched-environment
checks, selected dictionary restore, selected count restore and both flat-restore
rows. Native masses, inertias, COMs and joint frames match the separate N1 runs
and survive full reconstruction. All actual counts, padding and live masks match
the solver. All ten camera frames pass the existing physical-particle and robot
bounds checks, including both scene offsets and the partial count change.
Rendering leaves native rigid buffers and checkpoints unchanged.

The v2 particle-position errors are 0.1253 mm for environment 0 and 0.01964 mm
for environment 1 after three controls. The original limit is 0.01 mm. Partial
continuation and flat replay also retain trajectory failures. These errors are
reported independently from the successful reset and rendering checks.

The v3 diagnostic keeps the v2 runtime byte-identical and runs N1 scenes at the
same native origins as the two batched scenes. Their initial robot visual
transforms then match exactly; particle trajectories still differ by 0.1182 mm
and 0.02176 mm after three controls. Two additional N1 runs with identical code,
seed, origin and actions differ by 0.04714 mm at that point. Thus the original
10 micrometre gate also fails a self-repeat. These measurements establish a
repeatability limitation, not its complete cause or an acceptable replacement
tolerance. The v2 verdict remains FAIL. Broader repeated reference/port runs and
task-outcome measurements are needed before declaring behavioural equivalence.

## Reproduce the host verdict

The host verifier reads numeric NPZ records with `allow_pickle=False`, verifies
frozen source/request/probe/image identities and judges actual solver state and
camera data. It never executes the candidate when comparing results.

```sh
python -m tools.softbody.verification.gpu_batch_checks RECORD \
    tools/softbody/verification/protocols/gpu-batch-v2.json VERDICT.json
python -m tools.softbody.verification.gpu_origin_checks V3_RECORD \
    tools/softbody/verification/protocols/gpu-batch-v3.json V2_RECORD \
    tools/softbody/verification/protocols/gpu-batch-v2.json V2_ARCHIVE.tgz DIAGNOSTIC.json
pytest tools/softbody/test_batch_runtime.py tools/softbody/test_gpu_reset_selection.py \
    tools/softbody/test_render_only_poses.py tools/softbody/verification
```

`gpu-batching-results.json` retains protocols, verdicts and raw archive hashes,
including the failed v1 checks. Raw archives, built libraries and restricted
assets remain external to Git. The latest probe is `probe_gpu_batch.py`; use the
probe hash recorded by the chosen protocol when reproducing a historical run.

Only Fill currently opts into the batched task lifecycle. Legacy end-effector
controllers now use [independent per-environment IK and target memory](gpu-controller-batching.md);
the other five task classes still reject multiple environments. Full episodes, reference parity, heterogeneous
visual checkpoints, changed rigid topology, efficient rendering and distribution
packaging remain incomplete.

# Experimental shared-world Hang

`Hang-v0` supports multiple native GPU environments. Each row retains the
original 3,636-particle rope, material parameters, seeded rod pose and recorded
grasp. Robot mass, inertia, centers of mass and joint frames come from the
pinned reference pack for every native articulation. The loader continues to
disable the additional SAPIEN 3 mimic tendon used by its default URDF path.

Use the Linux CUDA setup and separately licensed Hang numeric pack described
in [README.md](README.md), then select:

```python
import gymnasium as gym
import mani_skill.envs.softbody.hang

env = gym.make(
    "Hang-v0", num_envs=2, sim_backend="physx_cuda",
    control_mode="pd_ee_target_delta_pose", obs_mode="state_dict",
    mpm_batch_particle_capacity=8192,
).unwrapped
obs, info = env.reset(seed=[101, 17])
```

The default batch capacity is 8,192 particles per row. Only the live prefix
participates in physics. Checkpoints include counts, masks, particle/material
state, robot/controller state, rod pose and five rope evaluation indices per
environment. The existing N1 checkpoint layout remains unpadded.

Initialization preserves the reference's ordering: reset the robot and
controller to their nominal targets, build the rope and select its evaluation
particles, then restore the recorded grasp after the normal controller reset.
An explicit user checkpoint takes precedence over this deferred grasp.
Partial reset applies the selected checkpoint rows and preserves the exposed
state of the other environment. State assignment is restricted to reset;
controllers act through physical joint drives.

The [Fill example](gpu-batching.md) shows the shared checkpoint selection API.
When changing particle counts, evaluation indices must refer to the saved
live particles. The verifier's reduced-count reset keeps the original five
evaluation particles and remaps their indices; it is a reset diagnostic with
changed material quantity, not a benchmark episode.

## Verification

The frozen v8 suite passes four native cases: two independent N1 seeds and
two N2 modes, `pd_ee_target_delta_pose` and `pd_ee_delta_pose_align`. Each row
uses its own native IK model, complete nine-joint starting configuration,
seven active arm joints and separate finger command. Seeds 101 and 17 select
recorded grasps 3 and 0. Initial particles, materials, joint/drive state and
evaluation indices match their independent N1 cases exactly; N1 seed 101 also
matches the older frozen implementation. Native rod pose comparison uses the
predeclared 1e-6 component limit for float32 scene-origin composition.

All 14 exact checkpoint comparisons, 16 native camera frames and 32 IK calls
pass. Checkpoint comparisons include selected actor poses and untouched
environment state. The selected particle count changes from 3,636 to 1,821
while the other environment retains 3,636 particles. All 36 task snapshot files
pass independent success and dense-reward equations reconstructed from numeric
particle positions/velocities, rod/finger poses and gripper joint state.

After saved-state replay, joint-position differences reach 4.7684e-7 radians,
within the existing 1e-3 lifecycle gate. Particle-position differences reach
0.3875 micrometres and remain descriptive. Native camera geometry retains the
existing 3 mm bound; visual inspection also checked the original grasp and
reduced-count frames. These short controls do not demonstrate a successful
hanging episode or establish reference trajectory parity.

The shared Panda helper also passes nine N1 CPU/GPU regression cases across
Hang, Pour, Write and Pinch, with all 27 dictionary, flat and rebuilt checkpoint
restores exact. These use the unchanged lifecycle-v2 probe and acceptance
gates. The largest joint replay difference is 7.5043e-5 radians in CPU Pour;
particle differences reach 0.4769 mm in GPU Pour and remain descriptive.

Two N2 Fill/Excavate regressions with the same runtime pass another 14 exact
checkpoint comparisons, 12 camera frames and 24 IK calls. Excavate's 12
snapshot files also pass its independent outcome/reward checks. These exercise
the shared reset path when no deferred initial grasp is supplied.

The [machine-readable results](gpu-hang-batching-results.json) retain the
frozen protocols, source/input/output hashes and independent verdicts.
The portable checker reproduces the saved verdict without importing the
candidate simulator:

```sh
python -m tools.softbody.verification.gpu_hang_batch_checks \
    RECORD tools/softbody/verification/protocols/gpu-batch-v8.json \
    OLD_LIFECYCLE_V2_RECORD PINNED_HANG_PACK VERDICT.json
```

Local synthetic fault tests exercise swapped environment state, invalid goal
indices, incorrect gripper commands, altered rewards and count-reduction
corruption. They are verifier contract tests, not physical evidence.

Full batched episodes and repeated reference comparisons remain incomplete.
The previous strict FK/particle failures remain failures. Pour, Write and
Pinch batching, official Write/Pinch benchmark validation, packaging and robust
interrupted supervisor recovery also remain under development. Shared native
physics still involves CPU transfers and IK; no throughput claim is made.

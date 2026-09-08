# Experimental shared-world Excavate

`Excavate-v0` uses one native GPU world with an independent granular terrain,
particle model, robot controller and target count per environment. Each row
preserves the original random draw order: target count, robot perturbation,
Perlin terrain and particle jitter. The task retains the original wall dimensions,
bucket visual SDF, granular material parameters and reward equations. The first
actual cooked bucket collision hull remains the separate reward-bound definition.

The default batch capacity is 32,768 particles per environment. Padding is not
part of the physical model. Checkpoints store live counts, masks, particles,
materials, controller/drive state and one float64 target count per row. Partial
reset restores only the selected task targets and particle models; a count-changing
reset rebuilds that model. Full scene reconstruction rebuilds every native model.
The existing N1 scalar target and unpadded checkpoint layout are preserved.

Use the Linux CUDA environment and separately licensed assets described in
[README.md](README.md), then select:

```python
import gymnasium as gym
import mani_skill.envs.softbody.excavate

env = gym.make(
    "Excavate-v0", num_envs=2, sim_backend="physx_cuda",
    control_mode="pd_ee_delta_pose_align", obs_mode="state_dict",
    mpm_batch_particle_capacity=32768,
).unwrapped
obs, info = env.reset(seed=[101, 17])
```

The [Fill example](gpu-batching.md) shows the shared checkpoint selection API.
`options["target_num"]` may provide a positive integer or one target for each
selected batch row. Controllers use physical joint drives, and only reset
assigns simulation state. IK and MPM body transfers still involve CPU work;
this implementation makes no throughput claim.

## Verification and retained failure

The independent host checks actual native ownership, timestep accounting,
live particle arrays, exact checkpoint fields and camera geometry. It reconstructs
Excavate outcomes, bucket membership and dense rewards from numeric particle
states, native bucket poses and actual collision hulls. N2 initialization must
match independent N1 seeds exactly. N1 seed 101 is additionally compared with
the old frozen implementation's particles, materials, target and robot state.

The first v6 run used 11,056 and 11,071 particles, with targets 1,113 and 873.
Its original lifecycle subset passed, including task and controller checks.
Inspection of a 3.1665 mm partial-reset particle replay difference then exposed
an omitted check: the selected `wall_1` had received `wall_2`'s pose. Adding actor
fields to the saved-state comparison produces four failures in the retained
v6 records. The original limited verdict and the later failing audit are both
preserved; the original pass does not establish a correct actor reset.

SAPIEN 3.0.3's indexed actor apply path compacts pose records while copying
the old `PxGpuActorPair` entries unchanged. Their `srcIndex` values refer to the
original layout. PhysX therefore reads the wrong pose for some selected actors.
The native allocation/dispatch is in
[physx_system.cpp](https://github.com/haosulab/SAPIEN/blob/3.0.3/src/physx/physx_system.cpp);
the gather kernel is in
[physx_system.cu](https://github.com/haosulab/SAPIEN/blob/3.0.3/src/physx/physx_system.cu).
The index contract is documented in
[PhysX 5.3.1](https://nvidia-omniverse.github.io/PhysX/physx/5.3.1/_api_build/class_px_scene.html).
The compatibility path keeps the full actor-data layout, with unchanged values
in unselected rows, while retaining indexed articulation-root updates. This
does not prove preservation of every hidden PhysX solver cache.

The v7 suite passes all six cases: four Excavate cases and two N2 Fill
controller/reset regressions. All 28 exact checkpoint comparisons, 28 camera
frames and 24 native IK calls pass. Actor restoration is included in the frozen
groups before execution. The two Excavate partial-reset particle errors fall
to 0.5812 and 0.7898 micrometres. Other Excavate replay errors reach 30.742
micrometres and remain descriptive; joint replay stays within the unchanged
1e-3 lifecycle limit. All 36 Excavate snapshot files pass independent target,
reward, outcome and bucket-membership checks. The native initialization and
material comparisons also pass.

The full protocols, numeric verdicts, archive hashes, source audit and preserved
v6 failure are in [gpu-excavate-batching-results.json](gpu-excavate-batching-results.json).
The N1 CPU/GPU regression also passes all six dictionary, flat and rebuilt
checkpoint restores with the same final runtime. Both backends restore the
saved 11,056 particles after a fresh 11,071-particle reset. GPU joint replay
differs by at most 2.3842e-7 radians; CPU joints replay exactly. Particle
differences remain descriptive (GPU 0.991 micrometres, CPU 30.794 micrometres).
These short controls produce no successful excavation demonstration.

## Reproduce

```sh
python -m tools.softbody.verification.gpu_excavate_batch_checks \
    RECORD tools/softbody/verification/protocols/gpu-batch-v7.json \
    OLD_LIFECYCLE_V2_RECORD VERDICT.json
```

The checker reads numeric records without importing candidate simulation code.
The baseline and input/probe/source identities are pinned in the protocol.
Local task and synthetic verifier tests are contract checks, not native physics
evidence. Full episodes, repeated-seed reference parity, the previous Excavate
spill failure, Pour/Write/Pinch batching and distribution packaging
remain incomplete. The original strict particle-trajectory failures remain failures.

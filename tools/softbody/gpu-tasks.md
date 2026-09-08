# Single-scene GPU task integration

All six legacy task prototypes now run through native GPU PhysX with CUDA MPM.
The environment keeps ownership of each rigid step. The MPM world prepares
particle integration and combined reactions before it, then publishes particle
state afterward. Robot compensation uses actual GPU joint state and the native
inertial model. Controllers set physical drive targets; they never assign robot
or object state during a rollout. Observations, rewards and captures read GPU
body/joint buffers, and checkpoint assignment remains restricted to reset.

Select `sim_backend='physx_cuda'` when constructing a task, with the same
external original assets and task-specific numeric packs used by the CPU
prototype. GPU PhysX requires PCM; CPU tasks retain the legacy non-PCM setting.
Both use PGS. Contact behavior across these paths is not certified equivalent.
The task API still requires `num_envs=1`. Shared primitive-world diagnostics do
not establish batched task observations, controls, resets or rendering.

## Frozen short-run experiment

`probe_gpu_tasks.py` uses seed 101, three preselected nonzero actions from RNG
seed 1043, and `pd_joint_target_delta_pos`. Each backend saves a reset checkpoint,
steps those actions, restores the checkpoint through reset, then repeats the
same actions. Each trial contains 75 rigid steps and 300 MPM substeps. The probe
records actual state, outcomes, source hashes and numeric archives and rejects
state assignment outside reset. Write uses the authored `line.h5` goal and
Pinch uses `squeeze_y.h5`; these are not official benchmark validation.

The host independently verifies the frozen input/probe/source identities,
physical backend, timestep representation, action values, finite numeric
traces, measured motion, GPU step accounting, and replay limits. The native
timestep is exactly the float32 representation of 2 ms. An initial verifier
mistakenly compared it to a float64 decimal; the corrected verdict fixes that
representation check without changing any physical tolerance. Both verdicts
are retained in the outer evidence workspace.

The frozen replay limits are 1e-5 for particle and joint positions, 1e-4 for
joint velocities, and exactly zero for drive targets and particle masses.
Other recorded fields are reported descriptively, not silently given wider
limits. Joint coordinates combine radians and metres in native order.

| Task | CPU particle replay error (m) | GPU particle replay error (m) | Short replay verdict |
|---|---:|---:|---|
| Fill | 5.2948e-5 | 5.8904e-5 | Fails particle limit |
| Excavate | 3.1695e-5 | 3.6098e-5 | Fails particle limit |
| Hang | 2.7568e-7 | 3.1293e-7 | Passes declared gates |
| Pour | 5.9596e-4 | 1.0794e-3 | Fails particle and joint gates |
| Write | 1.7882e-7 | 1.7882e-7 | Passes declared gates |
| Pinch | 1.4902e-7 | 1.4902e-7 | Passes declared gates |

Pour's CPU joint-velocity replay difference reaches 2.2575e-3. Its GPU joint
position and velocity differences reach 4.4946e-5 and 9.4594e-4. Saved particle
and joint state restore exactly at the initial checkpoint in the recorded
trials; subsequent trajectory differences remain unexplained by this test.
Native rigid-pose readback is not always exact after reset. Write's CPU height
map also differs by one millimetre at some later pixels despite sub-micrometre
particle differences; this field is descriptive in the frozen protocol.

These are three-action API, stepping and checkpoint tests. They do not prove
successful manipulation, full episode stability, reference parity, broader
controller support, visual accuracy or performance. The combined suite remains
failed, with all original thresholds unchanged.

## Retained setup and integration failures

Version 1 completed Fill, Excavate and Hang on both backends, then failed Pour
setup because its runner omitted the existing bottle-collision pack. Version 2
mounted the complete pack and completed Pour and Pinch on both backends and
Write on CPU. Write GPU failed because its stick agent still called native CPU
`set_qf`. Version 3 switches only that agent to the shared compensation mixin;
both Write backends then complete and pass the declared short replay gates.
Previously completed unaffected tasks were not rerun under the new labels.

| Record | Evidence archive SHA-256 |
|---|---|
| gpu-tasks-v1 | `5ba244da1cfae96ad3132000c42f9676aa8ebe4e1976a508fd7088341622cbc8` |
| gpu-tasks-v2 | `889fc21a48c9f838f2afb61815fbe17c6dfb3556d9976bbf0ab1d7809a14bc40` |
| gpu-tasks-v3 | `978691b610b5c9916287aabc49a809fe777041ce20158e109d84f20c52091e9b` |

[`gpu-task-results.json`](gpu-task-results.json) contains the frozen protocols,
host verdicts and aggregate status. The outer workspace keeps complete numeric
traces and logs under `artifacts/softbody/lambda/evidence-gpu-tasks-v*` and the
independent checker in `softbody_lab/gpu_task_checks.py`. Ten corruption tests
check the verifier itself; they are not simulation evidence.

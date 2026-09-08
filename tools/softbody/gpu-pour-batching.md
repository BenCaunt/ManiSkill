# Pour batching and cooked bottle walls

Pour now constructs 384 native convex wall shapes per bottle from a pinned,
external cooked pack. The cells follow the original visual wall profile and
leave the bottle cavity open. Each environment owns its shapes and rigid body;
the shapes share immutable native mesh data. Original mass, COM, inertia, scale,
friction, fluid SDFs, material recipe and robot drives remain explicit inputs.
The wall representation intentionally replaces MS2's closed rigid convex hull.

The [native loader](native/cooked/README.md) checks every blob and provenance
file before constructing shapes. Missing data or an incompatible adapter fails
explicitly. Supply `bottle_collision_dir`, `MANISKILL_BOTTLE_COLLISION_DIR`, or
the `collision-cooked` subdirectory of the numeric model pack, in that order.
No geometry is downloaded at runtime. Original restricted assets and derived
collision packs remain external to the repository and shared asset catalog.

Each batch row owns its fluid model, bottle, beaker, fill-height targets, target
ring and IK state. Seeded initialization preserves the original random draws.
Selected and reversed checkpoint rows retain their particle/material values,
controller state and target heights. Rendering uses scene-local particle IDs
and actual physical particle positions. The exact selected bottle-pose gate
still fails and is not described as a verified exact restore.

## Native GPU validation

GPU v18 completes four cases: seeds 101 and 17 individually, and two N2 cases
using absolute and aligned end-effector control. Its protocol was frozen before
execution. The independent evaluator checks 36 task snapshots and 2,304 native
convex pieces at initialization, then requires identical readback after reconfiguration.

| Check | Result |
|---|---|
| Native polygons, vertex/index caches, mesh identity and bounds | Pass |
| Original mass, COM, inertia, material and scale | Pass |
| Collision hull error, unchanged 1e-6 m limit | Pass; maximum 1.24e-16 m |
| Original fluid recipe, target geometry and task equations | Pass |
| Ordinary camera particle visibility | Fail in 16 views |
| Exact selected bottle-pose restoration | Fail in eight comparisons |
| Overall protocol | **FAIL24** |

Actual finger contact impulses and grasps occur in all four cases. These short
controls establish contact execution, not successful full manipulation. The old
v11 verdict remains FAIL28 under the same evaluator; its four cooking-geometry
failures disappear with the new pack. No threshold was relaxed.

Host evidence is `artifacts/softbody/lambda/evidence-gpu-batch-v18` and
`gpu-batch-v18-verdict.json`. The input archive SHA-256 is
`151d5875c7a510533429e65c5ce6fcb13eab78f4e83c417c2850378281db1330`;
the result archive SHA-256 is
`76167062a53ec4d474775b24641a53df30d2150f22692720bcabc9434fa1261a`.

GPU v19 keeps the same runtime and ordinary acceptance gates, and adds explicitly
labelled inspection views at initialization. Hiding only the bottle visual
reveals 1,467 or 1,648 correctly positioned particle pixels, depending on seed.
Every newly exposed pixel is explained by the original bottle segmentation and
frontmost depth. Hiding the robot as well reveals no additional fluid pixels.
Maximum sphere-surface error is 1.502 mm under the unchanged 3 mm gate. All
physical/checkpoint/native-buffer values and camera transforms remain identical;
restoring visibility exactly reproduces RGB, depth, position and segmentation.
The 12 altered-view comparisons pass. This establishes bottle occlusion for the
six initial environment rows, not visible fluid in the default camera. The
ordinary v19 protocol still returns the same FAIL24 list. Reduced-count and
reconfigured default views were not independently subjected to this intervention.

The diagnostic evaluator has synthetic fault controls for altered state, camera,
occluder identity, depth ordering and failed restoration. Host evidence is
`gpu-batch-v19-visibility-verdict.json`; input/result archive SHA-256 values are
`82ca1ac133b1f756509ddf19255634af8b60813abbb1d1dfdf481622cd08c73b` and
`e8f2f015fce5e09c60e9150a6f47b6a72a23fb6262defa8d86e59664483b4300`.

GPU v20 repeats all four cases after restricting the native actor adapter
requirement to MPM scenes. Ordinary rigid environments retain their previous
apply path; MPM reinstalls its opt-in when rebuilding the scene. All four cases
complete, the same ordinary FAIL24 list is retained, and all 12 visibility
comparisons pass. The final runtime source matches this frozen capture. The
input/result archive SHA-256 values are
`0d305f69c036c0569976119546aa960d296c8cc5dc8fe0d24afcfcb5634cb1ad` and
`cb3cf1333f3e4996390e1c3c307f67651425911175f91aac241fc3813a2f184c`.
The scoped runtime/verifier suite passed 304 tests before the dependency
restriction; its targeted follow-up passed 53 tests, including two new ordinary
scene and reconfiguration controls. No physics parameter changed.

## Full demonstration

The cooked-wall port replays the original 299-control seed-1 demonstration using
CPU PhysX and CUDA MPM. All 300 recorded states pass artifact integrity checks,
and an independent NumPy evaluator agrees with every recorded success label.
The original joint-position commands are identical in both engines; they act
through drives and physical contacts. No later pose assignment is used to make
the task succeed.

Both engines first succeed at control 297 and finish with 3,842 particles in the
beaker (0.0716159 kg). The reference reports 20 spilled particles; the port reports
zero. Maximum particle-identity distance is 0.72891 m, COM separation 0.003462 m,
joint-position difference 0.004726 rad and current-volume difference 9.636e-7 m3.
These are descriptive measurements; full-episode physical parity remains
uncalibrated. The earlier CoACD replay and its different transfer outcome remain
separate evidence.

The resumable worker runner pins and stages both native adapters and the cooked
pack. Its first cooked replay failed before simulation because a nested Docker
mount required creating a directory beneath a read-only mount. A separate
`/cooked-pack` mount fixed that error. The failed record is retained; the new
job completed, was collected, and was independently audited.

Host evidence is `artifacts/softbody/pour-cooked-integration-v1/` and the compact
[numeric summary](pour-cooked-results.json). The full replay result SHA-256 is
`40577922c8ec6393c4075a99fc17b9a84e5e5bcc9f8d711e3c65edc6ab89889d`.

The native compatibility build remains experimental. Full SAPIEN wheel and
portable pack preparation, broader manipulation seeds/controllers, batch
performance, and the strict pose/visibility failures remain unfinished.

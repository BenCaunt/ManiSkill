# Native selected actor reset: verified improvement, incomplete Pour acceptance

The SAPIEN 3.0.3 adapter fixes all six neighboring-state failures observed in
the two batched Pour cases. A partial reset now leaves unselected actor and
checkpoint state exact. Fill, Excavate and Hang regressions also pass their
existing gates: 21 exact checkpoint comparisons, 18 camera frames and 36
recorded native IK calls across three two-environment cases.

Pour still **fails 28 checks** under the unchanged strict protocol: eight
selected bottle pose comparisons, sixteen fluid-visibility checks and four
native collision-geometry comparisons. The earlier 34-failure result remains
retained. Neither this change nor the passing regression establishes full
reference parity, successful pouring, or completion of the six-task port.

## Native mechanism and independent evidence

SAPIEN's indexed CUDA gather copies an original `PxGpuActorPair` after
compacting the body data. Its `srcIndex` still points into the original full
layout. The adapter constructs compact pairs using each actual native
actor's `getInternalIslandNodeIndex()` and calls public `PxScene::applyActorData`.
The native pointers come from the type-checked C++ conduit and live only for
the duration of each call. The adapter validates ownership, duplicate actors,
buffer shape/dtype, finite states and quaternion validity before applying data.

Three zero-step native probes compare the adapter's readback with SAPIEN's
independent public GPU buffer. Reordered dynamic/kinematic updates preserve
every unselected actor exactly, and seven invalid-input cases leave the native
state unchanged. With the original bottle COM, a selected update has position
residual up to 1.49e-8 m and quaternion-component residual 5.96e-8. Twenty
read/apply repetitions accumulate up to 1.19e-6 in pose components. The
identity-COM controls are exact; these controls do not change the port's physics.
The strict selected-pose failures have not been waived.

## Reproduction and retained limits

See [native build instructions](native/README.md). The documented build produced
the exact GPU-tested binary:
`2c7ab84cdc9c6421be985164cf569f1630bbfea936dfab9d8e312c28e3b27feb`.
The probe is `probe_native_actors.py`; independent checks live in
`verification/native_actor_checks.py`. Protocols are
`verification/protocols/native-actor-probe-v1.json`,
`verification/protocols/gpu-batch-v11.json`,
`verification/protocols/gpu-batch-v12.json` and
`verification/protocols/gpu-batch-v12-hang.json`.
[Recorded verdicts and artifact hashes](gpu-native-actor-reset-results.json)
retain the full Pour failure list. The probe requires the experimental Pour
batch implementation in the tested working tree, which is not yet published.

The actor-inclusive Pour lifecycle uses the unchanged original bottle mass,
inertia, COM, 128 collision inputs, fluid recipe and camera view. Its remaining
geometry error is 0.000698666 m. Separate local native cooking experiments
reproduce that default error; tightening tolerances still fails the full
1e-6 m geometry gate and can stall on hull 97. No cooking experiment has
replaced a live task collider. The original opaque bottle view still fails
fluid pixel coverage; a separately identified particle visibility diagnostic
is needed before changing rendering acceptance.

Local validation: 231 host tests plus 14 subtests and 144 fork tests pass.
Portable checkers exactly reproduce the native probe PASS, Pour FAIL28, and
both shared-task regression PASS verdicts. All 515 runtime source files
(excluding the empty native-library mount point) match the tested source;
the Pour and shared-task regression runs use identical runtime bytes.

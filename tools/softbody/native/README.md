# SAPIEN 3.0.3 selected reset adapter

The pinned SAPIEN indexed actor update compacts its data but retains the full
buffer's `PxGpuActorPair.srcIndex`. A selected actor can receive the wrong state.
Applying the entire buffer avoids that indexing defect but can round untouched
actors with a nonidentity center of mass. This adapter uses the public native
`PxScene::applyActorData` with correctly indexed compact data. It checks native
types through pybind11's documented C++ conduit; it does not guess pointers or
access private object layouts.

The indexed joint position, velocity, force and drive-target methods also ignore
the supplied indices in SAPIEN 3.0.3. Resending all joint rows avoids a wrong-row
write but marks untouched articulations for kinematic recomputation. The native
link poses and velocities can then change during another environment's reset.
`apply_articulation_data` keeps the full native joint buffers and submits only
the selected global articulation indices to `PxScene::applyArticulationData`.
It rejects duplicate or out-of-range indices and synchronizes CUDA before the
temporary index buffer is released. The [scaling study](../gpu-scaling.md)
records the original failures and the repeated all-row checks.

Soft-body scenes opt into this extension for partial actor and joint resets on SAPIEN
3.0.3, including after scene reconfiguration. Ordinary rigid-body scenes retain
their previous dependency requirements and native apply path. Full resets retain
the original native API. Selected global-pose read/apply still has native
float32 roundoff: this adapter does not promise exact selected-pose restoration
or reference physics parity. Other SAPIEN versions retain the previous full
actor apply behavior and have not been verified by this diagnostic.

Build on Linux using the same Python environment as the simulation. The tested
image has Python 3.10, SAPIEN 3.0.3, Torch 2.5.1, GCC 11 and CUDA 11.8. Supply
the Eigen 3.4.0 source pinned by SAPIEN (commit
`3147391d946bb4b6c68edd901f2add6ac1f31f8c` from
<https://gitlab.com/libeigen/eigen/-/tree/3.4.0>). The build makes no downloads.
The small vendored pybind11 conduit headers include their license and immutable
source URLs/checksums in `vendor/provenance.json`.

```sh
python tools/softbody/native/build.py --eigen /path/to/eigen-3.4.0 --output /path/to/new-build
export PYTHONPATH="/path/to/new-build:$PYTHONPATH"
python -c 'import sapien; import sapien303_actor_bridge as b; print(b.abi())'
```

Rebuild older versions of the extension before using the selected-joint path;
an older binary without `apply_articulation_data` is rejected explicitly.

The adapter synchronizes a CPU staging copy during reset. It is not used for
simulation stepping. The GPU actor diagnostic uses the actual two-environment
Pour scene, reordered dynamic/kinematic selections, invalid-input rejection,
and independent public-buffer readback. Identity-COM cases are diagnostic
controls, not altered task physics or successful manipulation evidence.

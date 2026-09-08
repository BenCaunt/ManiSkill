# Native joint-frame and COM pose precision

[probe_pose_setters.py](probe_pose_setters.py) completed 157 native measurements
with the supplied Python on 2026-09-08 at 13:03:18 UTC. Full inputs, original
exported decimal tokens, raw getter arrays, float32 bit patterns, residual
vectors, case identifiers and provenance are in
[probe_pose_setters.results.json](probe_pose_setters.results.json).
Two fresh processes produced identical measurements, fixtures and summaries.

Reproduce from this checkout, choosing an unused output path:

```sh
mkdir -p .softbody-tmp
TMPDIR="$PWD/.softbody-tmp" PYTHONDONTWRITEBYTECODE=1 \
  /Users/bencaunt/Documents/ChatGPT/sim-infra/.venv/bin/python \
  tools/softbody/probe_pose_setters.py \
  --output .softbody-tmp/pose-setters-replay.json
```

The script reads the shared asset docs/catalog before constructing scenes; this
setter probe needs no geometry or mesh assets. `--asset-docs` and
`--asset-catalog` support another checkout layout. It creates only an explicit
`PhysxCpuSystem`, two native articulation links, and (for COM cases) one native
rigid dynamic component. There are zero physics steps, renderers, GPU simulation
systems, mesh loads, downloads, root/qpos assignments, or task imports. SAPIEN
emits Vulkan-discovery and missing-Pinocchio warnings at import; neither renderer
creation nor Pinocchio is used here.

All 10 exported COM poses are assigned to root, child and dynamic components.
Both frames of all 9 exported joints are tested with fixed and revolute-unwrapped
joints, using adjacent exported link COM records as contexts. These pairs do not
reconstruct robot topology or FK. Separate identity-COM controls isolate COM
dependence. Five authored controls include identity, nonidentity binary-exact
and decimal poses, a negative quaternion, and a half-turn with small translation.
Every fixture retains its source record's mass and principal inertia.

Maximum absolute component residuals, **native getter minus assigned
`sapien.Pose`**, were:

| Inputs / COM context | Measurements | Position (m) | Quaternion (wxyz) |
|---|---:|---:|---:|
| Exported COM poses | 30 | 0.0 | 1.1920928955078125e-07 |
| Exported joint frames / exported COM | 36 | 2.384185791015625e-07 | 1.7881393432617188e-07 |
| Exported joint frames / identity COM control | 36 | 0.0 | 1.7881393432617188e-07 |
| Authored COM poses | 15 | 0.0 | 0.0 |
| Authored joint frames / exported COM | 20 | 1.7881393432617188e-07 | 2.384185791015625e-07 |
| Authored joint frames / identity COM control | 20 | 0.0 | 0.0 |

For example, `panda_joint3`'s parent-frame Y input and `sapien.Pose` value were
both `-0.3160000741481781`; the native getter returned `-0.3160003125667572`.
The signed residual was `-2.384185791015625e-07` m.

The `sapien.Pose` constructor preserved every exported component exactly.
The authored decimal control separately measured constructor residuals of
`5.522669099811139e-09` m and `2.05435399802667e-08` in quaternion components.
Subtractions use float64 after losslessly widening native float32 getter values.
No input or readback is normalized, rounded, sign-aligned, or substituted.
All repeated-read, method/property, same-input-reassignment, cross-frame and
input-mutation residuals were zero. Both joint types agreed; all three COM
component roles agreed. Mass and inertia assignment/readback residuals were zero.

Provenance: Python 3.12.10, SAPIEN 3.0.3, NumPy 2.2.6, macOS 15.1 arm64.
The export SHA-256 is
`31acda9fd0d7d81474f90c2870db7a1a97d12d423cab38b4ab4be9c47c77c03e`;
its source commit is `493be36121a9dd06071a57172274babe617b789f`.
The JSON also records the exported source-capture hash, checkout commit, script,
native extension, `libsapien.dylib`, asset docs/catalog and dependency-header
hashes. The installed `math/quat.h` and `math/vec3.h` use float components;
`math/pose.h` defines float pose operations. The component headers declare COM
and joint-frame setters/getters, but do not supply their implementations.

This proves that native assignment/readback adds observable residuals beyond
the pose constructor, with joint translation residuals dependent on COM context
in these cases. The supplied Fill initial discrepancies
(`3.5762786865234375e-07` m and `1.7881393432617188e-07` per quaternion component)
are of comparable magnitude, but measure different quantities. This experiment
does not establish how these residuals propagate through Fill FK, separate
setter from getter arithmetic, compare SAPIEN 2 behavior, or identify a specific
source-backed defect or correction. It does not revisit reset ordering or test
contacts, drives, MPM, dynamics, two-way coupling, or full port parity.

Runtime physics, model parameters, reported state and acceptance limits remain
unchanged. Exit status zero means the diagnostic completed; only the independent
evaluator can issue a parity verdict. No physics fix follows from this evidence.

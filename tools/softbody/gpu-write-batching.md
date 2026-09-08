# Write batching: implementation and verification status

`WriteEnv` now opts into the shared MPM batch runtime. Each environment owns its
goal points, goal/current raster buffers, cached IoU, material builder, stick
links and wall bodies. Model parameters and seeded material generation retain
the single-environment recipe. Goals remain explicit external HDF5 files.

`reset(options={"level_file": ["line.h5", "elbow.h5"]})` selects a goal per
selected environment, in `env_idx` order. A single filename broadcasts over
selected environments. Checkpoints restore goal points with the corresponding
physical row; dictionary and flat forms accept reversed row indices. Robot
setter inputs are reordered to match the native boolean reset mask.

The goal always has 19,404 points. A checkpoint may restore a different live
material count within the declared capacity. Current-image rasterization and
reward distances use that live count, excluding padding. Only reset can restore
physical state. The original single-environment `goal_points`, `goal_image`,
`current_image`, `iou_buffer`, `goal_image_display`, `level_file` and
`level_sha256` views continue to refer to row zero.

Local tests use real CPU Warp raster kernels and real unfinalized MPM builders.
They cover distinct goals, own-row reward/observations, live counts, reversed
indices, selected checkpoint restoration and invalid inputs before mutation.
Separate comparisons with saved MS2 GPU captures pass 66 checks across seeds
101, 17 and 1 in N1/N3 initialization-only runs. Robot/contact construction and
physics integration are not exercised by those local comparisons.

The independent `verification/gpu_write_batch_checks.py` checks GPU captures
against original numeric robot/wall parameters, pinned reference initial states,
native TCP buffer rows, own-row reward equations, and the existing PTX arithmetic
interval raster contract. Ambiguous success or a pixel outside the arithmetic
bounds fails. It also retains the general actor-inclusive checkpoint, controller,
camera and replay checks. The capture suite includes distinct goals, partial
fresh/restored resets, reduced material count, reversed flat restoration and
reconfiguration. GPU v16 completes all six N1/N2 cases on the pinned Linux A10
worker. Independent task arithmetic, original particle/material/goal inputs,
controller, checkpoint, replay, RNG and rendering checks pass.

The overall verdict remains FAIL: all six cases differ from the exact original
`robot_com`, `joint_parent_pose` and `joint_child_pose` arrays (18 failures).
Maximum position-component differences are zero for COM, 2.98e-7 m for parent
frames and 5.22e-8 m for child frames. Quaternion components differ by at most
3.58e-7. An independent Linux reproduction constructs native articulation links
from URDF topology, applies original mass/joint inputs, and reads them back without
importing ManiSkill, loading meshes or simulating. All five model arrays match
every GPU case exactly. The original getter-byte failures reflect native frame
conversion; these measurements do not replace the exact gates. The host evidence
is `artifacts/softbody/model-frame-v1/verdict-v1.json`.

Earlier GPU v14/v15 failures exposed diagnostic TCP-name lookup and primitive
visual-bound capture errors. The probe now reads native entity names and uses
shape-specific bounds. Both failed archives remain retained. No physical model,
controller gains, arithmetic interval or acceptance tolerance was changed.
The host record is `artifacts/softbody/lambda/gpu-batch-v16-verdict.json`.

```sh
python -m pytest -q tools/softbody/test_write_batch.py
python -m tools.softbody.verification.gpu_write_batch_checks \
  records protocol.json saved-reference-root numeric-pack-root goal-directory verdict.json
```

The three goals are authored diagnostics, not official benchmark levels. Existing
failed Write manipulation/parity runs remain failures. This implementation does
not establish GPU batch performance, full task parity, or distributable packaging.
The shared partial-reset RNG fix now preserves unselected streams and maps seeds
in selected-row order. Nineteen local tests exercise actual BaseEnv reset dispatch;
GPU v16 passes the independent exact RNG stream checks. The earlier GPU v13 archive predates
this fix and Pinch batching. It is retained as a historical preparation, not the
current runtime candidate.

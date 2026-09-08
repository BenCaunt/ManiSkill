# Particle rendering and reset verification

The bridge supplies visual-only particle poses to SAPIEN 3.0's batched renderer
through a separate CUDA buffer. Native rigid rows are copied into its prefix;
particle rows follow them. Rendering cannot write back through that buffer into
PhysX. Particle entities retain their scene IDs across resets, avoiding the
minimal shader's signed-int16 segmentation saturation after repeated allocation.
Inactive pooled entities are hidden. Count or material changes release render
and camera groups before changing their membership.

SAPIEN can upload CPU transforms on the first draw after a scene edit. Newly
created camera groups therefore finish that initialization draw before the
normal CUDA pose update and the capture returned to the caller. This matters
for the robot as well as the particles: a particle-only image check initially
passed while the robot was displayed at an incorrect pose after reset.

The source-bound Fill GPU regression checks six stages: initial state, five
nonzero control actions, a 704-to-352-particle checkpoint restore, restoration
to 704 particles, expansion to 1,056 particles, and full scene reconstruction.
Every stage is captured twice without stepping between captures. Expansion
copies existing per-particle material inputs and translates the new particles
by 6 cm during reset. This is a rendering lifecycle diagnostic, not a changed
benchmark episode or a manipulation controller.

All twelve frames pass the original 3 mm surface allowance for camera XYZ
quantized to millimetres. Maximum particle surface error is 1.5481 mm; maximum
robot visual-bound violation is 1.4654 mm. Robot bounds come from each link's
visual mesh transformed by its actual physical pose. This catches incorrect
link placement, but is not an exact triangle-surface or appearance comparison.
The host also checks exact checkpoint and native rigid CUDA buffer values
before and after rendering, active/pool identities, absence of inactive-particle
pixels, camera coordinate conventions, and PNG equality with actual RGB tensors.

The v8 observation-camera regression also passes all six stages on both CPU
and GPU. Returned RGB/depth/segmentation arrays match the actual native sensor
capture exactly. At 128-by-128 resolution, each frame contains at least 132
particle pixels and 2,470 robot pixels. Maximum particle surface error is
1.4832 mm; maximum robot visual-bound violation is 1.2410 mm. This exercises
visual observations during ordinary steps and resets, as well as explicit
camera capture.

The v9 Write regression passes initial, stepped, checkpoint-restored and
reconstructed frames on both CPU and GPU. All 19,404 particles retain unique
representable scene IDs, and each image contains at least 38,009 particle pixels.
Maximum particle surface error is 1.6081 mm. This uses the existing authored
Write fixture for camera verification; it does not validate an official Write
benchmark level or its success criterion.

## Independent checks

The `verification` package uses NumPy/Pillow on the host, loads numeric records
with `allow_pickle=False`, and checks source, probe, request and image identities.
It does not execute candidate code when judging results. Its fault-injection
tests reject stale particle/robot pixels, changed physics, altered observation
images, invalid camera frames, ambiguous segmentation IDs and hidden-particle
leaks. Separate CPU unit checks exercise render-buffer ownership and CUDA device
aliases; those tests are not evidence of native GPU execution.

Given an extracted worker record and its independently frozen protocol:

```sh
python -m tools.softbody.verification.gpu_rendering_checks RECORD PROTOCOL.json VERDICT.json
python -m tools.softbody.verification.gpu_lifecycle_checks RECORD PROTOCOL.json VERDICT.json
pytest tools/softbody/verification tools/softbody/test_render_only_poses.py
```

The accepted protocols are supplied in `verification/protocols/`; use the one
whose input SHA-256 matches the worker record. Source identities differ between
the lifecycle run and later rendering revisions, as recorded in each protocol.

The matching probe is `probe_gpu_rendering.py`; the protocol supplies its request
and case matrix. Raw archives remain external to Git and are identified by their
SHA-256 values in `gpu-rendering-results.json`. Protocols and verdicts are retained
with the results, including failures. Asset packs and the built legacy Warp
library remain separate prerequisites under their original licenses.

## Retained failures and limits

- v2 exposed stale GPU particle positions, GPU reset membership exceptions, and
  Write segmentation-ID saturation. The shader saturates IDs; it does not wrap
  them. The corrected host interpretation and preliminary verdict are retained.
- v3 fixed reset membership and ID reuse, but native CPU render updates did not
  refresh the batched particle transforms: Fill errors still reached 26 mm.
- v4 passed the CPU count changes and rejected the GPU device alias at startup.
  v5 corrected that validation and passed particle checks, while visual review
  found incorrect robot images after count-changing reset.
- v6 added robot bounds and repeated images. The first image after three reset
  stages failed by 0.683 m; the immediate second image passed without a physics
  step. v7 fixes that first-image upload ordering and passes all twelve frames.

These rendering runs use single-environment tasks. The separate
[Fill batching suite](gpu-batching.md) verifies N2 scene ownership, partial reset
and camera geometry while retaining strict particle-trajectory failures.
Particle updates still copy
positions through CPU memory and update individual entity poses. The render-only
CUDA registration currently supports SAPIEN 3.0; SAPIEN 3.1 is rejected explicitly.
Live scenes with IDs beyond the shader's representable range need a wider
segmentation format. These checks do not establish multi-environment task
batching, full episode success, reference dynamics parity, photorealism or
production rendering performance.

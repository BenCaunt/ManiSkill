# Particle render-update performance

Batched GPU rendering now copies particle positions directly from the preserved
Warp allocation into the visual-only CUDA pose buffer. It avoids particle CPU
readback and per-entity Python pose assignments. The update takes approximately
0.08–0.09 ms per model in the measured A10 cases. CPU rendering and the native
single-view path retain entity-pose updates; inactive pool entries and particle
identities keep their existing reset behavior.

The implementation uses the actual stream exposed by the bundled Warp runtime.
It waits for preceding Torch writes to the destination, copies on the solver
stream, then makes subsequent Torch work wait for that copy. The solver retains
ownership of the borrowed particle allocation. This uses PyTorch 2.5.1's
[`ExternalStream` and `wait_stream`](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/cuda/streams.py);
the older bundled Warp lacks the current Warp stream-conversion API. Native
solver code, physical parameters, materials and controller code are unchanged.

## Measured scope

The paired experiment records 56 model configurations during six original task
probe runs: batched Fill, single/batched Write, and three single/batched Pour
cases. It includes partial reset, reduced particle counts and reconfiguration.
Each comparison uses the same physical particle state. The baseline pool method
is copied verbatim from commit `0d45e18fe8c140e044beabba8918a185ae7391b1`, with
the preceding caller's Torch copy. Five alternating trials time ten updates
each after warmup, including CUDA completion. All 56 comparisons exceed the
independently declared twofold minimum.

| Task | Active particles per model | Previous update | Direct CUDA update | Measured speedup |
| --- | ---: | ---: | ---: | ---: |
| Fill | 352–704 | 0.58–1.05 ms | 0.079–0.082 ms | 7.1–13.1× |
| Write | 9,702–19,404 | 12.15–24.31 ms | 0.086–0.090 ms | 139–277× |
| Pour | 4,267–9,036 | 5.45–11.41 ms | 0.082–0.090 ms | 64–137× |

These timings cover visual pose updates per model. They do not measure full
simulation-step or camera-render throughput. A batch updates each of its models;
the solver, controller, rigid-state transfer and other rendering costs remain.

All recorded CUDA positions agree exactly with physical particle positions.
Particle state is unchanged by the measurements. Blocking particle `.numpy()`
and native entity pose writes leaves the new CUDA path working and rejects the
old path. Borrowed tensor and Warp data pointers match. Delayed writes on a
nondefault Torch stream remain correctly ordered; an intentionally unsynchronized
copy fails that same check in every measured configuration. The independent
verifier also rejects modified poses, modified particle state, wrong stream
results and a negative control replaced with the expected answer.

The three local fallback configurations use real SAPIEN entities with deliberately
stale poses. They verify replacement of active poses, unchanged source positions,
preservation of inactive entries and quaternion columns, and the option to update
only the pose buffer. These are fallback data-flow checks, not a new CPU-renderer
or full single-view GUI validation.

## Task regressions and reproduction

The instrumented task runs retain the prior controls, reset sequence and numeric
limits. Fill passes. The two selected Write cases retain exactly six original
model-frame failures. The three selected Pour cases retain exactly 22 failures:
14 default-camera visibility gates and eight exact selected-bottle-pose gates.
No failure is added or removed. Prior full-episode failures and missing original
Pinch/Write benchmark levels remain outside this performance result.

Separate uninstrumented replays repeat all six cases using identical runtime
source, controls and numeric gates. Fill again passes; Write retains the same
six failures and Pour the same 22. This checks that the timing observer's baseline
entity updates did not hide a rendering regression. Pour's ten separate initial
visibility interventions pass: newly visible fluid pixels are explained by bottle
occlusion, physical state stays unchanged and restoring visibility restores the
original camera output exactly. These diagnostic views do not make the ordinary
camera gates pass.

The first clean replay (v22) incorrectly enabled Pour-only visibility diagnostics
for Fill and Write and failed those three cases. Its
[failure record](verification/render-performance-clean-v22-failure.json) is
retained. Correctly configured Fill/Write (v23) and Pour (v24) runs both complete;
their full independent verdicts and protocol hashes are in the result manifest.

The [verifier source audit](verification/render-performance-verifier-source-audit.json)
records one inherited metadata discrepancy: copied Pour protocols contain an old
`cooked_extensions.py` checksum. The separate inventory frozen before these runs
pins the actual helper. Its only addition is the worker's `pack_path` function;
the evaluator's imported validation functions are unchanged. The original
protocols remain available with that discrepancy recorded.

The [performance protocol](verification/render-performance-protocol.json),
[result manifest](verification/render-performance-results.json), matching probe,
baseline update method and independent verifier are retained together. The worker
runner expects the existing pinned `/home/ubuntu/softbody` layout, image, native
adapters and externally licensed reference assets; it does not download assets
or provision a GPU. The request supplies the source inventory and each helper's
checksum. Original benchmark assets and raw result archives remain external.

Given the verified, extracted instrumented worker archive:

```sh
python -m tools.softbody.verification.render_update_checks \
  /path/to/evidence-gpu-batch-v21 \
  tools/softbody/verification/render-performance-protocol.json \
  /path/to/new-verdict.json
python tools/softbody/test_particle_visuals.py
```

The optimized path targets the pinned SAPIEN 3.0 batched renderer and the tested
single-GPU legacy Warp runtime. Full task parity, multi-GPU operation, CUDA graph
capture and end-to-end performance are not established by these measurements.
These GPU replays execute the source checkout. The previously verified installed
wheel predates this rendering change.

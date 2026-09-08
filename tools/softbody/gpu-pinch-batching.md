# Pinch batching and selected reset RNG

Status: GPU v17 completes six N1/N2 cases; overall exact-model verdict FAIL18.

`PinchEnv` now opts into the shared MPM runtime. Each row owns its numeric level,
goal particles, camera data, projected goal, distance buffers, cached score and
robot links. Level selection preserves the original seeded nine-joint noise
recipe. A list of `level_file` names follows `env_idx` order; one name broadcasts
to the selected rows. All supplied levels are validated before initialization.

The original MPM material, domain, gravity and contact parameters remain in use.
Initial particle and physical robot state are deferred until after the upstream
controller reset. Nominal controller caches and existing drive buffers are
preserved. Native drive targets are distinct from controller caches; saved MS2
initial drive targets are zero.

Goal checkpoints have a padded `task_particles/goal` array with live counts from
`mpm_meta/count`, plus camera and normalization fields in `task`. Dictionary and
flat restores accept selected rows and changed particle counts. Only reset may
assign state. The shared restore path canonicalizes reversed indices before
native boolean-mask setters, keeping particles, physical state, drives and
controller caches associated with the requested rows. Single-environment public
goal properties continue to expose row zero.

The shared reset adapter also fixes upstream global reseeding during a partial
reset. Explicit seeds replace only selected main and episode streams, in selected
row order. Unselected MT19937 words, cursor and Gaussian cache remain exact.
Unseeded enhanced-determinism resets draw only from selected main streams.
Full-reset behavior retains the upstream seed expansion recipe.

## Evidence and limits

The combined local runtime suite passes 105 tests, including 19 RNG tests through
actual BaseEnv reset dispatch. Native physics and GPU rendering are stubbed in
these contract tests; real unfinalized MPM builders and CPU Warp buffers are used
for Pinch/Write task tests. Reversed dictionary/flat restoration is checked across
particles, physical state, controller caches and drive buffers in three rows.

The real-level initialization comparison uses three frozen MS2 diagnostic
captures, seeds 101, 17 and 1, each in N1 and N3. It passes 126 of 132 checks.
Particles, materials, joint states, goals and camera metadata match exactly.
The six failures are the same root-pose discrepancy: the original numeric level
differs from the MS2 pose captured after native application by at most
1.1920928955078125e-7. The new deferred checkpoint matches the original level
exactly. This local comparison does not execute native pose application, so the
strict captured-root failure remains unresolved. No tolerance was changed.

The inherited NumPy 2 meshgrid tuple/list incompatibility was fixed without
changing projection arithmetic. Candidate projection matches the saved goal
observation exactly. Independent NumPy/SciPy projection differs by at most
4.32266543115567e-8, within the existing 1e-7 gate.

The packaged independent verifier passes 59 tests. It checks the original
directed fourth-norm metric using arithmetic bounds, strict success, progress,
own-row native TCP observations/reward, camera projection, pinned numeric inputs,
requested level provenance and distinct fresh-reset goals. It retains the
existing 1e-12 initial-progress gate. The broader host verifier suite passes 110
tests; a subsequently added forged-progress fault control also passes.

An expanded original MS2 capture now covers all nine combinations of three
seed/goal pairs and joint-delta, end-effector-delta and end-effector-target-delta
controllers. Each combination has a four-control generation and fixture replay.
All 18 traces and 90 samples pass integrity and independent task arithmetic
audits. The verifier requires an exact matching reference combination.
GPU v17 exercises three N1 and three N2 cases against these frozen references.
Independent task arithmetic, particles/materials/goals, native TCP observations,
controllers, partial/reversed/count-changing checkpoints, replay, exact RNG streams
and rendering checks pass. There are 54 task snapshots, 21 selected checkpoint
comparisons and 24 camera checks. Overall FAIL18 is retained: each case's original
COM, parent-joint and child-joint arrays differ exactly. A separate native-only
Linux articulation reproduction matches every model array in every case exactly,
without ManiSkill, meshes or physics steps. This identifies native frame conversion;
it does not change the exact original-input gates.

A supplemental audit found that the task verifier omitted initial physical-root
comparison. Optional `pinch.initial_root_contract=1` adds exact root pose/velocity
checks; it passes 49 verifier tests including one-bit physical changes with an
unchanged controller cache. Re-evaluation with the original protocol reproduces
FAIL18 unchanged. A separately recorded post-capture protocol produces FAIL36,
adding 18 root-pose differences (maximum position component 3.73e-9 m and quaternion
component 2.75e-9; velocities exact). This is a development audit, not a gate
declared before capture. Both protocols and all verdicts remain retained.

Host evidence: `artifacts/softbody/lambda/gpu-batch-v17-verdict.json`,
`gpu-batch-v17-root-verdict-v1.json`, and `artifacts/softbody/model-frame-v1/`.
Full manipulation, physical fidelity and batch performance remain incomplete.
Original benchmark levels remain unavailable; diagnostic
fixtures and the existing unsuccessful full Pinch manipulation do not establish
benchmark acceptance. Earlier strict failures remain failures.

Local evidence: `artifacts/softbody/pinch-batch-v1/` in the host workspace.

```sh
python -m pytest -q tools/softbody/test_pinch_batch.py tools/softbody/test_batch_rng.py
python -m tools.softbody.verification.gpu_pinch_batch_checks \
  records protocol.json frozen-reference-root numeric-pack-root verdict.json
```

# Legacy controllers in shared GPU tasks

The legacy end-effector controllers now build an independent Pinocchio model
from each environment's native SAPIEN articulation. Every action row uses that
robot's joint configuration, active-joint mask and end-effector link index.
Target poses retain the original rotation-vector and `ee`, `base`, `ee_align`
composition rules. Each row uses the same native float32 pose operations as the
original single-environment adapter; the change does not use ManiSkill's newer
Euler-angle action convention.

Partial reset updates only selected target-pose rows. Checkpoint restoration
owns its target memory and requires one finite pose per environment. Failed IK
keeps that environment's starting joint targets while successful neighbours use
their own solutions. These operations set joint drives; they do not assign robot
or object poses. The model rebuild follows native scene reconstruction. IK still
runs on the CPU, and this implementation makes no throughput claim.

## Native verification

The frozen v4 smoke suite passes two N2 cases—target-relative pose delta and
aligned pose delta—and an N1 GPU regression. It records all 28 calls into the
actual native IK models. Every call succeeds and uses the recorded robot row,
target pose, arm mask and 100-iteration limit. The independent host reconstructs
target composition from pre-action poses and actions, and binds those values to
separately hashed numeric snapshots. Native IK solutions agree with the targets
the controller actually adopts.

The two N2 cases also pass all fourteen exact checkpoint comparisons, covering
selected restore, untouched state, a particle-count change and both flat-state
rows. All fourteen camera frames pass the existing physical-particle and robot
bounds checks. Model ownership is checked again after full scene reconstruction.
Maximum joint replay difference is 1.1921e-7 radians, within the existing 1e-3
controller lifecycle limit. Particle replay differences reach 2.5325e-5 m and
remain descriptive in this suite.

The v5 suite passes all eleven original Panda arm modes, including joint
velocity and joint position/velocity commands, with the same runtime source
and probe. Its two environments use seeds 101 and 17 and different action
scales. All 60 native IK calls, 77 exact checkpoint comparisons and 66 camera
frames pass. Maximum joint replay difference is 2.3842e-7 radians; particle
replay differences reach 7.5162e-5 m and remain descriptive. The separate strict
[particle-trajectory failures](gpu-batching.md) remain failures.

The original N1 action/target decoding was also checked locally against the
previous source on the same runtime: all 21 records agree exactly. Comparison
with older Linux records differs by up to 5.96e-8, so the cross-platform records
are not used as evidence of a code regression. Thirteen controller contract
tests cover composition, norm clipping, separate model creation, partial reset,
checkpoint ownership, failed IK and non-finite IK results. Eight independent
verifier fault tests reject stale/swapped robot rows, wrong targets or solutions,
missing calls, reported failure and non-finite telemetry.

The same runtime also passes the [23-case CPU/GPU regression](gpu-lifecycle.md),
including all six task classes and 69 exact checkpoint restores.

## Reproduce

Use the numeric archive and matching frozen protocol from
`gpu-controller-batching-results.json`:

```sh
python -m tools.softbody.verification.gpu_controller_batch_checks RECORD \
    tools/softbody/verification/protocols/gpu-batch-v4.json VERDICT.json
pytest tools/softbody/test_legacy_ee_batch.py tools/softbody/verification
```

The probe is `probe_gpu_batch.py`; use the hash recorded in the protocol for a
historical run. Asset packs and built libraries remain external under their
original licenses. The controller checker reads numeric evidence without
executing candidate code. Full task success, reference physics parity, batching
of the other five task classes and distribution packaging remain incomplete.

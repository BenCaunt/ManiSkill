"""Recompute legacy task outcomes from numeric state, without simulator imports.

Implements the benchmark predicates in ManiSkill 2 v0.5.3 task source:
https://github.com/mani-skill/ManiSkill/tree/493be36121a9dd06071a57172274babe617b789f/mani_skill2/envs/mpm
The legacy soft-body source terms in LEGACY-SOURCE-LICENSE.md apply. This is
an outcome audit, not a calibrated physical-fidelity or trajectory-parity gate.
"""
from pathlib import Path
import numpy as np
from .artifacts import InvalidArtifact, child_file, load_frame, validate_trace


def outcome(state, env_id, *, fill_height=None):
    x, velocity = state['x'], state['v']
    if env_id == 'Pinch-v0':
        from .pinch_checks import outcome as pinch_outcome
        return pinch_outcome(state)
    if env_id == 'Write-v0':
        from .write_checks import outcome as write_outcome
        return write_outcome(state)
    if env_id == 'Excavate-v0':
        target, = state['task_state']
        lifted = x[:, 2] > .2
        within = (x[:, 0] > -.12) & (x[:, 0] < .12) & (x[:, 1] > -.12) & (x[:, 1] < .12)
        lift_count = int(lifted.sum())
        quiet = np.count_nonzero((velocity > -.05) & (velocity < .05)) / velocity.size
        checks = dict(amount=bool(target-100 < lift_count < target+150),
            spill=int((~within).sum()) < 20, quiet=bool(quiet > .99))
        return dict(success=all(checks.values()), checks=checks, target_particles=int(target),
            lifted_particles=lift_count, lifted_mass_kg=float(state['mass'][lifted].sum()),
            spilled_particles=int((~within).sum()), spilled_mass_kg=float(state['mass'][~within].sum()),
            quiet_fraction=float(quiet))
    if env_id == 'Fill-v0':
        if fill_height is None or not np.isfinite(fill_height) or fill_height <= 0:
            raise ValueError('A source-backed beaker height is required')
        center = state['scene_actor_pose'][1, :2]
        bounded = ((x[:, :2]-center)**2).sum(1) < .04**2
        bounded &= x[:, 2] < fill_height
        inside = bounded & (x[:, 2] > 0)
        quiet = np.count_nonzero(abs(velocity) < .05) / velocity.size
        return dict(success=bool(inside.mean() > .9 and quiet > .99),
                    contained_particles=int(inside.sum()),
                    spilled_particles=int((~bounded & (x[:, 2] < .005)).sum()),
                    contained_mass_kg=float(state['mass'][inside].sum()), quiet_fraction=float(quiet))
    if env_id == 'Hang-v0':
        # Portable v2 records the source Panda link order: root/link1..8,
        # hand, tcp, left finger, right finger; rod is the last rigid row.
        rod = state['rigid_pose'][13]
        w, qx, qy, qz = rod[3:].astype(float)
        normal = np.array([2*(qx*qy-w*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz+w*qx)])
        selected = x[state['task_state'].astype(int)] - rod[:3]
        side = np.sign(selected @ normal)
        high_velocity = velocity[x[:, 2] > rod[2]-.03]
        quiet = np.count_nonzero(abs(high_velocity) < .05) / (high_velocity.size+.001)
        finger_distance = np.linalg.norm(state['rigid_pose'][11, :3]-state['rigid_pose'][12, :3])
        checks = dict(opposite_sides=bool(side[0] == side[1] and side[3] == side[4] and side[0] != side[3]),
                      top=bool(rod[2] < x[:, 2].max() < rod[2]+.05),
                      ends_down=bool(selected[0, 2] < 0 and selected[4, 2] < 0),
                      off_ground=bool(x[:, 2].min() > .03),
                      quiet=bool(quiet > .99), released=bool(finger_distance > .07))
        return dict(success=all(checks.values()), checks=checks,
                    finger_distance_m=float(finger_distance), quiet_fraction=float(quiet))
    if env_id == 'Pour-v0':
        if fill_height is None or not np.isfinite(fill_height) or fill_height <= 0:
            raise ValueError('A source-backed beaker height is required')
        # Scene order is ground, moving bottle, target beaker. No lower Z
        # bound is present in the original Pour containment predicate.
        center = state['scene_actor_pose'][2,:2]
        bounded = (((x[:,:2]-center)**2).sum(1)<.04**2) & (x[:,2]<fill_height)
        h1,h2 = state['task_state']
        above_start = int(np.count_nonzero(bounded & (x[:,2]>h1)))
        above_end = int(np.count_nonzero(bounded & (x[:,2]>h2)))
        spilled = ~bounded & (x[:,2]<.001)
        qx,qy = state['scene_actor_pose'][1,4:6].astype(float)
        upright_z = 1-2*(qx*qx+qy*qy)
        checks = dict(above_start=above_start>100, above_end=above_end<10,
            spill=int(spilled.sum())<100, upright=bool(upright_z>=.866), quiet=bool(np.all(abs(state['qvel'])<.05)))
        return dict(success=all(checks.values()),checks=checks,above_start=above_start,above_end=above_end,
            contained_particles=int(bounded.sum()),contained_mass_kg=float(state['mass'][bounded].sum()),
            spilled_particles=int(spilled.sum()),spilled_mass_kg=float(state['mass'][spilled].sum()),upright_z=float(upright_z))
    raise InvalidArtifact('Outcome audit currently supports Fill, Hang, Pour, Excavate, Write and Pinch')


def audit(trace: Path, *, fill_height=None):
    manifest = validate_trace(trace)
    if manifest['fixture'].get('initial_state_contract', {}).get('version') != 2:
        raise InvalidArtifact('Outcome audit requires explicit portable task state')
    results = []
    mismatches = []
    for sample in manifest['samples']:
        state = load_frame(child_file(trace, sample['path']))
        metrics = outcome(state, manifest['fixture']['env_id'], fill_height=fill_height)
        if manifest['fixture']['env_id']=='Pinch-v0':
            recorded=sample['metrics'].get('progress')
            if type(recorded) not in (int,float) or not np.isfinite(recorded) or not metrics['progress_lower']<=recorded<=metrics['progress_upper']:
                raise InvalidArtifact(f'Pinch recorded progress disagrees with audited particle distances at step {sample["step"]}')
        if manifest['fixture']['env_id']=='Write-v0':
            # MS2 reports float64, MS3 evaluate() exposes float32 tensors.
            # Require the same exactly rounded float32 score, not a fitted tolerance.
            recorded=sample['metrics'].get('iou')
            if type(recorded) not in (int,float) or not np.isfinite(recorded) or np.float32(recorded)!=np.float32(metrics['iou']):
                raise InvalidArtifact(f'Write recorded IoU disagrees with audited height images at step {sample["step"]}')
        if metrics['success'] != sample['metrics']['success']:
            mismatches.append(sample['step'])
        results.append({'step': sample['step'], **metrics})
    successful = [r['step'] for r in results if r['success']]
    return dict(env_id=manifest['fixture']['env_id'], seed=manifest['fixture']['seed'],
                scope='Independent legacy outcome recomputation; not calibrated physics parity',
                recorded_labels_match=not mismatches, mismatched_steps=mismatches,
                first_success_step=min(successful) if successful else None,
                successful_samples=len(successful), final=results[-1])

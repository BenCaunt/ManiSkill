"""Independent host checks for native GPU controller/reset lifecycle records."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .job_archive import file_hash, inventory


KINDS = ('dict', 'flat', 'reconfigure')
STEP_LABELS = ('warmup-1', 'warmup-2', 'expected', 'dict-after', 'flat-after', 'reconfigure-after')
RECORDS = ('initial', 'checkpoint', 'checkpoint-flat', *STEP_LABELS,
           *[kind + suffix for kind in KINDS for suffix in ('-checkpoint', '-before')])


def expected_action(case, initial_qpos, initial_ee):
    """Recompute the frozen input recipe from recorded initial proprioception."""
    mode = case['control_mode']
    q = initial_qpos[:7]
    if mode == 'pd_joint_pos':
        action = q + np.array([.002, -.003, .001, -.002, .001, .002, -.001])
    elif mode == 'pd_joint_pos_vel':
        action = np.r_[q + .002, np.full(7, .02)]
    elif mode == 'pd_joint_delta_pos_vel':
        action = np.r_[np.full(7, .01), np.full(7, .02)]
    elif mode == 'pd_ee_pose':
        pose = np.asarray(initial_ee, dtype=np.float32)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError('Invalid initial end-effector pose')
        action = np.r_[pose[:3] + [.001, -.002, .001],
            Rotation.from_quat(pose[[4, 5, 6, 3]]).as_rotvec() + [.002, -.001, .003]]
    elif 'ee' in mode:
        dimensions = 3 if mode.endswith('_pos') else 6
        action = np.array([.01, -.02, .01, .03, -.02, .01])[:dimensions]
    else:
        action = np.full(7, .01)
    if case['task'] in ('Hang', 'Pour', 'Pinch'):
        action = np.r_[action, 0.]
    return action.astype(np.float32)


def evaluate(root, protocol):
    root = Path(root)
    execution = json.loads((root / 'execution.json').read_text())
    failures, reports = [], {}
    for key in ('input_sha256', 'image_id'):
        if execution[key] != protocol[key]:
            failures.append('Frozen identity mismatch: ' + key)
    if execution['exit_code'] != 0:
        failures.append('Worker suite failed; retained per-case errors follow')
    if file_hash(root / 'request.json') != protocol['request_sha256']:
        failures.append('Frozen request mismatch')
    if file_hash(root / 'probe.py') != protocol['probe_sha256']:
        failures.append('Actual probe source mismatch')
    request = json.loads((root / 'request.json').read_text())
    prefix = 'mani_skill/envs/softbody/'
    expected_sources = {p[len(prefix):]: value for p, value in request['source_files'].items()
        if p.startswith(prefix) and p.endswith('.py') and '/' not in p[len(prefix):]}
    if inventory(root / 'source') != expected_sources:
        failures.append('Actual task source mismatch')
    if request['cases'] != protocol['cases']:
        failures.append('Case matrix changed')

    def load(path):
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        if not data or any(v.dtype.kind not in 'fibu' or not np.isfinite(v).all() for v in data.values()):
            raise ValueError('Invalid numeric archive: ' + str(path))
        return data

    for case in protocol['cases']:
        name = case['name']; directory = root / name
        if not (directory / 'result.json').is_file() or not (directory / 'execution.json').is_file():
            failures.append(name + ': missing case record')
            continue
        result = json.loads((directory / 'result.json').read_text())
        run_status = json.loads((directory / 'execution.json').read_text())['exit_code']
        if result['case'] != case or result['source_sha256'] != protocol['probe_sha256']:
            failures.append(name + ': case/probe identity mismatch')
        if result['complete'] is not True or run_status != 0:
            failures.append(name + ': ' + result.get('error', 'incomplete worker execution'))
            continue
        backend = case['backend']
        if result['physics_system'] != ('PhysxGpuSystem' if backend == 'gpu' else 'PhysxCpuSystem'):
            failures.append(name + ': wrong physical backend')
        if (result['control_mode'] != case['control_mode'] or result['rigid_dt'] != float(np.float32(.002))
                or result['control_dt'] != .05 or result['mpm_substeps'] != 4):
            failures.append(name + ': incorrect control/step configuration')
        if result['reset_guard'] is not True:
            failures.append(name + ': reset-only assignment guard failed')
        if set(result['files']) != {label + '.npz' for label in RECORDS}:
            raise ValueError('Incomplete lifecycle numeric record')
        arrays = {}
        for label in RECORDS:
            path = directory / (label + '.npz')
            if file_hash(path) != result['files'][path.name]:
                failures.append(name + ': numeric digest mismatch: ' + label)
            arrays[label] = load(path)
        initial = arrays['initial']; checkpoint = arrays['checkpoint']
        dof = 9 if case['task'] in ('Hang', 'Pour', 'Pinch') else 7
        if initial['qpos'].shape != (dof,):
            raise ValueError('Incorrect robot state shape')
        expected = expected_action(case, initial['qpos'], result['initial_ee_pose_at_base'])
        if not np.array_equal(result['action'], expected):
            failures.append(name + ': prescribed action recipe changed')
        if tuple(step['label'] for step in result['steps']) != STEP_LABELS:
            raise ValueError('Incomplete control steps')
        for step in result['steps']:
            outcome = step['outcome']
            if (not np.isfinite(outcome['reward']) or type(outcome['terminated']) is not bool
                    or type(outcome['truncated']) is not bool):
                raise ValueError('Invalid step outcome')
            if 'ee' in case['control_mode'] and step['ik_success'] is not True:
                failures.append(name + ': IK failed at ' + step['label'])
        count = checkpoint['mpm/x'].shape[1]
        if checkpoint['mpm/x'].shape != (1, count, 3) or not 0 < count <= 65536:
            raise ValueError('Invalid checkpoint particle shape/count')
        shapes = {'mpm/' + k: (1, count, *trailing) for k, trailing in
                  dict(x=(3,), v=(3,), F=(3, 3), C=(3, 3), vc=()).items()}
        shapes.update({'mpm_material/' + k: (1, count, *trailing) for k, trailing in
            dict(particle_mass=(), particle_vol=(), particle_type=(),
                 particle_mu_lam_ys=(3,), particle_friction_cohesion=(3,)).items()})
        shapes.update({'mpm_drives/' + k: (1, dof) for k in ('position', 'velocity')})
        if not set(shapes).issubset(checkpoint) or any(checkpoint[k].shape != shape for k, shape in shapes.items()):
            raise ValueError('Incomplete or malformed checkpoint arrays')
        if set(checkpoint) != set(arrays['dict-checkpoint']):
            raise ValueError('Restored checkpoint schema differs')
        flattened = np.concatenate([v.reshape(1, -1) for v in checkpoint.values()], axis=1)
        if not np.array_equal(flattened, arrays['checkpoint-flat']['flat']):
            failures.append(name + ': saved flat state differs from checkpoint dictionary')
        for field in ('x', 'v', 'F', 'C', 'vc'):
            if not np.array_equal(checkpoint['mpm/' + field][0], arrays['warmup-2'][field]):
                failures.append(name + ': saved checkpoint is not the recorded live state')
        required_groups = {'mpm', 'mpm_material', 'mpm_drives', 'task'}
        if not required_groups.issubset({key.split('/')[0] for key in checkpoint}):
            raise ValueError('Missing checkpoint groups')
        if 'target_delta' in case['control_mode'] and not any(key.startswith('controller/arm/') for key in checkpoint):
            failures.append(name + ': missing target-relative controller memory')
        if tuple(replay['kind'] for replay in result['replays']) != KINDS:
            raise ValueError('Missing reset mode')
        errors, exact = {}, {}
        for replay in result['replays']:
            kind = replay['kind']; restored = arrays[kind + '-checkpoint']
            if (replay['saved_particles'] != count or replay['restored_particles'] != count
                    or replay['world_steps'] != (25 if backend == 'gpu' else None)
                    or replay['rebuilt_physics'] != (kind == 'reconfigure')):
                failures.append(name + ': incorrect reset/step accounting: ' + kind)
            if set(restored) != set(checkpoint):
                raise ValueError('Reset changed checkpoint field set')
            mismatches = [key for key, value in checkpoint.items()
                if key.split('/')[0] in protocol['checkpoint_exact_groups']
                and not np.array_equal(value, restored[key])]
            exact[kind] = not mismatches
            if mismatches:
                failures.append(name + ': checkpoint values changed after ' + kind + ': ' + ','.join(mismatches))
            actual, expected = arrays[kind + '-after'], arrays['expected']
            errors[kind] = {}
            for field in ('x', 'qpos', 'qvel', 'drive_position', 'drive_velocity'):
                if actual[field].shape != expected[field].shape:
                    raise ValueError('Reset changed physical trajectory shape')
                errors[kind][field] = float(np.max(np.abs(actual[field].astype(float) - expected[field])))
            if errors[kind]['qpos'] > protocol['joint_replay_limit']:
                failures.append(f'{name}: {kind} joint replay={errors[kind]["qpos"]} exceeds {protocol["joint_replay_limit"]}')
        robot_motion = float(np.max(np.abs(arrays['warmup-2']['qpos'] - initial['qpos'])))
        if robot_motion == 0:
            failures.append(name + ': no measured robot motion')
        reports[name] = dict(exact_checkpoints=exact, replay_errors=errors,
            robot_motion_max_abs=robot_motion, particle_count=count,
            changed_particle_count=any(replay['fresh_particles'] != count for replay in result['replays']))
    return dict(passed=not failures, failures=failures, reports=reports, scope=protocol['scope'], full_port_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path); parser.add_argument('protocol', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = evaluate(args.root, json.loads(args.protocol.read_text()))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result['passed'] else 1)

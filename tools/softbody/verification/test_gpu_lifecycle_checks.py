"""Fault injection for lifecycle verification; synthetic data are not physics evidence."""
import json

import numpy as np
import pytest

from tools.softbody.verification.gpu_lifecycle_checks import KINDS, STEP_LABELS, evaluate, expected_action
from tools.softbody.verification.job_archive import file_hash


@pytest.fixture
def record(tmp_path):
    root = tmp_path
    (root / 'source').mkdir(); (root / 'source/base_env.py').write_text('Synthetic source identity')
    (root / 'probe.py').write_text('Synthetic probe identity')
    case = dict(name='synthetic-gpu', task='Fill', backend='gpu', control_mode='pd_ee_target_delta_pose')
    (root / 'request.json').write_text(json.dumps(dict(cases=[case], source_files={
        'mani_skill/envs/softbody/base_env.py': file_hash(root / 'source/base_env.py')})))
    protocol = dict(input_sha256='fixture-input', image_id='fixture-image',
        request_sha256=file_hash(root / 'request.json'), probe_sha256=file_hash(root / 'probe.py'),
        cases=[case], joint_replay_limit=.001,
        checkpoint_exact_groups=['mpm', 'mpm_material', 'mpm_drives', 'controller', 'task', 'task_particles'],
        scope='Synthetic verifier fixture')
    (root / 'execution.json').write_text(json.dumps(dict(exit_code=0,
        input_sha256=protocol['input_sha256'], image_id=protocol['image_id'])))
    directory = root / case['name']; directory.mkdir()
    (directory / 'execution.json').write_text('{"exit_code":0}')
    initial = {k: np.zeros((2, *shape), np.float32) for k, shape in dict(x=(3,), v=(3,), F=(3, 3), C=(3, 3), vc=()).items()}
    initial.update(qpos=np.zeros(7, np.float32), qvel=np.zeros(7, np.float32),
        drive_position=np.zeros(7), drive_velocity=np.zeros(7))
    moved = {k: v.copy() for k, v in initial.items()}
    moved['qpos'][:] = .001; moved['x'][:] = .001
    checkpoint = {'mpm/' + k: moved[k][None] for k in ('x', 'v', 'F', 'C', 'vc')}
    checkpoint.update({'mpm_material/' + k: np.ones((1, 2, *shape)) for k, shape in
        dict(particle_mass=(), particle_vol=(), particle_type=(), particle_mu_lam_ys=(3,), particle_friction_cohesion=(3,)).items()})
    checkpoint.update({'mpm_drives/' + k: np.zeros((1, 7)) for k in ('position', 'velocity')})
    checkpoint['controller/arm/target_pose'] = np.array([[.1, .2, .3, 1., 0., 0., 0.]])
    checkpoint['task/target'] = np.zeros((1, 2))
    arrays = dict(initial=initial, checkpoint=checkpoint,
        **{'checkpoint-flat': {'flat': np.concatenate([v.reshape(1, -1) for v in checkpoint.values()], axis=1)}})
    arrays.update({label: moved for label in STEP_LABELS})
    arrays.update({kind + '-checkpoint': checkpoint for kind in KINDS})
    arrays.update({kind + '-before': moved for kind in KINDS})
    files = {}
    for label, data in arrays.items():
        path = directory / (label + '.npz'); np.savez_compressed(path, **data); files[path.name] = file_hash(path)
    report = dict(case=case, source_sha256=protocol['probe_sha256'], complete=True,
        physics_system='PhysxGpuSystem', control_mode=case['control_mode'], rigid_dt=float(np.float32(.002)),
        control_dt=.05, mpm_substeps=4, reset_guard=True, files=files,
        action=expected_action(case, initial['qpos'], None).tolist(), initial_ee_pose_at_base=[0., 0., 0., 1., 0., 0., 0.],
        steps=[dict(label=label, outcome=dict(reward=0., terminated=False, truncated=False), ik_success=True) for label in STEP_LABELS],
        replays=[dict(kind=kind, saved_particles=2, restored_particles=2, fresh_particles=3,
            world_steps=25, rebuilt_physics=kind == 'reconfigure') for kind in KINDS])
    (directory / 'result.json').write_text(json.dumps(report))
    return root, protocol


def change_result(root, mutate):
    path = root / 'synthetic-gpu/result.json'; result = json.loads(path.read_text())
    mutate(result); path.write_text(json.dumps(result))


def change_arrays(root, label, mutate):
    path = root / 'synthetic-gpu' / (label + '.npz')
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    mutate(data); np.savez_compressed(path, **data)
    change_result(root, lambda result: result['files'].update({path.name: file_hash(path)}))


def test_synthetic_complete_lifecycle(record):
    assert evaluate(*record)['passed']


def test_changed_controller_memory_fails_even_with_updated_digest(record):
    root, protocol = record
    change_arrays(root, 'flat-checkpoint', lambda a: a['controller/arm/target_pose'].__setitem__((0, 0), .10001))
    assert any('checkpoint values changed after flat' in f for f in evaluate(root, protocol)['failures'])


def test_flat_state_does_not_match_dictionary(record):
    root, protocol = record
    change_arrays(root, 'checkpoint-flat', lambda a: a['flat'].__setitem__((0, 0), 99.))
    assert 'synthetic-gpu: saved flat state differs from checkpoint dictionary' in evaluate(root, protocol)['failures']


@pytest.mark.parametrize('key,value', [('restored_particles', 3), ('world_steps', 26), ('rebuilt_physics', True)])
def test_wrong_reset_bookkeeping_rejected(record, key, value):
    root, protocol = record
    change_result(root, lambda d: d['replays'][0].update({key: value}))
    assert 'synthetic-gpu: incorrect reset/step accounting: dict' in evaluate(root, protocol)['failures']


def test_unstable_joint_replay_fails(record):
    root, protocol = record
    change_arrays(root, 'reconfigure-after', lambda a: a['qpos'].__setitem__(0, .1))
    assert any('reconfigure joint replay=' in f for f in evaluate(root, protocol)['failures'])


def test_false_ik_success_cannot_hide_failure(record):
    root, protocol = record
    change_result(root, lambda d: d['steps'][3].update(ik_success=False))
    assert 'synthetic-gpu: IK failed at dict-after' in evaluate(root, protocol)['failures']


def test_wrong_action_fails(record):
    root, protocol = record
    change_result(root, lambda d: d['action'].__setitem__(0, .1))
    assert 'synthetic-gpu: prescribed action recipe changed' in evaluate(root, protocol)['failures']


def test_changed_source_fails(record):
    root, protocol = record
    (root / 'source/base_env.py').write_text('Altered source')
    assert 'Actual task source mismatch' in evaluate(root, protocol)['failures']


def test_nonfinite_physical_state_rejected(record):
    root, protocol = record
    change_arrays(root, 'flat-after', lambda a: a['qvel'].__setitem__(0, np.nan))
    with pytest.raises(ValueError, match='Invalid numeric archive'):
        evaluate(root, protocol)


def test_missing_material_array_rejected(record):
    root, protocol = record
    change_arrays(root, 'checkpoint', lambda a: a.pop('mpm_material/particle_vol'))
    with pytest.raises(ValueError, match='Incomplete or malformed checkpoint arrays'):
        evaluate(root, protocol)

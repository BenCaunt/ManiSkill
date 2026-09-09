import hashlib
import json

import h5py
import numpy as np
import pytest

from softbody_lab.demo_inputs import export_episode, verify
from softbody_lab.replay_demo import capture_arguments
from softbody_lab.task_checks import outcome


@pytest.mark.parametrize('fault', [None, 'changed_actions', 'symlink', 'unpinned_dependency', 'expired_lease'])
def test_reference_job_freezes_only_reset_controls_and_pinned_dependencies(tmp_path, fault):
    import time
    from softbody_lab.remote_replay import prepare_reference_demo
    from softbody_lab.job_archive import archive_inventory, file_hash
    inputs = tmp_path/'input'; inputs.mkdir()
    np.save(inputs/'initial.npy', np.arange(10, dtype=np.float32))
    np.save(inputs/'actions.npy', np.zeros((2, 7), dtype=np.float32))
    metadata = dict(env_id='Excavate-v0', seed=2, steps=2, registered_horizon=250,
        env_kwargs=dict(obs_mode='none', control_mode='pd_joint_pos', reward_mode='dense'),
        control_mode='pd_joint_pos', reset_kwargs=dict(seed=2, target_num=432),
        files={p:file_hash(inputs/p) for p in ('initial.npy', 'actions.npy')})
    (inputs/'input.json').write_text(json.dumps(metadata))
    (inputs/'future_states.npy').write_bytes(b'Must never be uploaded')
    if fault == 'changed_actions': (inputs/'actions.npy').write_bytes(b'changed')
    if fault == 'symlink':
        target=tmp_path/'elsewhere'; (inputs/'initial.npy').rename(target); (inputs/'initial.npy').symlink_to(target)
    lease = tmp_path/'lease.json'
    lease.write_text(json.dumps({'terminate_at_epoch':time.time()+(-1 if fault=='expired_lease' else 2000)}))
    dependencies = {'warp':'a'*64, 'sdf':'b'*64}
    if fault == 'unpinned_dependency': dependencies['sdf'] = 'latest'
    output = tmp_path/'job'
    if fault:
        with pytest.raises(ValueError):
            prepare_reference_demo(inputs, output, lease, 'sha256:'+'c'*64, dependencies=dependencies)
        assert not output.exists()
    else:
        handle = prepare_reference_demo(inputs, output, lease, 'sha256:'+'c'*64, dependencies=dependencies)
        request = json.loads((output/'payload/job.json').read_text())
        assert request['role'] == 'reference'
        assert request['reference_dependencies'] == dependencies
        files = archive_inventory(output/'input.tgz')
        assert set(p for p in files if p.startswith('reference_input/')) == {
            'reference_input/input.json', 'reference_input/initial.npy', 'reference_input/actions.npy'}
        assert not any(p.startswith('source/') for p in files)
        assert handle['phase'] == 'prepared'


def test_export_keeps_outcomes_out_and_preserves_reset_contract(tmp_path):
    source, output = tmp_path/'source', tmp_path/'input'
    source.mkdir()
    metadata = dict(env_info=dict(env_id='Excavate-v0', max_episode_steps=250,
        env_kwargs=dict(obs_mode='none', reward_mode='dense', control_mode='pd_joint_pos')),
        episodes=[dict(episode_id=3, episode_seed=15, elapsed_steps=2,
            reset_kwargs=dict(seed=15, target_num=598), control_mode='pd_joint_pos', info=dict(success=True))])
    (source/'trajectory.json').write_text(json.dumps(metadata))
    initial, actions = np.arange(10, dtype=np.float32), np.zeros((2, 7), dtype=np.float32)
    with h5py.File(source/'trajectory.h5', 'w') as f:
        group=f.create_group('traj_3')
        group['env_init_state']=initial
        group['actions']=actions
        group['success']=np.ones(2, dtype=bool)
        group['future_states']=np.ones((2, 10))
    result=export_episode('Excavate-v0', 3, source, output, {})
    assert set(p.name for p in output.iterdir())=={'input.json', 'initial.npy', 'actions.npy'}
    assert 'info' not in result and 'success' not in (output/'input.json').read_text()
    np.testing.assert_array_equal(np.load(output/'initial.npy', allow_pickle=False), initial)
    args=capture_arguments(output)
    assert args['role']=='reference' and args['reset_kwargs']=={'options': {'target_num': 598}}
    with (output/'actions.npy').open('ab') as f:
        f.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        capture_arguments(output)


@pytest.mark.parametrize('lfs', [False, True])
def test_demo_verifies_git_and_lfs_objects(tmp_path, lfs):
    path=tmp_path/'data'
    path.write_bytes(b'original')
    entry=dict(size=8, oid=hashlib.sha1(b'blob 8\0original').hexdigest())
    if lfs:
        entry['lfs']={'oid':hashlib.sha256(b'original').hexdigest()}
    verify(path, entry)
    path.write_bytes(b'modified')
    with pytest.raises(ValueError, match='checksum'):
        verify(path, entry)


def test_excavate_outcome_preserves_strict_task_boundaries():
    state=dict(x=np.zeros((1000, 3), dtype=np.float32), v=np.zeros((1000, 3), dtype=np.float32),
        mass=np.full(1000, .001), task_state=np.array([500]))
    state['x'][:500,2]=.25
    result=outcome(state, 'Excavate-v0')
    assert result['success'] and result['lifted_particles']==500
    state['x'][:100,2]=.2
    assert not outcome(state,'Excavate-v0')['checks']['amount']  # 400 equals excluded lower bound
    state['x'][:100,2]=.25
    state['x'][:20,0]=.12
    assert not outcome(state,'Excavate-v0')['checks']['spill']  # 20 equals excluded spill bound
    state['x'][:,0]=0
    state['v'][:10,:]=.05
    assert not outcome(state,'Excavate-v0')['checks']['quiet']  # exactly 99% is insufficient

"""Synthetic artifact fault injection; no physics evidence."""
import json
import numpy as np
import pytest

from compare_backend_replays import describe
from softbody_lab.artifacts import TraceWriter, digest_arrays, initial_numeric_state


def trace(root, role):
    pose = np.array([[0., 0., 0., 1., 0., 0., 0.]])
    state = dict(x=np.array([[0., 0., .5], [.1, 0., .5]]), v=np.zeros((2, 3)),
                 F=np.tile(np.eye(3), (2, 1, 1)), C=np.zeros((2, 3, 3)), vc=np.zeros(2),
                 mass=np.ones(2), qpos=np.zeros(2), qvel=np.zeros(2), sim_state=np.zeros(10),
                 drive_position=np.zeros(2), drive_velocity=np.zeros(2),
                 rigid_pose=np.repeat(pose, 5, axis=0), rigid_velocity=np.zeros((5, 6)),
                 root_pose=pose, root_velocity=np.zeros((1, 6)),
                 scene_actor_pose=np.repeat(pose, 5, axis=0), scene_actor_velocity=np.zeros((5, 6)),
                 task_state=np.array([2.]))
    fixture = dict(env_id='Excavate-v0', seed=1, initial_state_contract=dict(version=2,
        root_kind='fixed', derived_rigid_indices=[0],
        scene_actor_names=['ground', 'wall_0', 'wall_1', 'wall_2', 'wall_3'],
        scene_actor_types=['static']+['kinematic']*4))
    fixture['initial_numeric_sha256'] = digest_arrays(initial_numeric_state(state, fixture))
    provenance = dict(role=role, candidate_execution=dict(requested_sim_backend='physx_cuda',
        actual_sim_backend='physx_cuda', gpu_sim_enabled=True, mpm_device='cuda'))
    writer = TraceWriter(root, fixture=fixture, provenance=provenance, steps=1)
    writer.frame(time_s=0., state=state, metrics={'success': True})
    writer.frame(time_s=.05, state=state, metrics={'success': True}, action=[0., 0.])
    writer.finish()
    return root


@pytest.mark.parametrize('fault', [None, 'actual_backend', 'gpu_flag', 'mpm_device', 'actions', 'label'])
def test_backend_diagnostic_rejects_wrong_execution_and_audits_labels(tmp_path, fault):
    reference = trace(tmp_path/'reference', 'reference')
    candidate = trace(tmp_path/'candidate', 'candidate')
    path = candidate/'manifest.json'; manifest = json.loads(path.read_text())
    execution = manifest['provenance']['candidate_execution']
    if fault == 'actual_backend': execution['actual_sim_backend'] = 'physx_cpu'
    if fault == 'gpu_flag': execution['gpu_sim_enabled'] = False
    if fault == 'mpm_device': execution['mpm_device'] = 'cpu'
    if fault == 'actions': manifest['actions'][0][0] = .1
    if fault == 'label': manifest['samples'][-1]['metrics']['success'] = False
    path.write_text(json.dumps(manifest))
    if fault in ('actual_backend', 'gpu_flag', 'mpm_device', 'actions'):
        with pytest.raises(ValueError): describe(reference, candidate, 'physx_cuda')
    else:
        result = describe(reference, candidate, 'physx_cuda')
        assert result['label_mismatches'] == {'reference': [], 'candidate': [1] if fault == 'label' else []}
        assert result['first_success_step'] == {'reference': 0, 'candidate': 0}
        assert not any(result['max_errors'].values())

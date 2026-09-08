"""Synthetic verifier faults; these tests are not native batching evidence."""
import numpy as np
import pytest

from tools.softbody.verification.gpu_batch_checks import (check_snapshot, state_changes, particle_errors,
    reduction_indices, check_reduced_checkpoint)


@pytest.fixture
def snapshot():
    data = dict(counts=np.array([2, 3]), qpos=np.zeros((2, 7)), qvel=np.zeros((2, 7)),
                drive_position=np.zeros((2, 7)), drive_velocity=np.zeros((2, 7)))
    data['state/mpm_meta/count'] = data['counts'][:, None].copy()
    data['state/mpm_meta/mask'] = np.arange(4)[None] < data['counts'][:, None]
    for key, shape in dict(x=(3,), v=(3,), F=(3, 3), C=(3, 3), vc=(), mass=()).items():
        exposed = 'state/mpm_material/particle_mass' if key == 'mass' else 'state/mpm/' + key
        data[exposed] = np.zeros((2, 4, *shape), np.float32)
        for i, n in enumerate(data['counts']):
            data[f'actual/{i}/' + key] = np.full((n, *shape), i+1, np.float32)
            data[exposed][i, :n] = data[f'actual/{i}/' + key]
    return data


def test_masked_state_matches_actual_solver_particles(snapshot):
    assert not check_snapshot(snapshot, 2, 4)


def test_padding_cannot_be_reported_as_real_particles(snapshot):
    snapshot['state/mpm_meta/mask'][0, 3] = True
    assert any('mask' in f for f in check_snapshot(snapshot, 2, 4))


def test_nonzero_padding_is_rejected(snapshot):
    snapshot['state/mpm/x'][0, 3] = 1
    assert any('padding' in f for f in check_snapshot(snapshot, 2, 4))


def test_swapped_environment_particles_are_rejected(snapshot):
    snapshot['state/mpm/x'][0, :2] = snapshot['actual/1/x'][:2]
    assert any('differ from actual' in f for f in check_snapshot(snapshot, 2, 4))


def test_material_mass_must_come_from_actual_model(snapshot):
    snapshot['state/mpm_material/particle_mass'][0, 0] *= 2
    assert any('differ from actual' in f for f in check_snapshot(snapshot, 2, 4))


def test_partial_reset_detects_other_environment_target_changes(snapshot):
    snapshot['state/controller/target_qpos'] = np.ones((2, 7))
    changed = {k: v.copy() for k, v in snapshot.items()}
    changed['state/controller/target_qpos'][1, 0] = 0
    assert state_changes(snapshot, 1, changed, 1) == ['state/controller/target_qpos']
    assert not state_changes(snapshot, 0, changed, 0)


def test_missing_controller_checkpoint_field_is_not_ignored(snapshot):
    snapshot['state/controller/target_qpos'] = np.ones((2, 7))
    changed = {k: v.copy() for k, v in snapshot.items() if 'controller' not in k}
    assert state_changes(snapshot, 1, changed, 1) == ['state/controller/target_qpos']


def test_trajectory_comparison_uses_actual_solver_state(snapshot):
    changed = {k: v.copy() for k, v in snapshot.items()}
    changed['actual/0/x'][0, 0] += .125
    assert particle_errors(snapshot, 0, changed, 0)['x'] == .125


def test_reduction_keeps_and_remaps_goal_particles_without_changing_other_state():
    case = dict(capacity=12,preserve_task_particle_indices=True)
    before = {'counts':np.array([10,9]),'state/mpm/x':np.arange(72).reshape(2,12,3),
        'state/mpm_meta/count':np.array([[10],[9]]),
        'state/mpm_meta/mask':np.arange(12)[None]<np.array([[10],[9]]),
        'state/task/selected_indices':np.array([[1,3,4,6,9],[0,1,2,3,4]]),
        'state/actors/rod':np.arange(14).reshape(2,7)}
    keep = np.array([0,1,2,3,4,6,8,9])
    np.testing.assert_array_equal(reduction_indices(case,before),keep)
    reduced = {k:v[:1].copy() for k,v in before.items() if k.startswith('state/')}
    reduced['state/mpm/x'][:] = 0
    reduced['state/mpm/x'][0,:8] = before['state/mpm/x'][0,keep]
    reduced['state/mpm_meta/count'][:] = 8
    reduced['state/mpm_meta/mask'][:] = np.arange(12)<8
    reduced['state/task/selected_indices'][:] = [1,3,4,5,7]
    assert not check_reduced_checkpoint(case,before,reduced)
    reduced['state/task/selected_indices'][0,-1] = 9
    assert any('selected_indices' in f for f in check_reduced_checkpoint(case,before,reduced))
    reduced['state/actors/rod'][0,0] += 1
    assert any('actors/rod' in f for f in check_reduced_checkpoint(case,before,reduced))
    before['state/task/selected_indices'][0,-1] = 10
    with pytest.raises(ValueError,match='Invalid task particle'):
        reduction_indices(case,before)

"""Synthetic verifier faults; these tests are not native batching evidence."""
import numpy as np
import pytest

from tools.softbody.verification.gpu_batch_checks import check_snapshot, state_changes, particle_errors


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

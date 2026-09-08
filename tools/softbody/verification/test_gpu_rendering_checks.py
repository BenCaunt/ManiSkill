"""Synthetic pixel geometry tests, not renderer execution evidence."""
import numpy as np
import pytest

from tools.softbody.verification.gpu_rendering_checks import measure_frame


@pytest.fixture
def frame():
    centers = np.array([[0., 0., -.1], [.01, 0., -.1]], np.float32)
    position = np.zeros((1, 4, 4, 3), np.int16)
    position[..., 2] = -98; position[:, 2:, :, 0] = 10
    segmentation = np.ones((1, 4, 4, 1), np.int16); segmentation[:, 2:] = 2
    data = dict(particle_ids=np.array([1, 2]), particle_x=centers,
        position=position, segmentation=segmentation, depth=np.full((1, 4, 4, 1), 98, np.int16),
        rgb=np.zeros((1, 4, 4, 3), np.uint8), cam2world_gl=np.eye(4)[None],
        extrinsic_cv=np.diag([1., -1., -1., 1.])[:3][None],
        **{'before/mpm/x': centers[None].copy(), 'after/mpm/x': centers[None].copy()})
    return data, .0025, dict(minimum_particle_pixels=10, surface_distance_error_limit_m=.003)


def test_quantized_sphere_surface_fixture(frame):
    result = measure_frame(*frame)
    assert not result['failures'] and result['particle_pixels'] == 16


def test_stale_pixels_after_actual_particle_motion_fail(frame):
    data, radius, protocol = frame
    for key in ('particle_x', 'before/mpm/x', 'after/mpm/x'):
        data[key][..., 0] += .02
    assert any('do not match physical spheres' in f for f in measure_frame(data, radius, protocol)['failures'])


def test_rendering_must_not_advance_physics(frame):
    data, radius, protocol = frame
    data['before/mpm/x'][0, 0, 0] += .001
    assert 'Rendering changed physical/checkpoint state' in measure_frame(data, radius, protocol)['failures']


def test_missing_particle_pixels_fail(frame):
    data, radius, protocol = frame; data['segmentation'].fill(0)
    assert 'Too few actual particle pixels' in measure_frame(data, radius, protocol)['failures']


def test_corrupted_camera_frame_fails(frame):
    data, radius, protocol = frame; data['cam2world_gl'][0, 0, 3] = .5
    assert 'Camera coordinate transforms disagree' in measure_frame(data, radius, protocol)['failures']


def test_unrepresentable_signed_segmentation_ids_are_rejected(frame):
    data, radius, protocol = frame
    data['particle_ids'] = np.array([32768, 32769])
    data['segmentation'][0, :2] = -32768; data['segmentation'][0, 2:] = -32767
    assert 'Particle scene IDs exceed the segmentation texture range' in measure_frame(data, radius, protocol)['failures']


def test_aliased_particle_segmentation_cannot_be_accepted(frame):
    data, radius, protocol = frame; data['particle_ids'] = np.array([1, 1])
    with pytest.raises(ValueError, match='ambiguous'):
        measure_frame(data, radius, protocol)


def test_inactive_pool_particles_must_not_leave_ghost_pixels(frame):
    data, radius, protocol = frame
    protocol['require_visual_pool_checks'] = True
    data['pool_ids'] = np.array([1, 2, 3])
    data['segmentation'][0, 0, 0, 0] = 3
    assert any('Inactive pooled particles remain visible' in f for f in measure_frame(data, radius, protocol)['failures'])


def test_rendering_must_not_write_native_rigid_buffer(frame):
    data, radius, protocol = frame
    protocol['require_native_buffer_checks'] = True
    data['before_native_rigid'] = np.zeros((2, 13), np.float32)
    data['after_native_rigid'] = np.ones((2, 13), np.float32)
    assert 'Rendering changed the native rigid CUDA buffer' in measure_frame(data, radius, protocol)['failures']


def test_robot_pixels_must_follow_physical_pose(frame):
    data, radius, protocol = frame
    protocol.update(require_rigid_visual_bounds=True, minimum_rigid_pixels=1, rigid_bound_error_limit_m=.003)
    data['rigid_visual_ids'] = np.arange(10, 17)
    data['rigid_visual_bounds'] = np.tile(np.array([[-.01]*3, [.01]*3]), (7, 1, 1))
    data['rigid_visual_poses'] = np.tile(np.eye(4), (7, 1, 1))
    data['rigid_visual_poses'][:, 2, 3] = -.1
    data['segmentation'][0, 0, 0, 0] = 10
    assert not measure_frame(data, radius, protocol)['failures']
    data['rigid_visual_poses'][0, 0, 3] = .2
    assert any('Robot pixels fall outside' in f for f in measure_frame(data, radius, protocol)['failures'])


def test_returned_observation_cannot_reuse_stale_pixels(frame):
    data, radius, protocol = frame
    protocol['require_observation_pixels'] = True
    for key in ('rgb', 'depth', 'segmentation'):
        data['observation_' + key] = data[key].copy()
    data['observation_depth'][0, 0, 0, 0] += 1
    assert any('Returned observation differs' in f for f in measure_frame(data, radius, protocol)['failures'])

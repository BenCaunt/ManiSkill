"""Controller contract tests; native multi-scene IK is verified separately."""
from types import SimpleNamespace

import numpy as np
import pytest
import sapien
import torch
from scipy.spatial.transform import Rotation

from mani_skill.agents.controllers import PDJointPosController
from mani_skill.envs.softbody.controllers import LegacyEEPosController, LegacyEEPoseController
from mani_skill.utils.structs import Pose


def controller(cls=LegacyEEPoseController, count=2):
    value = cls.__new__(cls)
    value.articulation = SimpleNamespace(device=torch.device('cpu'))
    value.scene = SimpleNamespace(num_envs=count, _reset_mask=torch.ones(count, dtype=torch.bool))
    value.config = SimpleNamespace(frame='ee', use_delta=True, use_target=True, interpolate=False)
    return value


@pytest.mark.parametrize('kind,frame,delta', [
    ('pos', 'base', True), ('pos', 'ee', True), ('pos', 'base', False),
    ('pose', 'base', True), ('pose', 'ee', True), ('pose', 'ee_align', True), ('pose', 'base', False),
])
def test_each_row_preserves_scalar_legacy_pose_composition(kind, frame, delta):
    cls = LegacyEEPosController if kind == 'pos' else LegacyEEPoseController
    value = controller(cls); value.config.frame = frame; value.config.use_delta = delta
    poses = [sapien.Pose([.2, -.1, .3], Rotation.from_rotvec([.3, -.2, .4]).as_quat()[[3, 0, 1, 2]]),
             sapien.Pose([-.1, .2, .5], Rotation.from_rotvec([-.2, .5, -.1]).as_quat()[[3, 0, 1, 2]])]
    actions = np.array([[.02, -.03, .01, .04, -.02, .03], [-.01, .02, -.04, -.03, .01, -.05]], np.float32)
    if kind == 'pos': actions = actions[:, :3]
    original = actions.copy()
    output = value.compute_target_pose(Pose.create(poses), torch.tensor(actions)).raw_pose.numpy()
    for i, (previous, action) in enumerate(zip(poses, actions)):
        d = (sapien.Pose(action) if kind == 'pos' else
             sapien.Pose(action[:3], Rotation.from_rotvec(action[3:]).as_quat()[[3, 0, 1, 2]]))
        expected = d if not delta else previous * d if frame == 'ee' else d * previous
        if delta and frame == 'ee_align': expected.p = previous.p + action[:3]
        assert np.array_equal(output[i], np.r_[expected.p, expected.q])
    assert np.array_equal(actions, original)
    assert not np.array_equal(output[0], output[1])


def test_rotation_norm_clipping_is_per_environment():
    value = controller(); value.config.rot_bound = .1
    value.action_space_low = torch.full((6,), -.1); value.action_space_high = torch.full((6,), .1)
    action = torch.tensor([[0., 0., 0., 1., 1., 1.], [0., 0., 0., .2, 0., 0.]])
    decoded = value._clip_and_scale_action(action)
    torch.testing.assert_close(torch.linalg.vector_norm(decoded[:, 3:], dim=1), torch.tensor([.1, .02]))
    assert action[0, 3:].tolist() == [1., 1., 1.]


def test_partial_reset_keeps_untouched_ee_and_joint_targets():
    value = controller(); value.active_joint_indices = torch.tensor([0, 1])
    value.articulation.get_qpos = lambda: torch.tensor([[.1, .2], [.3, .4]])
    value.articulation.pose = Pose.create_from_pq(p=torch.zeros((2, 3)))
    value.ee_link = SimpleNamespace(pose=Pose.create_from_pq(p=torch.tensor([[.1, .2, .3], [.4, .5, .6]])))
    value._start_qpos = torch.ones((2, 2)); value._target_qpos = torch.full((2, 2), 2.)
    value._target_pose = Pose.create_from_pq(p=torch.ones((2, 3)))
    value.scene._reset_mask[:] = torch.tensor([False, True])
    old = value._target_pose.raw_pose.clone()
    value.reset()
    assert torch.equal(value._target_pose.raw_pose[0], old[0])
    assert torch.equal(value._target_pose.raw_pose[1], value.ee_link.pose.raw_pose[1])
    assert value._start_qpos[0].tolist() == [1., 1.] and value._target_qpos[0].tolist() == [2., 2.]
    assert torch.equal(value._target_qpos[1], value.articulation.get_qpos()[1])


@pytest.mark.parametrize('failure', ['reported', 'nonfinite'])
def test_ik_uses_each_native_model_and_failure_keeps_only_that_rows_joints(failure):
    value = controller(LegacyEEPosController)
    value.active_joint_indices = torch.tensor([0, 2]); value._joint_indices = np.array([0, 2])
    initial = torch.tensor([[.1, .2, .3], [.4, .5, .6]])
    value.articulation.get_qpos = lambda: initial
    value._native_robots = [SimpleNamespace(dof=3), SimpleNamespace(dof=3)]
    value._target_pose = Pose.create_from_pq(p=torch.tensor([[.1, .2, .3], [.4, .5, .6]]))
    value.qmasks = [np.array([True, False, True]), np.array([True, False, True])]
    value.ee_link_indices = [4, 7]; calls = []
    def model(index):
        def solve(link, pose, initial_qpos, active_qmask, max_iterations):
            calls.append((index, link, pose.p.copy(), initial_qpos.copy(), active_qmask.copy(), max_iterations))
            result = np.array([1., 2., 3.]) if index == 0 else np.array([4., 5., 6.])
            if index == 1 and failure == 'nonfinite': result[0] = np.nan
            return result, index == 0 or failure == 'nonfinite', None
        return SimpleNamespace(compute_inverse_kinematics=solve)
    value.pmodels = [model(0), model(1)]
    value._preprocess_action = lambda action: action
    applied = []; value.set_drive_targets = lambda target: applied.append(target.clone())
    value.set_action(torch.tensor([[.01, 0., 0.], [0., -.02, 0.]]))
    assert [(c[0], c[1]) for c in calls] == [(0, 4), (1, 7)]
    assert np.array_equal(calls[0][3], initial[0].numpy()) and np.array_equal(calls[1][3], initial[1].numpy())
    assert value.last_ik_success.tolist() == [True, False]
    assert applied[0][0].tolist() == [1., 3.]
    assert torch.equal(applied[0][1], initial[1, [0, 2]])


def test_initialization_builds_a_distinct_model_from_each_native_articulation(monkeypatch):
    value = controller(); value.config.ee_link = 'tool'
    models = [object(), object()]; links = [object(), object()]
    robots = [SimpleNamespace(dof=3, links=[object(), links[0]], create_pinocchio_model=lambda: models[0]),
              SimpleNamespace(dof=4, links=[object(), object(), links[1]], create_pinocchio_model=lambda: models[1])]
    value.articulation._objs = robots; value.articulation.links_map = {'tool': SimpleNamespace(_objs=links)}
    monkeypatch.setattr(PDJointPosController, '_initialize_joints', lambda self: setattr(self, 'active_joint_indices', torch.tensor([0, 2])))
    value._initialize_joints()
    assert value.pmodels == models and value.ee_link_indices == [1, 2]
    assert [mask.tolist() for mask in value.qmasks] == [[True, False, True], [True, False, True, False]]


def test_checkpoint_target_rows_are_owned_and_shape_checked():
    value = controller(); saved = Pose.create_from_pq(p=torch.ones((2, 3))).raw_pose
    value.set_state({'target_pose': saved}); saved[:, 0] = 99.
    assert value.get_state()['target_pose'][:, 0].tolist() == [1., 1.]
    before = value._target_pose.raw_pose.clone()
    for invalid in (saved[:1], torch.full((2, 7), float('nan'))):
        with pytest.raises(ValueError, match='one finite legacy target pose'):
            value.set_state({'target_pose': invalid})
        assert torch.equal(value._target_pose.raw_pose, before)

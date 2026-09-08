"""CPU API-contract checks; native partial-reset isolation is tested on GPU."""
from types import SimpleNamespace

import pytest
import torch

from mani_skill.envs import scene as scene_module
from mani_skill.envs.scene import ManiSkillScene


@pytest.fixture
def scene(monkeypatch):
    class NativeRecorder:
        def __init__(self):
            self.calls = {}

        def __getattr__(self, name):
            def record(*args):
                self.calls[name] = args
            return record

    monkeypatch.setattr(scene_module.physx, 'PhysxGpuSystem', NativeRecorder)
    monkeypatch.setattr(scene_module.sapien, 'CudaArray', lambda tensor: tensor)
    def apply_selected(system, indices):
        system.calls['native_selected_actors'] = (indices,)
    monkeypatch.setattr(scene_module, 'apply_selected_actor_data', apply_selected)
    value = ManiSkillScene.__new__(ManiSkillScene)
    value.px = NativeRecorder(); value.device = torch.device('cpu')
    value._needs_fetch = False
    value._use_native_actor_reset = True
    value._reset_mask = torch.tensor([False, True, False])
    # Native indices deliberately differ from environment indices and order.
    value.non_static_actors = [SimpleNamespace(_scene_idxs=torch.tensor([2, 1, 0]),
                                               _body_data_index=torch.tensor([4, 8, 2]))]
    value.articulations = {'robot': SimpleNamespace(_scene_idxs=torch.tensor([0, 2, 1]),
                                                     _data_index=torch.tensor([5, 9, 3]))}
    return value


def test_partial_reset_selects_native_actors_and_roots(scene):
    scene._gpu_apply_all()
    assert scene.px.calls['native_selected_actors'][0].tolist() == [8]
    assert 'gpu_apply_rigid_dynamic_data' not in scene.px.calls
    for name, expected in [('articulation_root_pose', [3]),
                           ('articulation_root_velocity', [3])]:
        handle, = scene.px.calls['gpu_apply_' + name]
        assert handle.dtype == torch.int32 and handle.is_contiguous() and handle.tolist() == expected
        assert any(handle is buffer for buffer in scene._gpu_reset_index_buffers)
    # Indexed joint calls in the pinned SAPIEN version would update the wrong rows.
    for name in ('qpos', 'qvel', 'qf', 'target_position', 'target_velocity'):
        assert scene.px.calls['gpu_apply_articulation_' + name] == ()
    assert scene._needs_fetch


def test_checkpoint_selection_overrides_restored_global_mask(scene):
    scene._reset_mask[:] = True
    scene._gpu_apply_all([2])
    assert scene.px.calls['gpu_apply_articulation_root_pose'][0].tolist() == [9]
    assert scene.px.calls['native_selected_actors'][0].tolist() == [4]
    assert 'gpu_apply_rigid_dynamic_data' not in scene.px.calls
    assert scene._reset_mask.all()


def test_full_reset_keeps_original_native_api_calls(scene):
    scene._gpu_apply_all([2, 0, 1])
    assert len(scene.px.calls) == 8 and all(args == () for args in scene.px.calls.values())


def test_ordinary_rigid_scene_does_not_require_native_actor_extension(scene):
    del scene._use_native_actor_reset
    scene._gpu_apply_all([1])
    assert scene.px.calls['gpu_apply_rigid_dynamic_data'] == ()
    assert 'native_selected_actors' not in scene.px.calls


def test_mpm_scene_opt_in_is_reinstalled_on_reconfiguration(monkeypatch):
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.envs.softbody.base_env import MPMBaseEnv
    monkeypatch.setattr(BaseEnv,'_setup_scene',lambda self:setattr(self,'scene',SimpleNamespace()))
    env=MPMBaseEnv.__new__(MPMBaseEnv)
    env._setup_scene(); first=env.scene
    assert first._use_native_actor_reset
    env._setup_scene()
    assert env.scene is not first and env.scene._use_native_actor_reset


def test_empty_selected_object_groups_do_not_fall_back_to_all(scene):
    scene._reset_mask = torch.tensor([False, False, True])
    scene.non_static_actors[0]._scene_idxs[:] = 0
    scene.articulations['robot']._scene_idxs[:] = 1
    scene._gpu_apply_all()
    assert 'gpu_apply_rigid_dynamic_data' not in scene.px.calls
    assert 'native_selected_actors' not in scene.px.calls
    assert 'gpu_apply_articulation_root_pose' not in scene.px.calls
    assert 'gpu_apply_articulation_root_velocity' not in scene.px.calls


@pytest.mark.parametrize('indices', [[], [1, 1], [-1], [3], [0.5], [[1]], [True]])
def test_invalid_explicit_selection_does_not_apply_any_native_buffer(scene, indices):
    with pytest.raises(ValueError, match='indices'):
        scene._gpu_apply_all(indices)
    assert not scene.px.calls and not scene._needs_fetch


def test_pending_apply_must_be_fetched_before_reset(scene):
    scene._needs_fetch = True
    with pytest.raises(AssertionError, match='must call _gpu_fetch_all'):
        scene._gpu_apply_all()
    assert not scene.px.calls

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
    value = ManiSkillScene.__new__(ManiSkillScene)
    value.px = NativeRecorder(); value.device = torch.device('cpu')
    value._needs_fetch = False
    value._reset_mask = torch.tensor([False, True, False])
    # Native indices deliberately differ from environment indices and order.
    value.non_static_actors = [SimpleNamespace(_scene_idxs=torch.tensor([2, 1, 0]),
                                               _body_data_index=torch.tensor([4, 8, 2]))]
    value.articulations = {'robot': SimpleNamespace(_scene_idxs=torch.tensor([0, 2, 1]),
                                                     _data_index=torch.tensor([5, 9, 3]))}
    return value


def test_partial_reset_selects_roots_and_uses_compatible_full_actor_layout(scene):
    scene._gpu_apply_all()
    assert scene.px.calls['gpu_apply_rigid_dynamic_data'] == ()
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
    assert scene.px.calls['gpu_apply_rigid_dynamic_data'] == ()
    assert scene._reset_mask.all()


def test_full_reset_keeps_original_native_api_calls(scene):
    scene._gpu_apply_all([2, 0, 1])
    assert len(scene.px.calls) == 8 and all(args == () for args in scene.px.calls.values())


def test_empty_selected_object_groups_do_not_fall_back_to_all(scene):
    scene._reset_mask = torch.tensor([False, False, True])
    scene.non_static_actors[0]._scene_idxs[:] = 0
    scene.articulations['robot']._scene_idxs[:] = 1
    scene._gpu_apply_all()
    assert 'gpu_apply_rigid_dynamic_data' not in scene.px.calls
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

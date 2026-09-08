"""CPU unit checks of render-buffer ownership; native GPU evidence is separate."""
from types import SimpleNamespace

import pytest
import torch

from mani_skill.envs import scene as scene_module
from mani_skill.envs.scene import ManiSkillScene


@pytest.fixture
def scene(monkeypatch):
    scene = ManiSkillScene.__new__(ManiSkillScene)
    rigid = torch.arange(26, dtype=torch.float32).reshape(2, 13)
    scene.px = SimpleNamespace(cuda_rigid_body_data=SimpleNamespace(torch=lambda: rigid, shape=rigid.shape))
    scene._gpu_render_only_bodies = [object()]
    scene._gpu_render_only_poses = torch.tensor([[.2, .3, .4, 1., 0., 0., 0.]])
    scene._gpu_render_pose_buffer = None
    scene.actors = {}; scene.articulations = {}
    # Exercise the real buffer composition method without pretending to execute
    # CUDA or Vulkan on a CPU host.
    monkeypatch.setattr(scene_module.sapien, 'CudaArray', lambda tensor: tensor)
    return scene


def test_render_buffer_cannot_write_back_to_physics(scene):
    rigid = scene.px.cuda_rigid_body_data.torch()
    original = rigid.clone(); poses = scene._gpu_render_only_poses.clone()
    rendered = scene._get_cuda_render_poses()
    torch.testing.assert_close(rendered[:2], rigid, rtol=0, atol=0)
    torch.testing.assert_close(rendered[2:, :7], poses, rtol=0, atol=0)
    rendered.fill_(99.)
    torch.testing.assert_close(rigid, original, rtol=0, atol=0)
    torch.testing.assert_close(scene._gpu_render_only_poses, poses, rtol=0, atol=0)
    scene._gpu_render_only_poses[0, 0] = .7
    rigid[0, 0] = .9
    refreshed = scene._get_cuda_render_poses()
    assert refreshed.data_ptr() == rendered.data_ptr()
    assert refreshed[0, 0] == .9 and refreshed[2, 0] == .7


def test_visual_indices_follow_native_rigid_rows(scene):
    assert scene._get_all_render_bodies() == [(scene._gpu_render_only_bodies[0], 2)]


def test_changed_physics_topology_requires_registration_rebuild(scene):
    scene._get_cuda_render_poses()
    scene.px.cuda_rigid_body_data.torch = lambda: torch.zeros((3, 13))
    with pytest.raises(RuntimeError, match='topology changed'):
        scene._get_cuda_render_poses()


def test_ordinary_scene_keeps_original_native_buffer(scene):
    scene._gpu_render_only_bodies = []
    assert scene._get_cuda_render_poses() is scene.px.cuda_rigid_body_data


def test_cuda_alias_registration_and_physical_body_rejection(scene, monkeypatch):
    scene.gpu_sim_enabled = True; scene.device = torch.device('cuda'); scene.render_system_group = None
    monkeypatch.setattr(scene_module, 'SAPIEN_RENDER_SYSTEM', '3.0')
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    poses = SimpleNamespace(shape=(1, 7), dtype=torch.float32,
                            device=torch.device('cuda:0'), is_contiguous=lambda: True)
    body = SimpleNamespace(entity=SimpleNamespace(find_component_by_type=lambda cls: None))
    scene.set_gpu_render_only_bodies([body], poses)
    assert scene._gpu_render_only_poses is poses
    body.entity.find_component_by_type = lambda cls: object()
    with pytest.raises(ValueError, match='cannot override physical bodies'):
        scene.set_gpu_render_only_bodies([body], poses)
    scene.render_system_group = object()
    with pytest.raises(RuntimeError, match='Release render groups'):
        scene.set_gpu_render_only_bodies([body], poses)

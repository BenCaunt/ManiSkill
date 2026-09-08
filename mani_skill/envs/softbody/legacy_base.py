"""Shared setup for the original ManiSkill 2 MPM task family."""
from pathlib import Path
import os
import numpy as np
import sapien
from mani_skill.utils.structs.types import SimConfig, SceneConfig, DefaultMaterialsConfig
from .base_env import MPMBaseEnv
from .mpm import wp


class LegacyMPMEnv(MPMBaseEnv):
    def __init__(self, *args, legacy_asset_dir=None, sdf_cache_dir=None, **kwargs):
        asset_dir = legacy_asset_dir or os.environ.get('MANISKILL_LEGACY_ASSET_DIR')
        if not asset_dir:
            raise ValueError('Provide legacy_asset_dir with the pinned ManiSkill 2 assets')
        self.legacy_asset_dir = Path(asset_dir).resolve()
        self.sdf_cache_dir = Path(sdf_cache_dir or os.environ.get('XDG_CACHE_HOME', '/tmp')) / 'maniskill-legacy-mpm-sdf'
        self.collision_geometry = []
        kwargs.setdefault('robot_uids', 'none')  # agent is explicitly constructed below
        kwargs.setdefault('control_mode', 'pd_joint_delta_pos')
        super().__init__(*args, **kwargs)

    def _restore_requested_drives(self):
        saved = getattr(self, '_legacy_requested_drive_tensor', None)
        if saved is not None:
            self.agent.robot.set_joint_drive_targets(saved, self.agent.robot.active_joints)
            self.scene.px.gpu_apply_articulation_target_position()
            self._legacy_requested_drive_tensor = None
        for joint, target in getattr(self, '_legacy_requested_drives', ()):
            joint.set_drive_target(target)
        self._legacy_requested_drives = ()

    def reset(self, *args, **kwargs):
        self._restore_requested_drives()
        return super().reset(*args, **kwargs)

    def _before_simulation_step(self):
        super()._before_simulation_step()
        if self.gpu_sim_enabled:
            targets = self.agent.robot.get_drive_targets().clone()
            limits = self.agent.robot.get_qlimits()
            clipped = targets.clamp(min=limits[..., 0], max=limits[..., 1])
            self._legacy_requested_drive_tensor = targets
            self.agent.robot.set_joint_drive_targets(clipped, self.agent.robot.active_joints)
            self.scene.px.gpu_apply_articulation_target_position()
            self.scene.px.gpu_apply_articulation_target_velocity()
            self.legacy_applied_drive_targets = clipped[0].detach().cpu().numpy()
            return
        # PhysX 4.1 clamps limited-joint drive positions while constructing
        # its constraints, without changing the public requested target.
        # Reproduce that solver input using real native PD drives; never
        # change qpos/qvel or spoof a readback. Restore the requested targets
        # after stepping so controller memory/checkpoints retain their meaning.
        # Source: NVIDIAGameWorks/PhysX 4.1, DyFeatherstoneArticulation.cpp,
        # angular lines2353-2357 and linear lines2493-2497.
        self._legacy_requested_drives = []
        joints = self.agent.robot._objs[0].active_joints
        applied = []
        for joint in joints:
            target = float(np.asarray(joint.drive_target).reshape(-1)[0])
            lower, upper = np.asarray(joint.limits).reshape(-1, 2)[0]
            clipped = float(np.clip(target, lower, upper))
            if clipped != target:
                self._legacy_requested_drives.append((joint, target))
                joint.set_drive_target(clipped)
            applied.append(float(np.asarray(joint.drive_target).reshape(-1)[0]))
        self.legacy_applied_drive_targets = np.asarray(applied)

    def _after_simulation_step(self):
        try:
            super()._after_simulation_step()
        finally:
            self._restore_requested_drives()


    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=500, control_freq=20,
                         scene_config=SceneConfig(enable_pcm=self.device.type == 'cuda', enable_tgs=False, solver_position_iterations=25,
                                                  solver_velocity_iterations=1),
                         default_materials_config=DefaultMaterialsConfig(static_friction=1., dynamic_friction=1.))


    def _configure_mpm_model(self, model):
        for prefix in ('static', 'body'):
            for key, value in dict(ke=100., kd=0., mu=1., ka=0.).items():
                setattr(model.struct, f'{prefix}_{key}', value)
        model.struct.ground_normal = wp.vec3(0., 0., 1.)
        model.struct.body_sticky = model.struct.ground_sticky = 1
        model.struct.particle_radius = .0025
        model.adaptive_grid = model.grid_contact = model.particle_contact = True


    def _load_ground(self):
        builder = self.scene.create_actor_builder()
        # SAPIEN planes have local +X as their outward normal. Rotate it to +Z;
        # the opposite sign makes the entire workspace lie inside the ground.
        builder.add_plane_collision(sapien.Pose(q=[np.sqrt(.5), 0., -np.sqrt(.5), 0.]))
        builder.initial_pose = sapien.Pose()
        self.ground = builder.build_static('ground')

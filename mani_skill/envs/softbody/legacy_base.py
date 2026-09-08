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


    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=500, control_freq=20,
                         scene_config=SceneConfig(enable_pcm=False, enable_tgs=False, solver_position_iterations=25,
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

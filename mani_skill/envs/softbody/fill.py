"""ManiSkill 3 port of the legacy Fill-v0 task (experimental).

Task equations/initialization adapted from ManiSkill 2 v0.5.3 fill_env.py,
493be36121a9dd06071a57172274babe617b789f. Restricted legacy assets are supplied
separately, retain their original scale, and are never downloaded implicitly.
"""

from pathlib import Path
import json
import os

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SimConfig, SceneConfig, DefaultMaterialsConfig
from mani_skill.sensors.camera import CameraConfig
from transforms3d.euler import euler2quat
from .base_env import MPMBaseEnv
from .geometry import register_visual_body, visual_meshes
from .mpm import MPMModelBuilder, wp


class LegacyPandaBucket(BaseAgent):
    uid = 'legacy_panda_bucket'
    urdf_config = {}

    def __init__(self, *args, legacy_asset_dir, **kwargs):
        self.urdf_path = str(Path(legacy_asset_dir) / 'descriptions/panda_bucket.urdf')
        if not Path(self.urdf_path).is_file():
            raise FileNotFoundError(self.urdf_path)
        super().__init__(*args, **kwargs)

    def _after_loading_articulation(self):
        # Preserve actual reference model inputs across URDF loader versions.
        # Mesh-cooked inertia, missing-inertia defaults, and joint-frame
        # quaternion conversions otherwise differ between SAPIEN 2 and 3.
        path = Path(__file__).with_name('legacy_fill_physics.json')
        parameters = json.loads(path.read_text())['robot_parameters']
        for link in parameters['links']:
            body = self.robot.links_map[link['name']]._objs[0]
            body.mass = link['mass']
            body.inertia = link['inertia']
            com = link['com']
            body.cmass_local_pose = sapien.Pose(com[:3], com[3:])
        for record in parameters['joints']:
            joint = self.robot.joints_map[record['name']]._objs[0]
            parent, child = record['parent_pose'], record['child_pose']
            joint.pose_in_parent = sapien.Pose(parent[:3], parent[3:])
            joint.pose_in_child = sapien.Pose(child[:3], child[3:])

    @property
    def _controller_configs(self):
        names = [f'panda_joint{i}' for i in range(1, 8)]
        def config(delta, target=False):
            return dict(arm=PDJointPosControllerConfig(names, -.1 if delta else None,
                        .1 if delta else None, 1000., 100., force_limit=100.,
                        friction=0., use_delta=delta, use_target=target,
                        normalize_action=delta), balance_passive_force=False)
        return {'pd_joint_delta_pos': config(True), 'pd_joint_pos': config(False),
                'pd_joint_target_delta_pos': config(True, True)}

    def before_simulation_step(self):
        # Legacy CPU controller cancels gravity AND Coriolis/centrifugal loads.
        # Merely disabling gravity (the normal MS3 default) is not equivalent.
        robot = self.robot._objs[0]
        passive = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        super().before_simulation_step()
        robot.set_qf(passive)


@register_env('Fill-v0', max_episode_steps=250)
class FillEnv(MPMBaseEnv):
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

    def _load_agent(self, options):
        self.agent = LegacyPandaBucket(self.scene, self._control_freq, self._control_mode,
                                      legacy_asset_dir=self.legacy_asset_dir,
                                      initial_pose=sapien.Pose([-.6, 0., 0.]))
        self.bucket = self.agent.robot.links_map['bucket']._objs[0]

    @property
    def _default_sensor_configs(self):
        return [CameraConfig('base_camera', sapien.Pose([-.4, 0., .4], euler2quat(0., np.pi/6, 0.)),
                             128, 128, np.pi/2, near=.001, far=10.)]

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig('render_camera', sapien.Pose([-.5, -.4, .6], euler2quat(0., np.pi/6, np.pi/2-np.pi/5)),
                            512, 512, 1., near=.001, far=10.)

    def _load_scene(self, options):
        builder = self.scene.create_actor_builder()
        # SAPIEN planes have local +X as their outward normal. Rotate it to +Z;
        # the opposite sign makes the entire workspace lie inside the ground.
        builder.add_plane_collision(sapien.Pose(q=[np.sqrt(.5), 0., -np.sqrt(.5), 0.]))
        builder.initial_pose = sapien.Pose()
        self.ground = builder.build_static('ground')
        path = self.legacy_asset_dir / 'deformable_manipulation/beaker.glb'
        if not path.is_file():
            raise FileNotFoundError(path)
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(str(path), scale=[.04] * 3)
        # A kinematic triangle mesh retains the real open cavity for PhysX too.
        # MS2 used its visual SDF for MPM but a convex hull for rigid collisions.
        builder.add_nonconvex_collision_from_file(str(path), scale=[.04] * 3)
        builder.initial_pose = sapien.Pose()
        self.target_beaker = builder.build_kinematic('target_beaker')
        self.beaker_body = self.target_beaker._bodies[0]
        # Kinematic inertial placeholders match the reference; they are not
        # physical measurements and have no effect on this fixed container.
        self.beaker_body.mass = 1.
        self.beaker_body.inertia = [1., 1., 1.]
        self.beaker_body.cmass_local_pose = sapien.Pose()
        vertices = np.concatenate([m.vertices for m in visual_meshes(self.beaker_body)])
        self._target_height = float(vertices[:, 2].max())
        self._target_radius = .04

    def _initialize_episode(self, env_idx, options):
        # Preserve legacy draw order: target XY, arm perturbation, then particles.
        rng = np.random.RandomState(int(self._episode_seed[0]))
        self.beaker_x = -.16 + (rng.rand() * 2 - 1) * .1
        self.beaker_y = (rng.rand() * 2 - 1) * .1
        self.target_beaker.set_pose(sapien.Pose([self.beaker_x, self.beaker_y, 0.]))
        qpos = np.array([-.188, .234, .201, -2.114, -.088, 1.35, 1.571])
        qpos[-2] += rng.normal(0., .03, 1)[0]
        self.agent.reset(torch.as_tensor(qpos, dtype=torch.float32, device=self.device)[None])
        self.agent.robot.set_pose(sapien.Pose([-.6, 0., 0.]))
        builder = MPMModelBuilder()
        builder.set_mpm_domain([.5, .5, .5], grid_length=.005)
        # This lifecycle finalizes after adding real particles. The legacy
        # reserve call inserts placeholder particles; using it here would add
        # them to the physical state instead of merely reserving storage.
        self.collision_geometry = [register_visual_body(builder, body, self.sdf_cache_dir)
                                   for body in (self.bucket, self.beaker_body)]
        E, nu = 1e4, .3
        builder.add_mpm_grid(pos=(-.2, -.01, .27), vel=(0., 0., 0.),
                             dim_x=int(.03 // .004), dim_y=int(.04 // .004), dim_z=int(.03 // .004),
                             cell_x=.004, cell_y=.004, cell_z=.004, density=3e3,
                             mu_lambda_ys=(E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu)), 1e4),
                             friction_cohesion=(.6, .05, 0.), type=1, jitter=True,
                             placement_x='center', placement_y='center', placement_z='start',
                             color=(1., 1., .5), random_state=rng)
        self.rebuild_mpm(builder, [self.bucket, self.beaker_body])

    def _configure_mpm_model(self, model):
        for prefix in ('static', 'body'):
            for key, value in dict(ke=100., kd=0., mu=1., ka=0.).items():
                setattr(model.struct, f'{prefix}_{key}', value)
        model.struct.ground_normal = wp.vec3(0., 0., 1.)
        model.struct.body_sticky = model.struct.ground_sticky = 1
        model.struct.particle_radius = .0025
        model.adaptive_grid = model.grid_contact = model.particle_contact = True

    def _task_counts(self):
        state = self.mpm_coupler.particle_state()
        x = state['x']
        center = self.beaker_body.entity_pose.p[:2]
        in_column = np.sum((x[:, :2] - center)**2, axis=1) < self._target_radius**2
        in_bounds = in_column & (x[:, 2] < self._target_height)
        inside = int(np.count_nonzero(in_bounds & (x[:, 2] > 0)))
        spill = int(np.count_nonzero(~in_bounds & (x[:, 2] < .005)))
        quiet = np.count_nonzero((state['v'] < .05) & (state['v'] > -.05)) / (len(x) * 3)
        return inside, spill, bool(inside / len(x) > .9 and quiet > .99)

    def evaluate(self):
        inside, spill, success = self._task_counts()
        return dict(success=torch.tensor([success], device=self.device),
                    contained_particles=torch.tensor([inside], device=self.device),
                    spilled_particles=torch.tensor([spill], device=self.device))

    def _get_obs_extra(self, info):
        pose = self.bucket.entity_pose
        return {**super()._get_obs_extra(info),
                'tcp_pose': torch.as_tensor(np.r_[pose.p, pose.q], device=self.device)[None],
                'target': torch.tensor([[self.beaker_x, self.beaker_y]], dtype=torch.float32, device=self.device)}

    def compute_dense_reward(self, obs, action, info):
        inside, spill, success = self._task_counts()
        if success:
            value = 2.5
        else:
            matrix = self.bucket.entity_pose.to_transformation_matrix()
            bucket_pos = (matrix @ np.array([0., .02, .08, 1.]))[:3]
            reach = 1 - np.tanh(10 * np.linalg.norm(bucket_pos[:2] - [self.beaker_x, self.beaker_y]))
            tilt = .4
            if reach > .9:
                base = (matrix @ np.array([0., -.01, .045, 1.]))[:3]
                bottom = (matrix @ np.array([0., .02, .08, 1.]))[:3]
                tilt = 1 - np.tanh(100 * (bottom[2] - base[2]))
            value = reach * .1 + inside / self.mpm_coupler.model.struct.n_particles - spill / 100 + tilt * .5
        return torch.tensor([value], dtype=torch.float32, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 2.5

    def get_state_dict(self):
        return {**super().get_state_dict(), 'task': {'beaker_xy': torch.tensor([[self.beaker_x, self.beaker_y]], dtype=torch.float64, device=self.device)}}

    def set_state_dict(self, state, env_idx=None):
        if not self._mpm_reset_active:
            raise RuntimeError('Task state assignment requires reset')
        xy = torch.as_tensor(state['task']['beaker_xy']).cpu().numpy()
        if xy.shape != (1, 2) or not np.isfinite(xy).all():
            raise ValueError('Invalid beaker target state')
        super().set_state_dict({k:v for k,v in state.items() if k != 'task'}, env_idx)
        self.beaker_x, self.beaker_y = xy[0]

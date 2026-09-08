"""ManiSkill 3 port of the legacy Fill-v0 task (experimental).

Task equations/initialization adapted from ManiSkill 2 v0.5.3 fill_env.py,
493be36121a9dd06071a57172274babe617b789f. Restricted legacy assets are supplied
separately, retain their original scale, and are never downloaded implicitly.
"""

import numpy as np
import sapien
import torch

from mani_skill.utils.registration import register_env
from mani_skill.sensors.camera import CameraConfig
from transforms3d.euler import euler2quat
from .geometry import register_visual_body, visual_meshes
from .mpm import MPMModelBuilder
from .bucket import LegacyBucketEnv


@register_env('Fill-v0', max_episode_steps=250)
class FillEnv(LegacyBucketEnv):
    @property
    def _default_sensor_configs(self):
        return [CameraConfig('base_camera', sapien.Pose([-.4, 0., .4], euler2quat(0., np.pi/6, 0.)),
                             128, 128, np.pi/2, near=.001, far=10.)]

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig('render_camera', sapien.Pose([-.5, -.4, .6], euler2quat(0., np.pi/6, np.pi/2-np.pi/5)),
                            512, 512, 1., near=.001, far=10.)

    def _load_scene(self, options):
        self._load_ground()
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

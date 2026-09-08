"""Native Excavate-v0, adapted from ManiSkill 2 v0.5.3 excavate_env.py.

The original metric geometry, granular recipe, random draw order, and task
equations are retained. See NOTICE.md for source and license information.
"""
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
from .bucket import LegacyBucketEnv
from .geometry import register_visual_body, convex_collision_meshes
from .mpm import MPMModelBuilder, register_collision_body
from . import perlin


@register_env('Excavate-v0', max_episode_steps=250)
class ExcavateEnv(LegacyBucketEnv):
    @property
    def _default_sensor_configs(self):
        return [CameraConfig('base_camera', sapien.Pose([-.2, 0., .4], euler2quat(0., np.pi/6, 0.)),
                             128, 128, np.pi/2, near=.001, far=10.)]

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig('render_camera', sapien.Pose([-.35, 0., .4], euler2quat(0., np.pi/6, 0.)),
                            512, 512, 1., near=.001, far=10.)

    def _load_scene(self, options):
        self._load_ground()
        poses = [sapien.Pose([0., -.1, .03]), sapien.Pose([0., .1, .03]),
                 sapien.Pose([-.1, 0., .03], [.7071068, 0., 0., .7071068]),
                 sapien.Pose([.1, 0., .03], [.7071068, 0., 0., .7071068])]
        self.walls = []
        for i, pose in enumerate(poses):
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[.12, .02, .03])
            builder.add_box_visual(half_size=[.12, .02, .03])
            builder.initial_pose = pose
            wall = builder.build_kinematic(f'wall_{i}')
            # Kinematic mass properties are inertial placeholders in both
            # engines, not measured wall properties or grasp tuning.
            body = wall._bodies[0]
            body.mass = 1.
            body.inertia = [1., 1., 1.]
            body.cmass_local_pose = sapien.Pose()
            self.walls.append(wall)
        # Legacy reward bounds use the first rigid collision hull, while MPM
        # contacts use the open visual SDF. Keep those separate definitions.
        vertices = convex_collision_meshes(self.bucket)[0].vertices
        self.vertices_mat = np.column_stack((vertices, np.ones(len(vertices))))

    @property
    def n_particles(self):
        return self.mpm_coupler.model.struct.n_particles

    def _initialize_episode(self, env_idx, options):
        rng = np.random.RandomState(int(self._episode_seed[0]))
        self.target_height = .2
        self.target_num = int(rng.choice(range(250, 1150), 1)[0])
        qpos = np.array([-.174, .457, .203, -1.864, -.093, 2.025, 1.588])
        qpos[:-1] += rng.normal(0., .02, len(qpos)-1)
        qpos[-1] += rng.normal(0., .2, 1)[0]
        self.agent.reset(torch.as_tensor(qpos, dtype=torch.float32, device=self.device)[None])
        self.agent.robot.set_pose(sapien.Pose([-.56, 0., 0.]))
        builder = MPMModelBuilder()
        builder.set_mpm_domain([.5, .5, .5], grid_length=.005)
        self.collision_geometry = [register_visual_body(builder, self.bucket, self.sdf_cache_dir)]
        bodies = [self.bucket, *[wall._bodies[0] for wall in self.walls]]
        for body in bodies[1:]:
            register_collision_body(builder, body)
        height_map = .06 + perlin.added_perlin([.03, .02, .02], [1, 2, 4],
                                              phases=[(0, 0)]*3, shape=(30, 30), random_state=rng)
        E, nu = 1e4, .3
        builder.add_mpm_from_height_map(pos=(0., 0., 0.), vel=(0., 0., 0.), dx=.005,
            height_map=height_map, density=3e3,
            mu_lambda_ys=(E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu)), 1e4),
            friction_cohesion=(.6, .05, 0.), type=1, jitter=True, color=(1., 1., .5), random_state=rng)
        self.rebuild_mpm(builder, bodies)
        if 'target_num' in options:
            target = options['target_num']
            if not np.isfinite(target) or int(target) != target or target <= 0:
                raise ValueError('target_num must be a positive integer')
            self.target_num = int(target)

    def _task_counts(self):
        state = self.mpm_coupler.particle_state()
        x, v = state['x'], state['v']
        lift = int(np.count_nonzero(x[:, 2] > self.target_height))
        spill = len(x) - int(np.count_nonzero((x[:, 0] > -.12) & (x[:, 0] < .12)
                                             & (x[:, 1] > -.12) & (x[:, 1] < .12)))
        quiet = np.count_nonzero((v < .05) & (v > -.05)) / (len(x) * 3)
        success = self.target_num-100 < lift < self.target_num+150 and spill < 20 and quiet > .99
        return lift, spill, bool(success)

    def evaluate(self):
        lift, spill, success = self._task_counts()
        return dict(success=torch.tensor([success], device=self.device),
                    lifted_particles=torch.tensor([lift], device=self.device),
                    spilled_particles=torch.tensor([spill], device=self.device))

    def _get_obs_extra(self, info):
        pose = self.bucket.entity_pose
        return {**super()._get_obs_extra(info),
                'tcp_pose': torch.as_tensor(np.r_[pose.p, pose.q], device=self.device)[None],
                'target': torch.tensor([[self.target_num]], dtype=torch.float32, device=self.device)}

    def compute_dense_reward(self, obs, action, info):
        return torch.tensor([self._dense_reward_value()], dtype=torch.float32, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 6.

    def get_state_dict(self):
        return {**super().get_state_dict(), 'task': {
            'target_num': torch.tensor([[self.target_num]], dtype=torch.float64, device=self.device)}}

    def set_state_dict(self, state, env_idx=None):
        if not self._mpm_reset_active:
            raise RuntimeError('Task state assignment requires reset')
        target = torch.as_tensor(state['task']['target_num']).cpu().numpy()
        if target.shape != (1, 1) or not np.isfinite(target).all() or target[0, 0] <= 0 or target[0, 0] != int(target[0, 0]):
            raise ValueError('Invalid target particle count')
        super().set_state_dict({k:v for k,v in state.items() if k != 'task'}, env_idx)
        self.target_num = int(target[0, 0])

    def _in_bbox_ids(self, particles_x, bbox):
        return np.where(
            (particles_x[:, 0] >= np.min(bbox[:, 0]))
            & (particles_x[:, 0] <= np.max(bbox[:, 0]))
            & (particles_x[:, 1] >= np.min(bbox[:, 1]))
            & (particles_x[:, 1] <= np.max(bbox[:, 1]))
            & (particles_x[:, 2] >= np.min(bbox[:, 2]))
            & (particles_x[:, 2] <= np.max(bbox[:, 2]))
        )[0]


    def _in_bucket_ids(self, particles_x, bbox, top_signs, bot_signs):
        return np.where(
            (particles_x[:, 0] >= np.min(bbox[:, 0]))
            & (particles_x[:, 0] <= np.max(bbox[:, 0]))
            & (particles_x[:, 1] >= np.min(bbox[:, 1]))
            & (particles_x[:, 1] <= np.max(bbox[:, 1]))
            & (particles_x[:, 2] >= np.min(bbox[:, 2]))
            & (particles_x[:, 2] <= np.max(bbox[:, 2]))
            & (top_signs > 0)
            & (bot_signs > 0)
        )[0]


    def _bucket_keypoints(self):
        gripper_mat = self.bucket.entity_pose.to_transformation_matrix()
        bucket_base_mat = np.array(
            [[1, 0, 0, 0], [0, 1, 0, -0.01], [0, 0, 1, 0.045], [0, 0, 0, 1]]
        )
        bucket_tlmat = np.array(
            [[1, 0, 0, -0.03], [0, 1, 0, -0.01], [0, 0, 1, 0.01], [0, 0, 0, 1]]
        )
        bucket_trmat = np.array(
            [[1, 0, 0, 0.03], [0, 1, 0, -0.01], [0, 0, 1, 0.01], [0, 0, 0, 1]]
        )
        bucket_blmat = np.array(
            [[1, 0, 0, -0.03], [0, 1, 0, 0.02], [0, 0, 1, 0.08], [0, 0, 0, 1]]
        )
        bucket_brmat = np.array(
            [[1, 0, 0, 0.03], [0, 1, 0, 0.02], [0, 0, 1, 0.08], [0, 0, 0, 1]]
        )
        bucket_base_pos = np.asarray((gripper_mat @ bucket_base_mat)[:3, 3], dtype=np.float32)
        bucket_tlpos = np.asarray((gripper_mat @ bucket_tlmat)[:3, 3], dtype=np.float32)
        bucket_trpos = np.asarray((gripper_mat @ bucket_trmat)[:3, 3], dtype=np.float32)
        bucket_blpos = np.asarray((gripper_mat @ bucket_blmat)[:3, 3], dtype=np.float32)
        bucket_brpos = np.asarray((gripper_mat @ bucket_brmat)[:3, 3], dtype=np.float32)
        return (
            bucket_base_pos,
            bucket_tlpos,
            bucket_trpos,
            bucket_blpos,
            bucket_brpos,
            gripper_mat,
        )


    def _get_bbox(self, points):
        return np.array(
            [
                [np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])],
                [np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2])],
            ]
        )


    def bucket_top_normal(self):
        (
            bucket_base_pos,
            bucket_tlpos,
            bucket_trpos,
            bucket_blpos,
            bucket_brpos,
            gripper_mat,
        ) = self._bucket_keypoints()
        bucket_top_normal = np.cross(
            bucket_base_pos - bucket_trpos, bucket_trpos - bucket_tlpos
        )
        bucket_top_normal /= np.linalg.norm(bucket_top_normal)
        return bucket_top_normal


    def particles_inside_bucket(self):
        # bucket boundary
        (
            bucket_base_pos,
            bucket_tlpos,
            bucket_trpos,
            bucket_blpos,
            bucket_brpos,
            gripper_mat,
        ) = self._bucket_keypoints()

        bucket_top_normal = np.cross(
            bucket_base_pos - bucket_trpos, bucket_trpos - bucket_tlpos
        )
        bucket_top_normal /= np.linalg.norm(bucket_top_normal)
        top_d = -np.sum(bucket_top_normal * bucket_base_pos)
        top_vec = np.array(list(bucket_top_normal) + [top_d])
        bucket_bot_normal = np.cross(
            bucket_base_pos - bucket_blpos, bucket_blpos - bucket_brpos
        )
        bucket_bot_normal /= np.linalg.norm(bucket_bot_normal)
        bot_d = -np.sum(bucket_bot_normal * bucket_base_pos)
        bot_vec = np.array(list(bucket_bot_normal) + [bot_d])

        vertices = (gripper_mat @ self.vertices_mat.T).T[:, :3]
        bbox = self._get_bbox(vertices)

        # pick particles
        particles_x = self.mpm_coupler.particle_state()["x"]
        ones = np.ones(len(particles_x))
        particles = np.column_stack((particles_x, ones.T))
        top_signs = particles @ top_vec.T
        bot_signs = particles @ bot_vec.T
        lifted_particles = particles_x[
            self._in_bucket_ids(particles_x, bbox, top_signs, bot_signs)
        ]
        return lifted_particles


    def _dense_reward_value(self, reward_info=False):
        if self._task_counts()[2]:
            if reward_info:
                return {"reward": 6.0}
            return 6.0
        particles_x = self.mpm_coupler.particle_state()["x"]

        stage = 0

        # spill reward
        spill_num = self.n_particles - len(
            np.where(
                (particles_x[:, 0] > -0.12)
                & (particles_x[:, 0] < 0.12)
                & (particles_x[:, 1] > -0.12)
                & (particles_x[:, 1] < 0.12)
            )[0]
        )
        spill_reward = -spill_num / 100

        (
            bucket_base_pos,
            bucket_tlpos,
            bucket_trpos,
            bucket_blpos,
            bucket_brpos,
            gripper_mat,
        ) = self._bucket_keypoints()

        lifted_particles = self.particles_inside_bucket()
        lift_num = len(lifted_particles)
        lift_reward = (
            min(lift_num / self.target_num, 1)
            - max(0, lift_num - self.target_num - 500) * 0.001
        )

        gripper_pos = self.bucket.entity_pose.p
        height_dist = (
            max(self.target_height + 0.05 - np.mean(lifted_particles[:, 2]), 0)
            if len(lifted_particles) > 0
            else 1
        )
        # reaching reward & height reward & flat reward
        if height_dist > 0.1 and lift_num > self.target_num + 300:
            reaching_reward = 1
            height_reward = 1 - np.tanh(3 * height_dist)
            flat_dist = 0.5 * (
                max(bucket_base_pos[2] + 0.01 - bucket_blpos[2], 0)
                + max(bucket_blpos[2] - bucket_brpos[2], 0)
            )
            flat_reward = 1 - np.tanh(50 * flat_dist)
            stage = 1
        elif height_dist <= 0.1:
            lift_reward = (
                1
                + min(lift_num / self.target_num, 1)
                - max(0, lift_num - self.target_num - 100) * 0.001
            )
            reaching_reward = 1
            height_reward = 1 - np.tanh(3 * height_dist)
            flat_dist = 0.5 * (
                max(bucket_base_pos[2] - 0.01 - bucket_blpos[2], 0)
                + max(bucket_blpos[2] - bucket_brpos[2], 0)
            )
            flat_reward = 1 - np.tanh(50 * flat_dist)
            stage = 2
        else:
            if (
                gripper_pos[0] > -0.1
                and gripper_pos[0] < 0.1
                and gripper_pos[1] > -0.1
                and gripper_pos[1] < 0.1
            ):
                dist = gripper_pos[2] + max(0.04 - gripper_pos[0], 0)
                reaching_reward = 1 - np.tanh(10 * dist)
            else:
                reaching_reward = 0
            height_reward = 0
            flat_reward = 0

        reward = (
            reaching_reward * 0.5
            + lift_reward
            + height_reward
            + spill_reward
            + flat_reward
        )
        if reward_info:
            return {
                "reward": reward,
                "reaching_reward": reaching_reward,
                "lift_reward": lift_reward,
                "lift_num": lift_num,
                "target_num": self.target_num,
                "flat_reward": flat_reward,
                "height_reward": height_reward,
                "spill_reward": spill_reward,
                "stage": stage,
                "height_dist": height_dist,
            }
        return reward

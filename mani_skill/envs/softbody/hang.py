"""Native Hang-v0 using the original recorded rope grasps and MPM inputs.

Adapted from ManiSkill 2 v0.5.3 hang_env.py. The separately supplied numeric pack
retains its original source terms; see NOTICE.md. No pickle is loaded here.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
from transforms3d.quaternions import axangle2quat

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from .legacy_base import LegacyMPMEnv
from .legacy_panda import LegacyPanda
from .geometry import load_numeric_pack_file, register_reference_collision_body
from .mpm import MPMModelBuilder


HANG_PACK_MANIFEST_SHA256 = 'bd9a93ad23798da1b34bbb6b64f4bd2470a4740cbfbbec519ee5b584720db431'


@register_env('Hang-v0', max_episode_steps=350)
class HangEnv(LegacyMPMEnv):
    _supports_mpm_batch = True
    _batch_particle_capacity = 8192

    def __init__(self, *args, legacy_mpm_data_dir=None, **kwargs):
        directory = legacy_mpm_data_dir or os.environ.get('MANISKILL_LEGACY_MPM_DATA')
        if not directory:
            raise ValueError('Provide legacy_mpm_data_dir with the pinned numeric Hang reference export')
        self.legacy_mpm_data_dir = Path(directory).resolve()
        manifest = self.legacy_mpm_data_dir / 'export.json'
        if manifest.stat().st_size > 2 * 1024**2 or hashlib.sha256(manifest.read_bytes()).hexdigest() != HANG_PACK_MANIFEST_SHA256:
            raise ValueError('Hang numeric pack manifest checksum mismatch')
        self.reference_pack = json.loads(manifest.read_text())
        self.rope_starts = load_numeric_pack_file(self.legacy_mpm_data_dir, 'initial-states.npz',
                                                  self.reference_pack['files']['initial-states.npz'])
        self.reference_geometry = [load_numeric_pack_file(self.legacy_mpm_data_dir, r['file'],
                                    self.reference_pack['files'][r['file']]) for r in self.reference_pack['geometry']]
        super().__init__(*args, **kwargs)

    def _load_agent(self, options):
        self.agent = LegacyPanda(self.scene, self._control_freq, self._control_mode,
            legacy_asset_dir=self.legacy_asset_dir, robot_parameters=self.reference_pack['robot_parameters'],
            initial_pose=sapien.Pose([-.46, 0., 0.]))
        self.hands = self.agent.robot.links_map['panda_hand']._objs
        self.leftfingers = self.agent.robot.links_map['panda_leftfinger']._objs
        self.rightfingers = self.agent.robot.links_map['panda_rightfinger']._objs
        self.hand, self.leftfinger, self.rightfinger = self.hands[0], self.leftfingers[0], self.rightfingers[0]

    def _load_scene(self, options):
        self._load_ground()
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[.3, .01, .01])
        builder.add_box_visual(half_size=[.3, .01, .01],
                              material=sapien.render.RenderMaterial(base_color=[1., 0., 0., 1.]))
        builder.initial_pose = sapien.Pose()
        self.rod = builder.build_kinematic('rod')
        self.rod_bodies = self.rod._bodies
        self.rod_body = self.rod_bodies[0]
        reference = next(r for r in self.reference_pack['geometry'] if r['name'] == 'rod')
        for body in self.rod_bodies:
            body.mass = reference['mass']
            body.inertia = reference['inertia']
            body.cmass_local_pose = sapien.Pose(reference['com'][:3], reference['com'][3:])
        self._selected_indices = np.zeros((self.num_envs, 5), dtype=np.int64)
        self.rope_start_indices = np.zeros(self.num_envs, dtype=np.int64)

    @property
    def _default_sensor_configs(self):
        return [CameraConfig('base_camera', sapien.Pose([.45, 0., .5], euler2quat(0., np.pi/5, np.pi)),
                             128, 128, np.pi/2, near=.001, far=10.)]

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig('render_camera', sapien.Pose([.2, 1., .5], euler2quat(0., .2, 4.4)),
                            512, 512, 1., near=.001, far=10.)

    def _initialize_episode(self, env_idx, options):
        if self._mpm_batch is not None:
            return self._initialize_batch_episode(env_idx)
        rng = np.random.RandomState(int(self._episode_seed[0]))
        radius = .2 + rng.rand() * .03
        angle = np.pi/4 + rng.rand() * np.pi/2
        height = .2 + rng.rand() * .1
        self.rod.set_pose(sapien.Pose([np.sin(angle)*radius, np.cos(angle)*radius, height],
                                    axangle2quat([0, 0, 1], -angle)))
        nominal = np.array([0., np.pi/16, 0., -np.pi*5/6, 0., np.pi-.2, np.pi/4, 0., 0.])
        self.agent.reset(torch.as_tensor(nominal, dtype=torch.float32, device=self.device)[None])
        # ManiSkill 2 agent.reset also resets controller memory. Capture that
        # nominal target before restoring the recorded grasp below.
        self.agent.controller.reset()
        self.agent.robot.set_pose(sapien.Pose([-.46, 0., 0.]))
        builder, bodies = self._hang_builder(rng, 0)
        self.rebuild_mpm(builder, bodies)
        x = self.mpm_coupler.particle_state()['x']
        self.selected_indices = self._select_rope_particles(x)
        index = int(rng.randint(len(self.rope_starts['mpm_x'])))
        self.rope_start_index = index
        checkpoint = self.get_state_dict()
        checkpoint['mpm'] = {key: torch.as_tensor(self.rope_starts['mpm_'+key][index], device=self.device)[None]
                              for key in checkpoint['mpm']}
        robot = self._recorded_robot_state(index)
        checkpoint['articulations'][self.agent.robot.name] = torch.as_tensor(robot, device=self.device)[None]
        # The source recording also stores qacc. Its SAPIEN 3 setter does not
        # change native acceleration readback; retain it only in the source pack.
        # Apply the recorded grasp after the normal controller reset, preserving
        # the nominal target memory captured above, exactly as in the N1 path.
        self.defer_initial_state(checkpoint)

    def _hang_builder(self, rng, index):
        builder = MPMModelBuilder()
        builder.set_mpm_domain([.5, .5, .5], grid_length=.015)
        bodies = []
        for record, arrays in zip(self.reference_pack['geometry'], self.reference_geometry):
            body = self.rod_bodies[index] if record['name'] == 'rod' else self.agent.robot.links_map[record['name']]._objs[index]
            register_reference_collision_body(builder, body, record, arrays)
            bodies.append(body)
        E, nu, cell = 1e4, .3, .004
        builder.add_mpm_grid(pos=(.1, 0., .05), vel=(0., 0., 0.),
            dim_x=int(.4//cell), dim_y=int(.022//cell), dim_z=int(.022//cell),
            cell_x=cell, cell_y=cell, cell_z=cell, density=300.,
            mu_lambda_ys=(E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu)), 1e4),
            friction_cohesion=(.6, .05, 0.), type=0, jitter=True,
            placement_x='center', placement_y='center', placement_z='start',
            color=(1., 1., .5), random_state=rng)
        return builder, bodies

    @staticmethod
    def _select_rope_particles(x):
        lower, upper = x.min(0), x.max(0)
        dim = np.argmax(upper-lower)
        length = upper[dim]-lower[dim]
        return np.asarray([np.flatnonzero((lower[dim]+length*lo < x[:, dim])
            & (x[:, dim] < lower[dim]+length*hi))[0] for lo, hi in
            ((.05, .15), (.25, .35), (.48, .52), (.65, .75), (.85, .95))], dtype=np.int64)

    def _recorded_robot_state(self, index):
        return np.r_[self.rope_starts['robot_root_pose'][index], self.rope_starts['robot_root_vel'][index],
                       self.rope_starts['robot_root_qvel'][index], self.rope_starts['robot_qpos'][index],
                       self.rope_starts['robot_qvel'][index]]

    def _initialize_batch_episode(self, env_idx):
        indices = self._mpm_batch.indices(env_idx)
        random_states, poses = [], []
        for index in indices:
            rng = np.random.RandomState(int(self._episode_seed[index]))
            radius = .2 + rng.rand() * .03
            angle = np.pi/4 + rng.rand() * np.pi/2
            height = .2 + rng.rand() * .1
            poses.append(sapien.Pose([np.sin(angle)*radius, np.cos(angle)*radius, height],
                                    axangle2quat([0, 0, 1], -angle)))
            random_states.append(rng)
        self.rod.set_pose(Pose.create(poses, device=self.device))
        nominal = np.array([0., np.pi/16, 0., -np.pi*5/6, 0., np.pi-.2, np.pi/4, 0., 0.])
        self.agent.reset(torch.as_tensor(np.repeat(nominal[None],len(indices),axis=0),dtype=torch.float32,device=self.device))
        self.agent.controller.reset()
        self.agent.robot.set_pose(sapien.Pose([-.46, 0., 0.]))
        for index, rng in zip(indices, random_states):
            builder, bodies = self._hang_builder(rng,index)
            self.rebuild_mpm(builder,bodies,env_idx=index)
        for index, rng in zip(indices, random_states):
            self._selected_indices[index] = self._select_rope_particles(self.mpm_couplers[index].particle_state()['x'])
            self.rope_start_indices[index] = int(rng.randint(len(self.rope_starts['mpm_x'])))
        def select(value):
            return {k:select(v) for k,v in value.items()} if isinstance(value,dict) else value[indices].clone()
        checkpoint = select(self.get_state_dict())
        for row, index in enumerate(indices):
            start = self.rope_start_indices[index]
            n = self.mpm_couplers[index].model.struct.n_particles
            for key, value in checkpoint['mpm'].items():
                recorded = torch.as_tensor(self.rope_starts['mpm_'+key][start],device=self.device)
                if recorded.shape != value[row,:n].shape:
                    raise ValueError('Pinned rope grasp particle layout differs from the initialized rope')
                value[row].zero_(); value[row,:n] = recorded
            checkpoint['articulations'][self.agent.robot.name][row] = torch.as_tensor(self._recorded_robot_state(start),device=self.device)
        self.defer_initial_state(checkpoint)

    def _evaluation_indices(self, index):
        return self._selected_indices[index] if self._mpm_batch is not None else self.selected_indices

    def _configure_mpm_model(self, model):
        super()._configure_mpm_model(model)
        model.struct.particle_radius = .005

    def evaluate(self):
        return {'success': torch.tensor([self._task_success(i) for i in range(self.num_envs)], device=self.device)}

    def _get_obs_extra(self, info):
        hands = [self.rigid_pose(body) for body in self.hands]
        rods = [self.rigid_pose(body) for body in self.rod_bodies]
        return {**super()._get_obs_extra(info),
            'tcp_pose': torch.as_tensor(np.array([np.r_[p.p,p.q] for p in hands]),device=self.device),
            'target': torch.as_tensor(np.array([np.r_[p.p,p.q] for p in rods]),device=self.device)}

    def compute_dense_reward(self, obs, action, info):
        return torch.tensor([self._dense_reward_value(i) for i in range(self.num_envs)], dtype=torch.float32, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 6.

    def get_state_dict(self):
        if self._mpm_batch is not None:
            return {**super().get_state_dict(),'task':{
                'selected_indices':torch.as_tensor(self._selected_indices.copy(),device=self.device)}}
        return {**super().get_state_dict(), 'task': {
            'selected_indices': torch.as_tensor(self.selected_indices, device=self.device)[None]}}

    def set_state_dict(self, state, env_idx=None):
        if not self._mpm_reset_active:
            raise RuntimeError('Task state assignment requires reset')
        indices = torch.as_tensor(state['task']['selected_indices']).cpu().numpy()
        selected = self._mpm_batch.indices(env_idx) if self._mpm_batch is not None else [0]
        counts = (torch.as_tensor(state['mpm_meta']['count']).cpu().numpy() if self._mpm_batch is not None
                  else np.array([[torch.as_tensor(state['mpm']['x']).shape[1]]]))
        if (indices.shape != (len(selected), 5) or counts.shape != (len(selected),1)
                or not np.isfinite(indices).all() or np.any(indices != np.floor(indices))
                or np.any(indices < 0) or np.any(indices >= counts)):
            raise ValueError('Invalid rope evaluation particle indices')
        super().set_state_dict({k:v for k,v in state.items() if k != 'task'}, env_idx)
        if self._mpm_batch is not None:
            self._selected_indices[selected] = indices.astype(np.int64)
        else:
            self.selected_indices = indices[0].astype(np.int64)


    def _task_success(self, index=0, **kwargs):
        particles_x = self.mpm_couplers[index].particle_state()['x']
        particles_v = self.mpm_couplers[index].particle_state()['v']
        lf_pos = self.rigid_pose(self.leftfingers[index]).p
        rf_pos = self.rigid_pose(self.rightfingers[index]).p
        finger_dist = np.linalg.norm(lf_pos - rf_pos)
        pose = self.rigid_pose(self.rod_bodies[index])
        center = pose.p
        normal = pose.to_transformation_matrix()[:3, :3] @ np.array([0, 1, 0])
        x = particles_x[self._evaluation_indices(index)]
        dirs = x - center
        signs = np.sign(dirs @ normal)
        side = signs[0] == signs[1] and signs[3] == signs[4] and (signs[0] != signs[3])
        top = np.max(particles_x[:, 2]) > center[2] and np.max(particles_x[:, 2]) < center[2] + 0.05
        down = dirs[0, 2] < 0 and dirs[4, 2] < 0
        bottom = np.min(particles_x[:, 2]) > 0.03
        high_ind = np.where(particles_x[:, 2] > center[2] - 0.03)
        particles_v = particles_v[high_ind]
        return bool(side and top and down and bottom and (len(np.where((particles_v < 0.05) & (particles_v > -0.05))[0]) / (len(particles_v) * 3 + 0.001) > 0.99) and (finger_dist > 0.07))

    def _dense_reward_value(self, index=0):
        gripper_width = self.agent.robot._objs[index].get_qlimits()[-1, 1] * 2
        if self._task_success(index):
            reward = 6
            reaching_reward = 1
            center_reward = 1
            side_reward = 1
            top_reward = 1
            release_reward = 1
        else:
            gripper_pos = self.rigid_pose(self.hands[index]).p
            particles_x = self.mpm_couplers[index].particle_state()['x']
            distance = np.min(np.linalg.norm(particles_x - gripper_pos, axis=-1))
            reaching_reward = 1 - np.tanh(10.0 * distance)
            center_reward = 0.0
            pose = self.rigid_pose(self.rod_bodies[index])
            rod_center = pose.p
            rope_center = particles_x[self._evaluation_indices(index)[2]]
            distance = np.linalg.norm(rod_center[:2] - rope_center[:2])
            center_reward += 0.5 * (1 - np.tanh(10.0 * distance))
            if rod_center[2] >= rope_center[2]:
                distance = rod_center[2] - rope_center[2]
                center_reward += 0.5 * (1 - np.tanh(10.0 * distance))
            else:
                center_reward += 0.5
            top_reward = 0
            bottom_reward = 0
            release_reward = 0
            if rod_center[2] >= rope_center[2]:
                side_reward = 0
            else:
                bottom_reward = 1 - np.tanh(10.0 * max(0, 0.04 - np.min(particles_x[:, 2])))
                rod_normal = pose.to_transformation_matrix()[:3, :3] @ np.array([0, 1, 0])
                x = particles_x[self._evaluation_indices(index)]
                dirs = x - rod_center
                signs = np.sign(dirs @ rod_normal)
                side_reward = 0.5 * (int(signs[0] != signs[3]) + int(signs[0] == signs[1] and signs[3] == signs[4] and (signs[0] != signs[3])))
                if signs[0] == signs[1] and signs[3] == signs[4] and (signs[0] != signs[3]):
                    top_reward = 0.25 * (int(dirs[0, 2] < 0) + int(dirs[1, 2] < 0) + int(dirs[3, 2] < 0) + int(dirs[4, 2] < 0))
                    reaching_reward = 1
                    if top_reward > 0.9:
                        release_reward = np.sum(self.agent.robot.get_qpos()[index, -2:].detach().cpu().numpy()) / gripper_width
            reward = reaching_reward + center_reward + side_reward + top_reward + release_reward + bottom_reward * 0.2
        return reward

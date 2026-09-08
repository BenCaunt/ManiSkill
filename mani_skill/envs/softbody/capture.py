"""Numeric adapter for the external softbody_lab comparison harness.

No reference simulator code is imported. Fixture assignment occurs only through
native environment reset; snapshots always read the candidate's actual state.
"""
import numpy as np
import torch

from .fill import FillEnv
from .excavate import ExcavateEnv
from .hang import HangEnv
from .pour import PourEnv
from .base_env import MATERIAL_FIELDS


def numeric_tree(value, *, unbatch=False):
    if isinstance(value, dict):
        return {k: numeric_tree(v, unbatch=unbatch) for k, v in value.items()}
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
        if unbatch:
            value = value[0]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class CaptureAdapter:
    def __init__(self, env_id, *, control_mode, env_kwargs):
        tasks = {'Fill-v0': FillEnv, 'Excavate-v0': ExcavateEnv, 'Hang-v0': HangEnv, 'Pour-v0': PourEnv}
        if env_id not in tasks:
            raise NotImplementedError(f'Native task capture is not yet implemented for {env_id}')
        self.control_mode = control_mode
        self.env_id = env_id
        self.env = tasks[env_id](obs_mode='none', control_mode=control_mode, **env_kwargs)

    def _physical_actors(self):
        env = self.env
        expected = {'Fill-v0': ['ground', 'target_beaker'], 'Excavate-v0': ['ground', 'wall_0', 'wall_1', 'wall_2', 'wall_3'],
                    'Hang-v0': ['ground', 'rod'], 'Pour-v0': ['ground', 'bottle', 'target_beaker']}[self.env_id]
        if sorted(env.scene.actors) != sorted(expected):
            raise RuntimeError('Unexpected physical actor set')
        if env.agent.robot._objs[0].root.joint.type != 'fixed':
            raise RuntimeError('Portable tasks require a fixed-base robot')
        return [env.scene.actors[name]._bodies[0] for name in expected]

    def _recorded_bodies(self):
        if self.env_id == 'Hang-v0':
            return [self.env.agent.robot.links_map[r['name']]._objs[0]
                    for r in self.env.reference_pack['robot_parameters']['links']] + [self.env.rod_body]
        return self.env.mpm_coupler.bodies

    def reset(self, *, seed, reset_kwargs, replay=None):
        env = self.env
        # The recorder stores keyword arguments to gym.Env.reset. Match the
        # reference's options nesting; never silently ignore task reset options.
        if set(reset_kwargs) - {'options'}:
            raise ValueError('Unsupported reset keyword arguments')
        options = dict(reset_kwargs.get('options') or {})
        env.reset(seed=seed, options=options)
        if replay is not None:
            state, fixture, material = replay
            if fixture.get('initial_state_contract', {}).get('version') != 2:
                raise ValueError('Candidate replay requires a portable version 2 fixture')
            if material is None:
                raise ValueError('Particle material data is required for replay')
            current = self.snapshot()
            for key in ('scene_actor_pose', 'scene_actor_velocity'):
                if not np.array_equal(current[key][0], state[key][0]):
                    raise ValueError('Fixed ground state differs from the fixture')
            checkpoint = env.get_state_dict()
            checkpoint['mpm'] = {k: torch.as_tensor(state[k])[None] for k in checkpoint['mpm']}
            checkpoint['mpm_material'] = {k: torch.as_tensor(material[k])[None] for k in MATERIAL_FIELDS}
            robot = np.r_[state['root_pose'][0], state['root_velocity'][0], state['qpos'], state['qvel']]
            checkpoint['articulations'][env.agent.robot.name] = torch.as_tensor(robot)[None]
            for i, body in enumerate(self._physical_actors()[1:], 1):
                actor = np.r_[state['scene_actor_pose'][i], state['scene_actor_velocity'][i]]
                checkpoint['actors'][body.name] = torch.as_tensor(actor)[None]
            checkpoint['mpm_drives'] = {k: torch.as_tensor(state['drive_'+k])[None] for k in ('position', 'velocity')}
            def batched(value):
                return {k: batched(v) for k, v in value.items()} if isinstance(value, dict) else torch.as_tensor(value, dtype=torch.float32)[None]
            controller = batched(fixture['controller_state'])
            if controller:
                checkpoint['controller'] = controller
            task_key = {'Fill-v0': 'beaker_xy', 'Excavate-v0': 'target_num', 'Hang-v0': 'selected_indices', 'Pour-v0': 'fill_heights'}[self.env_id]
            checkpoint['task'][task_key] = torch.as_tensor(state['task_state'], dtype=torch.float64)[None]
            env.reset(seed=seed, options={**options, 'reset_to_env_states': {'env_states': checkpoint}})
        return self.snapshot()

    def snapshot(self):
        env = self.env
        model = env.mpm_coupler.model
        robot = env.agent.robot._objs[0]
        root = robot.root
        joints = robot.active_joints
        bodies = self._recorded_bodies()
        actors = self._physical_actors()
        return {**env.mpm_coupler.particle_state(),
                'mass': model.struct.particle_mass.numpy()[:model.struct.n_particles].copy(),
                'qpos': robot.qpos.copy(), 'qvel': robot.qvel.copy(),
                'sim_state': env.get_state().detach().cpu().numpy().reshape(-1),
                # Canonical telemetry uses float64 because SAPIEN 2 exposes
                # native float targets as Python scalars. Conversion is exact.
                'drive_position': np.array([j.drive_target for j in joints], dtype=np.float64).reshape(-1),
                'drive_velocity': np.array([j.drive_velocity_target for j in joints], dtype=np.float64).reshape(-1),
                'rigid_pose': np.asarray([np.r_[b.entity_pose.p, b.entity_pose.q] for b in bodies]),
                'rigid_velocity': np.asarray([np.r_[b.linear_velocity, b.angular_velocity] for b in bodies]),
                'root_pose': np.asarray([np.r_[root.entity_pose.p, root.entity_pose.q]]),
                'root_velocity': np.asarray([np.r_[root.linear_velocity, root.angular_velocity]]),
                'scene_actor_pose': np.asarray([np.r_[b.entity_pose.p, b.entity_pose.q] for b in actors]),
                'scene_actor_velocity': np.asarray([np.zeros(6, dtype=np.float32),
                                                    *[np.r_[a.linear_velocity, a.angular_velocity] for a in actors[1:]]]),
                'task_state': np.asarray([env.beaker_x, env.beaker_y] if self.env_id == 'Fill-v0'
                                         else [env.target_num] if self.env_id == 'Excavate-v0'
                                         else [env.h1, env.h2] if self.env_id == 'Pour-v0'
                                         else env.selected_indices, dtype=np.float64)}

    def description(self):
        env = self.env
        model = env.mpm_coupler.model
        material = {k: getattr(model.struct, k).numpy()[:model.struct.n_particles].copy() for k in MATERIAL_FIELDS}
        keys = ('dx', 'inv_dx', 'grid_dim_x', 'grid_dim_y', 'grid_dim_z', 'particle_radius',
                'body_ke', 'body_kd', 'body_mu', 'body_ka', 'body_sticky', 'ground_sticky',
                'static_ke', 'static_kd', 'static_mu', 'static_ka')
        return {'additional_state_fields': ['vol'] if self.env_id == 'Pour-v0' else [],
                'initial_state_contract': dict(version=2, derived_rigid_indices=list(range(13)) if self.env_id == 'Hang-v0' else [] if self.env_id == 'Pour-v0' else [0], root_kind='fixed',
                    scene_actor_names=[a.name for a in self._physical_actors()],
                    scene_actor_types=['static', 'dynamic', 'kinematic'] if self.env_id == 'Pour-v0' else ['static'] + ['kinematic']*(len(self._physical_actors())-1)),
                'controller_state': numeric_tree(env.agent.get_controller_state(), unbatch=True),
                'control_mode': self.control_mode, 'control_dt': float(env.control_timestep),
                'rigid_dt': float(env.scene.px.timestep), 'mpm_dt': float(env.mpm_dt),
                'mpm_substeps': env.mpm_coupler.substeps,
                'parameters': {k: numeric_tree(getattr(model.struct, k)) for k in keys}}, material

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        robot = self.env.agent.robot._objs[0]
        if np.max(abs(robot.qpos)) > 10 or np.max(abs(robot.qvel)) > 100:
            raise RuntimeError('Unstable robot state')
        return {**numeric_tree(info, unbatch=True), 'reward': float(reward[0]),
                'terminated': bool(terminated[0]), 'truncated': bool(truncated[0])}

    def metrics(self):
        return numeric_tree(self.env.evaluate(), unbatch=True)

    def close(self):
        self.env.close()

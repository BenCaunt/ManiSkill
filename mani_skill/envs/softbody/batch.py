"""Per-scene MPM state under one ManiSkill native GPU physics world."""
import copy

import numpy as np
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.batched_rng import BatchedRNG
from .gpu_coupling import MPMGPUWorld
from .particle_visuals import ParticleVisualPool


MATERIAL_FIELDS = ('particle_mass', 'particle_vol', 'particle_mu_lam_ys',
                   'particle_friction_cohesion', 'particle_type')
PARTICLE_FIELDS = dict(x='particle_q', v='particle_qd', F='particle_F',
                       C='particle_C', vc='particle_volume_correction', vol='particle_vol')


def clone_tree(value):
    return {k: clone_tree(v) for k, v in value.items()} if isinstance(value, dict) else value.clone()


class MPMBatchRuntime:
    def __init__(self, env, count, capacity):
        self.env, self.count, self.capacity = env, count, capacity
        self.builders = [None] * count
        self.couplers = [None] * count
        self.pools = [None] * count
        self.render_poses = None
        self.reset_indices = None

    def indices(self, indices=None):
        if indices is None:
            return list(range(self.count))
        value = torch.as_tensor(indices).detach().cpu().numpy()
        if (value.ndim != 1 or not len(value) or value.dtype.kind not in 'iu'
                or len(np.unique(value)) != len(value) or np.any((value < 0) | (value >= self.count))):
            raise ValueError('Reset indices must be unique in-range integer environment indices')
        return value.tolist()

    @property
    def partial_reset(self):
        return self.reset_indices is not None and len(self.reset_indices) < self.count

    @staticmethod
    def seed_values(seed, count):
        if seed is None:
            return None
        values = np.asarray(seed)
        if values.ndim == 0:
            values = values.reshape(1)
        if (values.ndim != 1 or len(values) not in (1, count) or values.dtype.kind not in 'iu'
                or np.any(values < 0) or np.any(values > 2**32-1)):
            raise ValueError('Provide one integer seed or one seed per selected environment, in [0, 2**32)')
        values = values.astype(np.int64, copy=True)
        if len(values) == 1 and count > 1:
            values = np.r_[values, np.random.RandomState(values[0]).randint(2**31, size=count-1)]
        return values

    def set_main_rng(self, seed):
        # BaseEnv's default explicit reseed replaces every stream. A partial
        # reset must preserve both the values and draw positions of neighbours.
        if seed is None:
            return
        env = self.env; indices = self.reset_indices
        values = self.seed_values(seed, len(indices))
        replacement = BatchedRNG.from_seeds(values, backend=env._batched_rng_backend)
        env._main_seed[indices] = values
        env._batched_main_rng[indices] = replacement.rngs
        if 0 in indices:
            env._main_rng = np.random.RandomState(env._main_seed[0])

    def set_episode_rng(self, seed, env_idx):
        env = self.env; indices = self.indices(env_idx)
        if indices != self.reset_indices:
            raise ValueError('Episode RNG selection differs from the active reset')
        if seed is None and not env._enhanced_determinism:
            return
        values = (env._batched_main_rng[indices].randint(2**31) if seed is None
                  else self.seed_values(seed, len(indices)))
        replacement = BatchedRNG.from_seeds(values, backend=env._batched_rng_backend)
        env._episode_seed[indices] = values
        env._batched_episode_rng[indices] = replacement.rngs
        env._episode_rng = env._batched_episode_rng[0]

    def reset(self, seed, options):
        env = self.env
        options = dict(options or {})
        selected = self.indices(options.get('env_idx'))
        reconfigure = options.get('reconfigure', False) or (
            env._reconfig_counter == 0 and env.reconfiguration_freq != 0)
        if reconfigure and len(selected) != self.count:
            raise RuntimeError('Cannot reconfigure only part of a shared world')
        world = env.mpm_gpu_world
        if world is not None and (world.pending_step or world._active):
            raise RuntimeError('Cannot reset during a pending shared physics step')
        if world is not None and world.failed and len(selected) != self.count:
            raise RuntimeError('A failed shared world requires a full reset')
        seeds = self.seed_values(seed, len(selected))
        if seeds is not None and len(selected) == self.count and selected != sorted(selected):
            # BaseEnv's full reset APIs take seeds in global environment order.
            seed = np.empty(self.count, dtype=np.int64)
            seed[selected] = seeds
        restore = options.pop('reset_to_env_states', None)
        self.reset_indices = selected
        env._mpm_initial_checkpoint = None
        env._mpm_reset_active = True
        try:
            obs, info = BaseEnv.reset(env, seed=seed, options=options)
            state = restore['env_states'] if restore is not None else env._mpm_initial_checkpoint
            if state is not None:
                if isinstance(state, dict):
                    env.set_state_dict(state, selected)
                else:
                    env.set_state(state, selected)
                info = {**env.get_info(), 'reconfigure': info['reconfigure']}
                obs = env.get_obs(info)
                env._last_obs = obs
            return obs, info
        except BaseException:
            if env.mpm_gpu_world is not None:
                env.mpm_gpu_world.failed = True
                for coupler in env.mpm_gpu_world.couplers:
                    coupler.failed = True
            raise
        finally:
            self.reset_indices = None
            env._mpm_initial_checkpoint = None
            env._mpm_reset_active = False

    def rebuild(self, builder, bodies, index):
        env = self.env
        if not env._mpm_reset_active or self.reset_indices is None or index not in self.reset_indices:
            raise RuntimeError('MPM model rebuilding must target an environment being reset')
        if not 0 < len(builder.mpm_particle_q) <= self.capacity:
            raise ValueError(f'Particle count must fit the declared batch capacity {self.capacity}')
        scene = env.scene.sub_scenes[index]
        dt = scene.get_timestep()
        substeps = round(dt / env.mpm_dt)
        if substeps < 1 or not np.isclose(substeps * env.mpm_dt, dt, rtol=1e-7, atol=1e-10):
            raise ValueError('PhysX timestep must be an integer multiple of mpm_dt')
        model = builder.finalize(env.mpm_device)
        model.gravity = np.asarray(env.sim_config.scene_config.gravity, dtype=np.float32)
        env._configure_mpm_model(model)
        states = [model.state() for _ in range(substeps + 1)]
        builder.init_model_state(model, states)
        self.builders[index] = builder
        # Keep each untouched model's actual current buffer order. Reconstructing
        # the scheduler must not restore its original (pre-rollout) buffer order.
        records = [(c.scene, c.model, c.states, c.bodies) if c is not None else None for c in self.couplers]
        records[index] = (scene, model, states, bodies)
        if not hasattr(self, '_pending_records'):
            self._pending_records = records
        else:
            for i, record in enumerate(records):
                if record is not None:
                    self._pending_records[i] = record
        if any(record is None for record in self._pending_records):
            return
        world = MPMGPUWorld(env.scene.px)
        couplers = [world.add_model(scene, model, states, bodies, mpm_dt=env.mpm_dt)
                    for scene, model, states, bodies in self._pending_records]
        if len(self.reset_indices) != self.count and env.mpm_gpu_world is not None:
            world.steps = env.mpm_gpu_world.steps
        self.couplers = couplers
        env.mpm_gpu_world = world
        env.last_coupling_step = None
        del self._pending_records
        self.setup_visuals()

    def setup_visuals(self):
        env = self.env
        if not env.scene.can_render():
            return
        for i in range(self.count):
            if self.pools[i] is None:
                self.pools[i] = ParticleVisualPool(env.scene.sub_scenes[i])
        changed = self.render_poses is None or any(pool.needs_change(c.model) for pool, c in zip(self.pools, self.couplers))
        if changed:
            env._invalidate_particle_render_groups()
        for pool, coupler in zip(self.pools, self.couplers):
            pool.configure(coupler.model)
        if changed:
            bodies = [body for pool in self.pools for body in pool.components]
            self.render_poses = torch.zeros((len(bodies), 7), dtype=torch.float32, device=env.device)
            self.render_poses[:, 3] = 1.
            env.scene.set_gpu_render_only_bodies(bodies, self.render_poses)
        self.update_visuals()

    def update_visuals(self):
        if self.render_poses is None:
            return
        offset = 0
        for pool, coupler in zip(self.pools, self.couplers):
            positions = pool.update(coupler)
            self.render_poses[offset:offset + len(positions), :3] = torch.as_tensor(positions, device=self.env.device)
            offset += len(pool.entities)

    def padded(self, material=False):
        records = []
        for coupler in self.couplers:
            if coupler is None:
                raise RuntimeError('All batched MPM models must be initialized')
            if material:
                records.append({k: getattr(coupler.model.struct, k).numpy()[:coupler.model.struct.n_particles]
                                for k in MATERIAL_FIELDS})
            else:
                records.append(coupler.particle_state())
        output = {}
        for key, value in records[0].items():
            output[key] = torch.zeros((self.count, self.capacity, *value.shape[1:]),
                                      dtype=torch.as_tensor(value).dtype, device=self.env.device)
            for i, record in enumerate(records):
                value = record[key]
                output[key][i, :len(value)] = torch.as_tensor(value, device=self.env.device)
        return output

    def metadata(self):
        counts = torch.tensor([c.model.struct.n_particles for c in self.couplers], device=self.env.device)
        return dict(count=counts[:, None], mask=torch.arange(self.capacity, device=self.env.device)[None] < counts[:, None])

    def state_dict(self):
        env = self.env
        state = BaseEnv.get_state_dict(env)
        state.update(mpm=self.padded(), mpm_material=self.padded(True), mpm_meta=self.metadata(),
                     mpm_drives=dict(position=env.agent.robot.get_drive_targets(), velocity=env.agent.robot.get_drive_velocities()),
                     controller=env.agent.get_controller_state())
        return clone_tree(state)

    def restore(self, state, env_idx=None):
        env = self.env
        if not env._mpm_reset_active:
            raise RuntimeError('Soft-body state assignment is only allowed during reset')
        indices = self.indices(env_idx)
        if any(i not in self.reset_indices for i in indices):
            raise ValueError('Checkpoint targets must be among the environments being reset')
        if indices != sorted(indices):
            order = np.argsort(indices).tolist()
            def reorder(value):
                if isinstance(value, dict):
                    return {k: reorder(v) for k, v in value.items()}
                value = torch.as_tensor(value)
                if value.ndim < 1 or len(value) != len(indices):
                    raise ValueError('Checkpoint tensor must match selected environment rows')
                return value[order]
            # Native articulation/actor setters consume boolean mask order.
            return self.restore(reorder(state), sorted(indices))
        size = len(indices)
        counts = torch.as_tensor(state['mpm_meta']['count']).detach().cpu().numpy()
        if (counts.shape != (size, 1) or not np.isfinite(counts).all() or np.any(counts != counts.astype(np.int64))
                or np.any((counts < 1) | (counts > self.capacity))):
            raise ValueError('Invalid batched particle counts')
        counts = counts[:, 0].astype(int)
        mask = torch.as_tensor(state['mpm_meta']['mask']).detach().cpu().numpy()
        if not np.array_equal(mask, np.arange(self.capacity)[None] < counts[:, None]):
            raise ValueError('Particle mask must be the live prefix specified by count')
        values, materials = {}, {}
        for name, target, template in [('mpm', values, self.padded()), ('mpm_material', materials, self.padded(True))]:
            if set(state[name]) != set(template):
                raise ValueError('Checkpoint particle field layout changed')
            for key, expected in template.items():
                value = torch.as_tensor(state[name][key]).detach().cpu().numpy()
                if value.shape != (size, *expected.shape[1:]) or not np.isfinite(value).all():
                    raise ValueError('Invalid batched particle tensor: ' + key)
                if any(np.any(value[row, n:] != 0) for row, n in enumerate(counts)):
                    raise ValueError('Inactive particle padding must be zero')
                target[key] = value
        for row, n in enumerate(counts):
            if any(np.any(materials[k][row, :n] <= 0) for k in ('particle_mass', 'particle_vol')):
                raise ValueError('Live particle mass and volume must be positive')
            if not np.isin(materials['particle_type'][row, :n], (0, 1, 2)).all():
                raise ValueError('Unsupported particle material type')
            if bool(np.any(materials['particle_type'][row, :n] == 2)) != self.couplers[indices[row]].has_fluid_particles:
                raise ValueError('Restored material changes the fluid state layout')
            if 'vol' in values and np.any(values['vol'][row, :n] <= 0):
                raise ValueError('Live fluid volume must be positive')
        drives = state['mpm_drives']
        if set(drives) != {'position', 'velocity'}:
            raise ValueError('Both drive target arrays are required')
        drives = {k: torch.as_tensor(v, device=env.device) for k, v in drives.items()}
        if any(v.shape != (size, env.agent.robot.max_dof) or not torch.isfinite(v).all() for v in drives.values()):
            raise ValueError('Invalid batched drive targets')
        def merge(value, template):
            if isinstance(template, dict):
                if not isinstance(value, dict) or set(value) != set(template):
                    raise ValueError('Controller state layout changed')
                return {k: merge(value[k], child) for k, child in template.items()}
            value = torch.as_tensor(value, device=env.device)
            if value.shape != (size, *template.shape[1:]) or not torch.isfinite(value).all():
                raise ValueError('Invalid batched controller state')
            result = template.clone(); result[indices] = value.to(result.dtype)
            return result
        controller = merge(state['controller'], env.agent.get_controller_state())
        for row, index in enumerate(indices):
            n = counts[row]
            if n != self.couplers[index].model.struct.n_particles:
                builder = copy.copy(self.builders[index])
                colors = np.asarray(builder.mpm_particle_colors)
                if not np.all(colors == colors[0]):
                    raise ValueError('Count-changing checkpoints require uniform particle colors')
                builder.mpm_particle_q = values['x'][row, :n].tolist()
                builder.mpm_particle_qd = values['v'][row, :n].tolist()
                for key, field in dict(particle_mass='mpm_particle_mass', particle_vol='mpm_particle_volume',
                                      particle_mu_lam_ys='mpm_particle_mu_lam_ys', particle_friction_cohesion='mpm_particle_friction_cohesion',
                                      particle_type='mpm_particle_type').items():
                    setattr(builder, field, materials[key][row, :n].tolist())
                builder.mpm_particle_colors = np.repeat(colors[:1], n, axis=0).tolist()
                self.rebuild(builder, self.couplers[index].bodies, index)
        extras = {'mpm', 'mpm_material', 'mpm_meta', 'mpm_drives', 'controller'}
        BaseEnv.set_state_dict(env, {k: v for k, v in state.items() if k not in extras}, indices)
        env.agent.set_controller_state(controller)
        old_mask = env.scene._reset_mask
        env.scene._reset_mask = torch.zeros(self.count, dtype=torch.bool, device=env.device)
        env.scene._reset_mask[indices] = True
        try:
            env.agent.robot.set_joint_drive_targets(drives['position'], env.agent.robot.active_joints)
            env.agent.robot.set_joint_drive_velocity_targets(drives['velocity'], env.agent.robot.active_joints)
            env.scene.px.gpu_apply_articulation_target_position()
            env.scene.px.gpu_apply_articulation_target_velocity()
        finally:
            env.scene._reset_mask = old_mask
        env.agent._mpm_passive_adapter = None
        for row, index in enumerate(indices):
            coupler = self.couplers[index]; n = counts[row]
            for key, value in materials.items():
                array = getattr(coupler.model.struct, key); full = array.numpy(); full[:n] = value[row, :n]; array.assign(full)
            for buffer in coupler.states:
                for key, field in PARTICLE_FIELDS.items():
                    if key not in values:
                        continue
                    array = getattr(buffer.struct, field); full = array.numpy(); full[:n] = values[key][row, :n]; array.assign(full)
                if 'vol' not in values:
                    array = buffer.struct.particle_vol; full = array.numpy(); full[:n] = materials['particle_vol'][row, :n]; array.assign(full)
                buffer.struct.error.zero_()
            coupler.failed = False; coupler.pending_step = None
        env.last_coupling_step = None
        self.update_visuals()

    def restore_flat(self, state, env_idx=None):
        indices = self.indices(env_idx)
        value = torch.as_tensor(state, device=self.env.device)
        if value.ndim != 2 or value.shape[0] != len(indices) or not torch.isfinite(value).all():
            raise ValueError('Invalid flat batched checkpoint')
        cursor = 0
        def unpack(template):
            nonlocal cursor
            if isinstance(template, dict):
                return {k: unpack(v) for k, v in template.items()}
            shape = (len(indices), *template.shape[1:]); width = int(np.prod(shape[1:]))
            if cursor + width > value.shape[1]:
                raise ValueError('Batched checkpoint is too short')
            result = value[:, cursor:cursor + width].reshape(shape)
            cursor += width
            return result
        restored = unpack(self.env.get_state_dict())
        if cursor != value.shape[1]:
            raise ValueError('Batched checkpoint has extra values')
        self.env.set_state_dict(restored, indices)

    def clear(self):
        self.builders = [None] * self.count
        self.couplers = [None] * self.count
        self.pools = [None] * self.count
        self.render_poses = None
        if hasattr(self, '_pending_records'):
            del self._pending_records

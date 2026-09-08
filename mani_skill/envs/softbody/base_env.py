"""Experimental single-scene MPM integration with the ManiSkill 3 lifecycle."""

import copy
import numpy as np
import torch
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from .mpm import MPMCoupler, wp
from .gpu_coupling import MPMGPUWorld

MATERIAL_FIELDS = ('particle_mass', 'particle_vol', 'particle_mu_lam_ys',
                   'particle_friction_cohesion', 'particle_type')


class MPMBaseEnv(BaseEnv):
    """Single-scene CPU/GPU PhysX coupling and particle sphere visualization.

    Tasks build their rigid scene normally and call rebuild_mpm(builder, bodies)
    during _initialize_episode, after setting their reset poses. Both engines
    retain their own state; only reaction forces cross the coupling boundary.
    GPU robot controllers must supply a fresh complete mpm_baseline_qf buffer.
    Multiple task instances in one native GPU world remain unsupported here.
    """

    SUPPORTED_OBS_MODES = ("state", "state_dict", "none", "sensor_data", "any_textures", "pointcloud")
    max_checkpoint_particles = 65536
    PARTICLE_CHECKPOINT_GROUPS = ('mpm', 'mpm_material')

    def __init__(self, *args, mpm_device="cuda", mpm_dt=.0005,
                 num_envs=1, sim_backend="physx_cpu", **kwargs):
        if num_envs != 1:
            raise NotImplementedError("MPM task lifecycle currently supports one PhysX scene")
        if mpm_device not in ("cpu", "cuda"):
            raise ValueError("mpm_device must be cpu or cuda")
        self.mpm_device = mpm_device
        self.mpm_dt = float(mpm_dt)
        if not np.isfinite(self.mpm_dt) or self.mpm_dt <= 0:
            raise ValueError("mpm_dt must be positive and finite")
        self.mpm_coupler = None
        self.mpm_gpu_world = None
        self._mpm_builder = None
        self._mpm_reset_active = False
        self.last_coupling_step = None
        self._particle_entities = []
        self._particle_visual_pool = []
        self._particle_visual_components = []
        self._particle_visual_specs = []
        self._particle_render_poses = None
        self._mpm_initial_checkpoint = None
        if sim_backend.split(':')[0] in ('gpu', 'cuda', 'physx_cuda') and not sapien.physx.is_gpu_enabled():
            sapien.physx.enable_gpu()
        wp.init()
        super().__init__(*args, num_envs=1, sim_backend=sim_backend, **kwargs)

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        restore = options.pop('reset_to_env_states', None)
        self._mpm_initial_checkpoint = None
        self._mpm_reset_active = True
        try:
            # The base reset-to-state path skips _initialize_episode. MPM must
            # first build fresh topology/material buffers, including after a
            # scene reconfiguration, then restore the checkpoint during reset.
            obs, info = super().reset(seed=seed, options=options)
            state = restore['env_states'] if restore is not None else self._mpm_initial_checkpoint
            if state is not None:
                if isinstance(state, dict):
                    self.set_state_dict(state, options.get('env_idx'))
                else:
                    self.set_state(state, options.get('env_idx'))
                info = {**self.get_info(), 'reconfigure': info['reconfigure']}
                obs = self.get_obs(info)
                self._last_obs = obs
            return obs, info
        finally:
            self._mpm_initial_checkpoint = None
            self._mpm_reset_active = False

    def defer_initial_state(self, state):
        """Restore a task's recorded start after native controller reset finishes."""
        if not self._mpm_reset_active:
            raise RuntimeError('Initial state selection is only allowed during reset')
        self._mpm_initial_checkpoint = state

    def rebuild_mpm(self, builder, bodies):
        """Build fresh buffers during reset, including clearing prior failures."""
        if not self._mpm_reset_active:
            raise RuntimeError("MPM rebuilding is only allowed during reset")
        dt = self.scene.sub_scenes[0].get_timestep()
        substeps = round(dt / self.mpm_dt)
        if substeps < 1 or not np.isclose(substeps * self.mpm_dt, dt, rtol=1e-7, atol=1e-10):
            raise ValueError("PhysX timestep must be an integer multiple of mpm_dt")
        model = builder.finalize(self.mpm_device)
        model.gravity = np.asarray(self.sim_config.scene_config.gravity, dtype=np.float32)
        self._configure_mpm_model(model)
        states = [model.state() for _ in range(substeps + 1)]
        builder.init_model_state(model, states)
        if self.gpu_sim_enabled:
            self.mpm_gpu_world = MPMGPUWorld(self.scene.px)
            self.mpm_coupler = self.mpm_gpu_world.add_model(self.scene.sub_scenes[0], model, states,
                                                          bodies, mpm_dt=self.mpm_dt)
        else:
            self.mpm_coupler = MPMCoupler(self.scene.sub_scenes[0], model, states,
                                          bodies, mpm_dt=self.mpm_dt)
        self._mpm_builder = builder
        self.last_coupling_step = None
        self._setup_particle_rendering()

    def _setup_particle_rendering(self):
        if not self.scene.can_render():
            return
        scene = self.scene.sub_scenes[0]
        count = self.mpm_coupler.model.struct.n_particles
        # Minimal shaders omit PointCloudComponent. Sphere visuals populate
        # normal camera color/depth/segmentation. They have no physics component
        # and do not enter the simulation state registry. Separate entities are
        # required because attached render-shape local poses are immutable.
        # Keep identities across resets. Recreating every visual monotonically
        # consumes scene IDs, overflowing minimal shaders' signed-16-bit labels
        # even for the 19,404-particle Write task after only one reset.
        colors = np.asarray(self.mpm_coupler.model.mpm_particle_colors[:count], dtype=np.float32)
        radius = self.mpm_coupler.model.struct.particle_radius
        specs = [(radius, *color) for color in colors]
        changed = count != len(self._particle_entities) or any(
            previous != current for previous, current in zip(self._particle_visual_specs, specs))
        if changed and self.gpu_sim_enabled:
            self._invalidate_particle_render_groups()
        prototypes = {}
        for i, color in enumerate(colors):
            if i < len(self._particle_visual_pool) and self._particle_visual_specs[i] == specs[i]:
                self._particle_visual_components[i].visibility = 1.
                continue
            key = tuple(color)
            if key not in prototypes:
                material = sapien.render.RenderMaterial(base_color=[*key, 1.], roughness=.8)
                prototypes[key] = sapien.render.RenderShapeSphere(radius, material)
            component = sapien.render.RenderBodyComponent()
            component.attach(prototypes[key].clone())
            if i < len(self._particle_visual_pool):
                entity = self._particle_visual_pool[i]
                entity.remove_component(self._particle_visual_components[i])
                entity.add_component(component)
                self._particle_visual_components[i] = component
                self._particle_visual_specs[i] = specs[i]
            else:
                entity = sapien.Entity()
                entity.name = f'mpm_particle_visual_only_{i}'
                entity.add_component(component)
                scene.add_entity(entity)
                self._particle_visual_pool.append(entity)
                self._particle_visual_components.append(component)
                self._particle_visual_specs.append(specs[i])
        for component in self._particle_visual_components[count:]:
            component.visibility = 0.
        self._particle_entities = self._particle_visual_pool[:count]
        if self.gpu_sim_enabled and (changed or self._particle_render_poses is None):
            self._particle_render_poses = torch.zeros((len(self._particle_visual_pool), 7),
                                                      dtype=torch.float32, device=self.device)
            self._particle_render_poses[:, 3] = 1.
            self.scene.set_gpu_render_only_bodies(self._particle_visual_components, self._particle_render_poses)
        self._update_particle_rendering()

    def _invalidate_particle_render_groups(self):
        """Release groups before changing their render-object membership."""
        for sensor in (*self.scene.sensors.values(), *self.scene.human_render_cameras.values()):
            camera = getattr(sensor, 'camera', None)
            if camera is not None:
                camera.camera_group = None
        self.scene.camera_groups.clear()
        self.scene.render_system_group = None
        self.scene._sensors_initialized = False
        self.scene._human_render_cameras_initialized = False

    def _update_particle_rendering(self):
        if self._particle_entities:
            positions = self.mpm_coupler.states[0].struct.particle_q.numpy()[:len(self._particle_entities)]
            for entity, position in zip(self._particle_entities, positions):
                entity.pose = sapien.Pose(position)
            if self._particle_render_poses is not None:
                self._particle_render_poses[:len(positions), :3] = torch.as_tensor(positions, device=self.device)

    def _after_control_step(self):
        self._update_particle_rendering()
        super()._after_control_step()

    def _configure_mpm_model(self, model):
        """Task-specific contact/material configuration, applied before buffers."""

    def _before_simulation_step(self):
        super()._before_simulation_step()
        if self.mpm_coupler is None:
            raise RuntimeError("Task reset did not build an MPM model")
        if self.gpu_sim_enabled:
            baseline = self.agent.mpm_baseline_qf if self.agent is not None else None
            self.mpm_gpu_world.prepare_step(baseline_qf=baseline)
        else:
            self.mpm_coupler.prepare_step()

    def _after_simulation_step(self):
        self.last_coupling_step = (self.mpm_gpu_world.complete_step()[0] if self.gpu_sim_enabled
                                   else self.mpm_coupler.complete_step())
        super()._after_simulation_step()

    def rigid_pose(self, body):
        """Read scene-local native GPU pose without overwriting pending reset data."""
        if not self.gpu_sim_enabled or not isinstance(body, sapien.physx.PhysxRigidBodyComponent):
            return body.entity_pose
        px = self.scene.px
        if not self._mpm_reset_active:
            if isinstance(body, sapien.physx.PhysxArticulationLinkComponent):
                px.gpu_fetch_articulation_link_pose()
            else:
                px.gpu_fetch_rigid_dynamic_data()
        row = px.cuda_rigid_body_data.torch()[body.gpu_pose_index, :7].cpu().numpy()
        return sapien.Pose(row[:3], row[3:])

    def reset_rigid_velocity(self, body, linear, angular):
        if not self._mpm_reset_active:
            raise RuntimeError('Rigid state assignment is only allowed during reset')
        if self.gpu_sim_enabled:
            data = self.scene.px.cuda_rigid_body_data.torch()
            data[body.gpu_pose_index, 7:13] = torch.as_tensor([*linear, *angular], dtype=data.dtype, device=data.device)
        else:
            body.linear_velocity = linear
            body.angular_velocity = angular

    def rigid_velocity(self, body):
        """Read world linear/angular velocity, including native GPU link data."""
        if not isinstance(body, sapien.physx.PhysxRigidBodyComponent):
            return np.zeros(6, dtype=np.float32)
        if not self.gpu_sim_enabled:
            return np.r_[body.linear_velocity, body.angular_velocity]
        px = self.scene.px
        if not self._mpm_reset_active:
            if isinstance(body, sapien.physx.PhysxArticulationLinkComponent):
                px.gpu_fetch_articulation_link_velocity()
            else:
                px.gpu_fetch_rigid_dynamic_data()
        return px.cuda_rigid_body_data.torch()[body.gpu_pose_index, 7:13].cpu().numpy().copy()

    def _get_obs_agent(self):
        return super()._get_obs_agent() if self.agent is not None else {}

    def _mpm_tensors(self):
        if self.mpm_coupler is None:
            raise RuntimeError("Task reset did not build an MPM model")
        return {key: torch.as_tensor(value, device=self.device).unsqueeze(0)
                for key, value in self.mpm_coupler.particle_state().items()}

    def _mpm_material_tensors(self):
        model = self.mpm_coupler.model
        return {key: torch.as_tensor(getattr(model.struct, key).numpy()[:model.struct.n_particles],
                                     device=self.device).unsqueeze(0) for key in MATERIAL_FIELDS}

    def _get_obs_extra(self, info):
        return {**super()._get_obs_extra(info), "mpm": self._mpm_tensors()}

    def _flatten_raw_obs(self, obs):
        if self.obs_mode == "state":
            return common.flatten_state_dict(self._flat_leaves(obs), use_torch=True, device=self.device)
        return super()._flatten_raw_obs(obs)

    def _flat_leaves(self, state):
        return {key: self._flat_leaves(value) if isinstance(value, dict)
                else torch.as_tensor(value, device=self.device).reshape(self.num_envs, -1)
                for key, value in state.items()}

    def get_state_dict(self):
        state = super().get_state_dict() if self.agent is not None else self.scene.get_sim_state()
        if self.agent is not None:
            # Native CPU drive targets and controller memory are separate from
            # qpos/qvel. Both are needed to replay target-relative controllers.
            state['mpm_drives'] = {
                'position': self.agent.robot.get_drive_targets(),
                'velocity': self.agent.robot.get_drive_velocities(),
            }
        state["mpm"] = self._mpm_tensors()
        state['mpm_material'] = self._mpm_material_tensors()
        def copy_tree(value):
            return {k: copy_tree(v) for k, v in value.items()} if isinstance(value, dict) else value.clone()
        return copy_tree(state)

    def get_state(self):
        # Checkpoints must preserve float64 task inputs. The observation-oriented
        # common flattener casts nested float64 leaves to float32. Concatenating
        # leaves directly promotes dtypes without rounding those saved inputs.
        leaves = common.flatten_dict_keys(self._flat_leaves(self.get_state_dict()))
        return torch.cat(list(leaves.values()), dim=1)

    def set_state_dict(self, state, env_idx=None):
        if not self._mpm_reset_active:
            raise RuntimeError("Soft-body state assignment is only allowed during reset")
        if self.mpm_coupler is None:
            raise RuntimeError("Build the task's MPM model before restoring its state")
        if env_idx is not None and list(torch.as_tensor(env_idx).cpu().numpy()) != [0]:
            raise ValueError("Only the single environment index [0] is supported")
        expected = self.mpm_coupler.particle_state()
        if set(state.get("mpm", {})) != set(expected):
            raise ValueError(f"Restored MPM fields must match {sorted(expected)}")
        positions = torch.as_tensor(state['mpm']['x'])
        if positions.ndim != 3 or positions.shape[0] != 1 or positions.shape[2] != 3:
            raise ValueError('Invalid MPM position shape')
        count = positions.shape[1]
        self._validate_checkpoint_count(count)
        values = {}
        for key, initial in expected.items():
            value = torch.as_tensor(state["mpm"][key]).detach().cpu().numpy()
            if value.shape != (1, count, *initial.shape[1:]) or not np.isfinite(value).all():
                raise ValueError(f"Invalid MPM state field: {key}")
            if key == 'vol' and np.any(value <= 0):
                raise ValueError('Current fluid particle volumes must be positive')
            values[key] = value[0]
        template = self._mpm_material_tensors()
        if set(state.get('mpm_material', {})) != set(template):
            raise ValueError('Restored state must contain the MPM material arrays')
        materials = {}
        for key, initial in template.items():
            value = torch.as_tensor(state['mpm_material'][key]).detach().cpu().numpy()
            if value.shape != (1, count, *initial.shape[2:]) or not np.isfinite(value).all():
                raise ValueError(f'Invalid MPM material array: {key}')
            if key in ('particle_mass', 'particle_vol') and np.any(value <= 0):
                raise ValueError(f'MPM {key} must be positive')
            if key == 'particle_type' and np.any(~np.isin(value, (0, 1, 2))):
                raise ValueError('MPM particle types must be 0, 1 or 2')
            materials[key] = value[0]
        if bool(np.any(materials['particle_type'] == 2)) != self.mpm_coupler.has_fluid_particles:
            raise ValueError('Restored material changes the fluid state layout')
        drives = None
        if self.agent is not None:
            joints = self.agent.robot._objs[0].active_joints
            drives = state.get('mpm_drives', {})
            if set(drives) != {'position', 'velocity'}:
                raise ValueError('Restored robot state requires both drive target arrays')
            drives = {k: torch.as_tensor(v).detach().cpu().numpy() for k, v in drives.items()}
            if any(v.shape != (1, len(joints)) or not np.isfinite(v).all() for v in drives.values()):
                raise ValueError('Invalid robot drive targets')
            def restore_controller(value, template):
                if isinstance(template, dict):
                    if not isinstance(value, dict) or set(value) != set(template):
                        raise ValueError('Controller state does not match the selected control mode')
                    return {k: restore_controller(value[k], v) for k, v in template.items()}
                tensor = torch.as_tensor(value, device=self.device)
                if tensor.shape != template.shape or not torch.isfinite(tensor).all():
                    raise ValueError('Invalid controller state tensor')
                return tensor.to(dtype=template.dtype).clone()
            controller = restore_controller(state.get('controller', {}), self.agent.get_controller_state())
        if count != self.mpm_coupler.model.struct.n_particles:
            self._resize_checkpoint_particles(values, materials)
        super().set_state_dict({k: v for k, v in state.items() if k not in ('mpm', 'mpm_material', 'mpm_drives', 'controller')}, env_idx)
        if drives is not None:
            self.agent.set_controller_state(controller)
            self.agent.robot.set_joint_drive_targets(torch.as_tensor(drives['position'], device=self.device), self.agent.robot.active_joints)
            self.agent.robot.set_joint_drive_velocity_targets(torch.as_tensor(drives['velocity'], device=self.device), self.agent.robot.active_joints)
            if self.gpu_sim_enabled:
                self.scene.px.gpu_apply_articulation_target_position()
                self.scene.px.gpu_apply_articulation_target_velocity()
                self.agent._mpm_passive_adapter = None
        for key, value in materials.items():
            target = getattr(self.mpm_coupler.model.struct, key)
            full = target.numpy(); full[:len(value)] = value; target.assign(full)
        for buffer in self.mpm_coupler.states:
            for key, field in (("x", "particle_q"), ("v", "particle_qd"), ("F", "particle_F"),
                               ("C", "particle_C"), ("vc", "particle_volume_correction")):
                target = getattr(buffer.struct, field)
                full = target.numpy()
                full[:len(values[key])] = values[key]
                target.assign(full)
            target = buffer.struct.particle_vol
            volume = values.get('vol', materials['particle_vol'])
            full = target.numpy(); full[:len(volume)] = volume; target.assign(full)
            buffer.struct.error.zero_()
        self.mpm_coupler.failed = False
        self.mpm_coupler.pending_step = None
        if self.gpu_sim_enabled:
            self.mpm_gpu_world.failed = False
            self.mpm_gpu_world.pending_step = False
        self.last_coupling_step = None
        self._update_particle_rendering()

    def _validate_checkpoint_count(self, count):
        capacity = min(self.max_checkpoint_particles,
                       getattr(self, 'observation_particle_capacity', self.max_checkpoint_particles))
        if not 0 < count <= capacity:
            raise ValueError(f'Checkpoint particle count must be between 1 and {capacity}')

    def _resize_checkpoint_particles(self, values, materials):
        """Rebuild only particle buffers, retaining the fresh scene's rigid model.

        Current task checkpoints omit visual colors. Count-changing restore is
        therefore supported for uniform-color recipes; heterogeneous visuals
        need an explicit per-particle visual checkpoint contract first.
        """
        if not self._mpm_reset_active or self._mpm_builder is None:
            raise RuntimeError('Particle topology may only be rebuilt during reset')
        colors = np.asarray(self._mpm_builder.mpm_particle_colors)
        if not len(colors) or not np.all(colors == colors[0]):
            raise ValueError('Count-changing checkpoints require a uniform-color particle recipe')
        builder = copy.copy(self._mpm_builder)
        builder.mpm_particle_q = values['x'].tolist()
        builder.mpm_particle_qd = values['v'].tolist()
        for source, target in (('particle_mass', 'mpm_particle_mass'),
                               ('particle_vol', 'mpm_particle_volume'),
                               ('particle_mu_lam_ys', 'mpm_particle_mu_lam_ys'),
                               ('particle_friction_cohesion', 'mpm_particle_friction_cohesion'),
                               ('particle_type', 'mpm_particle_type')):
            setattr(builder, target, materials[source].tolist())
        builder.mpm_particle_colors = np.repeat(colors[:1], len(values['x']), axis=0).tolist()
        self.rebuild_mpm(builder, self.mpm_coupler.bodies)

    def set_state(self, state, env_idx=None):
        if not self._mpm_reset_active:
            raise RuntimeError('Soft-body state assignment is only allowed during reset')
        state = torch.as_tensor(state, device=self.device)
        if state.ndim != 2 or state.shape[0] != 1 or not torch.isfinite(state).all():
            raise ValueError("Expected finite batched soft-body state with shape (1, state_size)")
        template = self.get_state_dict()
        live_count = self.mpm_coupler.model.struct.n_particles
        particle_width = sum(int(np.prod(value.shape[2:]))
                             for group in self.PARTICLE_CHECKPOINT_GROUPS for value in template[group].values())
        flat_size = sum(value.numel() for value in common.flatten_dict_keys(template).values())
        particle_values = state.shape[1] - (flat_size - live_count*particle_width)
        if particle_values % particle_width:
            raise ValueError('Soft-body state vector has an inconsistent particle layout')
        count = particle_values // particle_width
        self._validate_checkpoint_count(count)
        cursor = 0
        def unpack(template, particle_group=False):
            nonlocal cursor
            result = {}
            for key, value in template.items():
                if isinstance(value, dict):
                    result[key] = unpack(value, key in self.PARTICLE_CHECKPOINT_GROUPS)
                else:
                    shape = (1, count, *value.shape[2:]) if particle_group else value.shape
                    size = int(np.prod(shape[1:]))
                    if cursor + size > state.shape[1]:
                        raise ValueError("Soft-body state vector is too short")
                    result[key] = state[:, cursor:cursor+size].reshape(shape)
                    cursor += size
            return result
        restored = unpack(template)
        if cursor != state.shape[1]:
            raise ValueError("Soft-body state vector has extra values")
        self.set_state_dict(restored, env_idx)

    def _clear(self):
        self._particle_render_poses = None
        self._particle_entities = []
        self._particle_visual_pool = []
        self._particle_visual_components = []
        self._particle_visual_specs = []
        self.mpm_coupler = None
        self.mpm_gpu_world = None
        self._mpm_builder = None
        self.last_coupling_step = None
        super()._clear()

"""Experimental single-scene MPM integration with the ManiSkill 3 lifecycle."""

import numpy as np
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from .mpm import MPMCoupler, wp


class MPMBaseEnv(BaseEnv):
    """CPU PhysX with CPU or CUDA MPM; state observations only for now.

    Tasks build their rigid scene normally and call rebuild_mpm(builder, bodies)
    during _initialize_episode, after setting their reset poses. Both engines
    retain their own state; only reaction forces cross the coupling boundary.
    """

    SUPPORTED_OBS_MODES = ("state", "state_dict", "none")

    def __init__(self, *args, mpm_device="cuda", mpm_dt=.0005,
                 num_envs=1, sim_backend="physx_cpu", **kwargs):
        if num_envs != 1 or sim_backend not in ("auto", "cpu", "physx_cpu"):
            raise NotImplementedError("MPM currently supports one CPU PhysX scene")
        if mpm_device not in ("cpu", "cuda"):
            raise ValueError("mpm_device must be cpu or cuda")
        self.mpm_device = mpm_device
        self.mpm_dt = float(mpm_dt)
        if not np.isfinite(self.mpm_dt) or self.mpm_dt <= 0:
            raise ValueError("mpm_dt must be positive and finite")
        self.mpm_coupler = None
        self._mpm_reset_active = False
        self.last_coupling_step = None
        wp.init()
        super().__init__(*args, num_envs=1, sim_backend="physx_cpu", **kwargs)

    def reset(self, *, seed=None, options=None):
        self._mpm_reset_active = True
        try:
            return super().reset(seed=seed, options=options)
        finally:
            self._mpm_reset_active = False

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
        self.mpm_coupler = MPMCoupler(self.scene.sub_scenes[0], model, states,
                                      bodies, mpm_dt=self.mpm_dt)
        self.last_coupling_step = None

    def _configure_mpm_model(self, model):
        """Task-specific contact/material configuration, applied before buffers."""

    def _before_simulation_step(self):
        super()._before_simulation_step()
        if self.mpm_coupler is None:
            raise RuntimeError("Task reset did not build an MPM model")
        self.mpm_coupler.prepare_step()

    def _after_simulation_step(self):
        self.last_coupling_step = self.mpm_coupler.complete_step()
        super()._after_simulation_step()

    def _get_obs_agent(self):
        return super()._get_obs_agent() if self.agent is not None else {}

    def _mpm_tensors(self):
        if self.mpm_coupler is None:
            raise RuntimeError("Task reset did not build an MPM model")
        return {key: torch.as_tensor(value, device=self.device).unsqueeze(0)
                for key, value in self.mpm_coupler.particle_state().items()}

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
        state["mpm"] = self._mpm_tensors()
        return state

    def get_state(self):
        # Particle tensors have more dimensions than the base rigid-state flattener.
        return common.flatten_state_dict(self._flat_leaves(self.get_state_dict()), use_torch=True, device=self.device)

    def set_state_dict(self, state, env_idx=None):
        if not self._mpm_reset_active:
            raise RuntimeError("Soft-body state assignment is only allowed during reset")
        if self.mpm_coupler is None:
            raise RuntimeError("Build the task's MPM model before restoring its state")
        if env_idx is not None and list(torch.as_tensor(env_idx).cpu().numpy()) != [0]:
            raise ValueError("Only the single environment index [0] is supported")
        expected = self.mpm_coupler.particle_state()
        if set(state.get("mpm", {})) != set(expected):
            raise ValueError("Restored state must contain x, v, F, C, and vc")
        values = {}
        for key, initial in expected.items():
            value = torch.as_tensor(state["mpm"][key]).detach().cpu().numpy()
            if value.shape != (1, *initial.shape) or not np.isfinite(value).all():
                raise ValueError(f"Invalid MPM state field: {key}")
            values[key] = value[0]
        super().set_state_dict({k: v for k, v in state.items() if k != "mpm"}, env_idx)
        for buffer in self.mpm_coupler.states:
            for key, field in (("x", "particle_q"), ("v", "particle_qd"), ("F", "particle_F"),
                               ("C", "particle_C"), ("vc", "particle_volume_correction")):
                target = getattr(buffer.struct, field)
                full = target.numpy()
                full[:len(values[key])] = values[key]
                target.assign(full)
            buffer.struct.error.zero_()
        self.mpm_coupler.failed = False
        self.mpm_coupler.pending_step = None
        self.last_coupling_step = None

    def set_state(self, state, env_idx=None):
        state = torch.as_tensor(state, device=self.device)
        if state.ndim != 2 or state.shape[0] != 1 or not torch.isfinite(state).all():
            raise ValueError("Expected finite batched soft-body state with shape (1, state_size)")
        cursor = 0
        def unpack(template):
            nonlocal cursor
            result = {}
            for key, value in template.items():
                if isinstance(value, dict):
                    result[key] = unpack(value)
                else:
                    size = int(np.prod(value.shape[1:]))
                    if cursor + size > state.shape[1]:
                        raise ValueError("Soft-body state vector is too short")
                    result[key] = state[:, cursor:cursor+size].reshape(value.shape)
                    cursor += size
            return result
        restored = unpack(self.get_state_dict())
        if cursor != state.shape[1]:
            raise ValueError("Soft-body state vector has extra values")
        self.set_state_dict(restored, env_idx)

    def _clear(self):
        self.mpm_coupler = None
        self.last_coupling_step = None
        super()._clear()

"""Check actual ManiSkill hooks, observations, and reset replay using free fall.

This analytic case has no rigid collider: it exercises empty coupling and MPM
time advancement through BaseEnv.step, independent of the contact probes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mani_skill.envs.softbody.base_env import MPMBaseEnv
from mani_skill.envs.softbody.mpm import MPMModelBuilder, wp
from mani_skill.utils.structs.types import SimConfig


class BallisticEnv(MPMBaseEnv):
    diagnostic_density = 1000.
    diagnostic_dimension = 2
    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=500, control_freq=50)

    def _initialize_episode(self, env_idx, options):
        builder = MPMModelBuilder()
        builder.set_mpm_domain([.4, .4, .6], grid_length=.01)
        builder.add_mpm_grid(pos=(0., 0., .3), vel=(.05, 0., .1),
                             dim_x=self.diagnostic_dimension, dim_y=self.diagnostic_dimension, dim_z=self.diagnostic_dimension,
                             cell_x=.005, cell_y=.005, cell_z=.005,
                             density=self.diagnostic_density, mu_lambda_ys=(0., 0., 0.),
                             friction_cohesion=(0., 0., 0.), type=0, jitter=False)
        self.rebuild_mpm(builder, [])

    def _configure_mpm_model(self, model):
        model.grid_contact = False
        model.particle_contact = False
        model.adaptive_grid = False


def run(device, output):
    wp.config.kernel_cache_dir = str((output.parent / "warp-cache").resolve())
    env = BallisticEnv(mpm_device=device, robot_uids="none", obs_mode="state_dict",
                       reward_mode="none", render_backend="none")
    try:
        initial_obs, _ = env.reset(seed=42)
        initial = {k: v.clone() for k, v in initial_obs["extra"]["mpm"].items()}
        snapshots = [env.get_state_dict()]
        trajectory = [initial["x"].numpy().copy()]
        max_position_error = 0.
        max_velocity_error = 0.
        for i in range(5):
            obs, reward, terminated, truncated, info = env.step(None)
            assert env.observation_space.contains({"agent": {}, "extra": {"mpm": {
                k: v.numpy() for k, v in obs["extra"]["mpm"].items()}}})
            assert reward.shape == terminated.shape == truncated.shape == (1,)
            assert not terminated.any() and not truncated.any()
            assert int(info["elapsed_steps"][0]) == i + 1
            # Symplectic Euler: acceleration is applied before every particle move.
            dt = env.mpm_dt
            n = (i + 1) * 40
            gravity = np.array([0., 0., -9.81])
            x = initial["x"].numpy() + initial["v"].numpy() * (n * dt) + gravity * (dt**2 * n * (n + 1) / 2)
            v = initial["v"].numpy() + gravity * n * dt
            max_position_error = max(max_position_error, float(np.max(abs(obs["extra"]["mpm"]["x"].numpy() - x))))
            max_velocity_error = max(max_velocity_error, float(np.max(abs(obs["extra"]["mpm"]["v"].numpy() - v))))
            trajectory.append(obs["extra"]["mpm"]["x"].numpy().copy())
            snapshots.append(env.get_state_dict())
        # Restore a midpoint through reset and replay the remaining control steps.
        midpoint = snapshots[2]
        restored, _ = env.reset(options={"reset_to_env_states": {"env_states": midpoint}})
        assert all(torch.equal(restored["extra"]["mpm"][k], v) for k, v in midpoint["mpm"].items())
        replay_error = 0.
        for target in trajectory[3:]:
            obs, *_ = env.step(None)
            replay_error = max(replay_error, float(np.max(abs(obs["extra"]["mpm"]["x"].numpy() - target))))
        flat_state = env.get_state().clone()
        expected_state = env.get_state_dict()["mpm"]
        env.step(None)
        env.reset(options={"reset_to_env_states": {"env_states": flat_state}})
        assert all(torch.equal(env.get_state_dict()["mpm"][k], v) for k, v in expected_state.items())
        reset_obs, _ = env.reset(seed=42)
        assert all(torch.equal(reset_obs["extra"]["mpm"][k], v) for k, v in initial.items())
        reset_obs, _ = env.reset(seed=42, options={"reconfigure": True})
        assert all(torch.equal(reset_obs["extra"]["mpm"][k], v) for k, v in initial.items())
        outside_reset_rejected = False
        try:
            env.set_state_dict(midpoint)
        except RuntimeError:
            outside_reset_rejected = True
        # A changed reset recipe must not silently replace checkpoint materials.
        # This pressureless analytic fixture has no physical robot or grasp.
        checkpoint = env.get_state_dict()
        original_mass = checkpoint['mpm_material']['particle_mass']
        env.diagnostic_density = 1200.
        env.reset(seed=42)
        assert not torch.equal(env.get_state_dict()['mpm_material']['particle_mass'], original_mass)
        env.reset(seed=42, options={'reset_to_env_states': {'env_states': checkpoint}})
        assert all(torch.equal(env.get_state_dict()['mpm_material'][k], v)
                   for k, v in checkpoint['mpm_material'].items())
        report = {"probe": "maniskill3-mpm-environment-lifecycle", "device": device,
                  "passed": max_position_error < 3e-6 and max_velocity_error < 3e-5 and replay_error < 3e-6 and outside_reset_rejected,
                  "max_position_error_m": max_position_error, "max_velocity_error_m_s": max_velocity_error,
                  "reset_replay_error_m": replay_error, "outside_reset_assignment_rejected": outside_reset_rejected,
                  "checkpoint_material_overrides_changed_reset_recipe": True,
                  "particles": 27, "control_steps": 5, "simulated_time_s": .1,
                  "fixture": {"creator": "sim-infra port work", "license": "Apache-2.0 for probe; bundled MPM has separate terms",
                              "source": "repo://ManiSkill/tools/softbody/probe_environment.py", "units": "m, kg, s",
                              "frame": "Z-up", "material_basis": "Pressureless diagnostic particles; no calibrated material claim",
                              "retrieved": "2026-09-08", "file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
        np.savez_compressed(output / "trajectory.npz", x=np.asarray(trajectory))
        return report
    finally:
        env.close()


def check_flat_observation(device):
    env = BallisticEnv(mpm_device=device, robot_uids="none", obs_mode="state",
                       reward_mode="none", render_backend="none")
    try:
        obs, _ = env.reset(seed=42)
        assert obs.shape == (1, 27 * 25)
        assert env.observation_space.contains(obs.numpy())
        obs, *_ = env.step(None)
        assert obs.shape == (1, 27 * 25) and torch.isfinite(obs).all()
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "error": "Probe did not finish"}
    try:
        report = run(args.device, args.output)
        check_flat_observation(args.device)
        report["flat_observation_and_state_replay"] = True
    except BaseException as exc:
        report["passed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)

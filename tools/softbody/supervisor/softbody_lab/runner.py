"""Simulation-side capture. Run this in a separate process/container per rollout."""

from __future__ import annotations

import importlib
import importlib.metadata
import copy
import os
import platform
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np

from .artifacts import (
    InvalidArtifact, TraceWriter, child_file, digest_arrays, digest_json, json_value,
    load_frame, sha256, validate_trace, write_json, initial_numeric_state,
)
from .fixtures import load_fixture
from .telemetry import snapshot_with_telemetry, validate_fields, WRENCH_FIELD

REFERENCE_COMMIT = "493be36121a9dd06071a57172274babe617b789f"
CANDIDATE_BASE = "62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3"
ENVIRONMENTS = ("Fill-v0", "Hang-v0", "Pour-v0", "Pinch-v0", "Write-v0", "Excavate-v0")


def restore_legacy_actor_pose(actor, desired, pose_type):
    """Restore actual entity pose despite SAPIEN 2 COM-transform roundoff.

    This is reset-only API compensation, never a reported-state substitution.
    A changed orientation or nonconvergent translation remains a hard failure.
    """
    desired = np.asarray(desired)
    request = desired.astype(np.float64).copy()
    for _ in range(4):
        actor.set_pose(pose_type(request[:3], desired[3:]))
        actual = np.r_[actor.pose.p, actor.pose.q]
        if np.array_equal(actual, desired):
            return
        if not np.array_equal(actual[3:], desired[3:]):
            raise InvalidArtifact('Legacy actor pose orientation did not restore exactly')
        request[:3] += desired[:3] - actual[:3]
    raise InvalidArtifact('Legacy actor pose translation did not restore exactly')


def command_output(args, *, cwd=None):
    p = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=30)
    if p.returncode:
        raise RuntimeError(f"{' '.join(args[:2])} failed: {p.stderr.strip()}")
    return p.stdout.strip()


def doctor() -> dict:
    result = {"platform": platform.system(), "machine": platform.machine(),
              "python": platform.python_version(), "cuda_reference_ready": False}
    try:
        result["gpu"] = command_output([
            "nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"])
        result["cuda_reference_ready"] = platform.system() == "Linux"
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        result["gpu_error"] = str(exc)
    result["meaning"] = ("Hardware preflight only; compilation and physics probes remain required."
                         if result["cuda_reference_ready"] else
                         "Local evaluator development is supported; legacy MPM needs Linux/NVIDIA CUDA.")
    return result


def provenance(source: Path, role: str) -> dict:
    commit = command_output(["git", "rev-parse", "HEAD"], cwd=source)
    dirty = command_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=source)
    if role == "reference" and (commit != REFERENCE_COMMIT or dirty):
        raise RuntimeError("Reference capture requires the clean pinned ManiSkill 2 checkout")
    runtime = doctor()
    runtime["packages"] = sorted(f"{d.metadata['Name']}=={d.version}"
                                 for d in importlib.metadata.distributions())
    runtime["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    runtime["image_id"] = os.environ.get("SOFTBODY_IMAGE_ID")
    capture_files = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    untracked = command_output(['git', 'ls-files', '--others', '--exclude-standard', '-z'], cwd=source)
    untracked_hashes = {name: sha256(source / name) for name in untracked.split('\0')
                        if name and (source / name).is_file() and not (source / name).is_symlink()}
    return {"run_id": uuid.uuid4().hex, "role": role, "source_commit": commit,
            "source_dirty": bool(dirty),
            "untracked_source_sha256": untracked_hashes,
            "source_archive_sha256": os.environ.get('SOFTBODY_SOURCE_ARCHIVE_SHA256') or None,
            "source_diff_sha256": digest_json(command_output(["git", "diff", "HEAD", "--binary"], cwd=source)),
            "runtime": runtime, "capture_version": digest_json(capture_files)}


class LegacyAdapter:
    """Read-only instrumentation except reset and ordinary environment actions."""

    def __init__(self, env_id, *, control_mode, env_kwargs):
        import gymnasium as gym
        import mani_skill2.envs  # noqa: F401 - task registration
        self.env = gym.make(env_id, obs_mode="none", control_mode=control_mode,
                            **env_kwargs).unwrapped
        self.control_mode = control_mode
        self.env_id = env_id

    def _physical_actors(self):
        # The particle renderer also owns actors in SAPIEN 2. Only collision
        # actors participate in the portable physical scene state.
        actors = [a for a in self.env._actors if a.get_collision_shapes()]
        actors.sort(key=lambda a: a.name)
        if self.env_id in ('Excavate-v0', 'Write-v0'):
            ground = [a for a in actors if a.name == 'ground']
            if len(ground) != 1 or {a.id for a in actors} != {a.id for a in ground + self.env.walls}:
                raise RuntimeError('Unexpected walled-task physical actor set')
            # MS2 gives all four walls the same name. Stable construction order
            # maps them to the unique MS3 registry IDs wall_0 through wall_3.
            return ground + self.env.walls
        if self.env_id == 'Pour-v0':
            if [a.name for a in actors] != ['bottle', 'ground', 'target_beaker']:
                raise RuntimeError('Unexpected Pour physical actor set')
            return [actors[1], actors[0], actors[2]]
        expected = ['ground'] if self.env_id == 'Pinch-v0' else ['ground', 'rod'] if self.env_id == 'Hang-v0' else ['ground', 'target_beaker']
        if [a.name for a in actors] != expected:
            raise RuntimeError('Unexpected physical actor set')
        return actors

    def reset(self, *, seed, reset_kwargs, replay=None, native_initial_state=None):
        if replay is not None and native_initial_state is not None:
            raise ValueError('Choose one reset state source')
        # Legacy Write/Pinch pop level_file from options. Preserve the actual
        # requested reset arguments for provenance and later fixture replay.
        self.env.reset(seed=seed, **copy.deepcopy(reset_kwargs))
        if native_initial_state is not None:
            if native_initial_state.shape != self.env.get_state().shape:
                raise ValueError('Demonstration native state layout differs from this reference')
            # The original demonstration contains one initialization and real
            # controls. Restore only that initial state, never later outcomes.
            self.env.set_state(native_initial_state)
        if replay is not None:
            state, fixture, material = replay
            # Restoring a fixture is part of reset, never performed between actions.
            if fixture.get('initial_state_contract', {}).get('version') == 2:
                import sapien.core as sapien
                env = self.env
                robot = env.agent.robot
                root = state['root_pose'][0]
                robot.set_pose(sapien.Pose(root[:3], root[3:]))
                robot.set_qpos(state['qpos'])
                robot.set_qvel(state['qvel'])
                robot.set_root_velocity(state['root_velocity'][0, :3])
                robot.set_root_angular_velocity(state['root_velocity'][0, 3:])
                for actor, pose, velocity in zip(self._physical_actors(), state['scene_actor_pose'], state['scene_actor_velocity']):
                    if self.env_id == 'Pour-v0' and actor.type == 'dynamic':
                        restore_legacy_actor_pose(actor, pose, sapien.Pose)
                    else:
                        actor.set_pose(sapien.Pose(pose[:3], pose[3:]))
                    if actor.type != 'static':
                        actor.set_velocity(velocity[:3])
                        actor.set_angular_velocity(velocity[3:])
                if self.env_id == 'Fill-v0':
                    env.beaker_x, env.beaker_y = state['task_state']
                elif self.env_id == 'Excavate-v0':
                    env.target_num = int(state['task_state'][0])
                elif self.env_id == 'Pour-v0':
                    env.h1, env.h2 = state['task_state']
                elif self.env_id == 'Pinch-v0':
                    points=state['task_state'][2:].reshape(-1,3).astype(np.float32)
                    if len(points)!=env.n_particles:raise ValueError('Pinch goal particle count mismatch')
                    env.goal_array.assign(points)
                    env.total_deformed_distance=state['task_state'][:2].copy()
                    env.target_dist=.3*sum(env.total_deformed_distance)
                    for key in ('goal_depths','goal_rgbs','goal_cam_pos','goal_cam_rot','goal_cam_intrinsic'):
                        env.info[key]=state[key].copy()
                    env.goal_depths=env.info['goal_depths'];env.goal_rgbs=env.info['goal_rgbs']
                    env.goal_points=env.get_goal_points();env._chamfer_dist=None
                elif self.env_id == 'Write-v0':
                    import warp as wp
                    from mpm.height_rasterizer import rasterize_clear_kernel, rasterize_kernel
                    points=state['task_state'].reshape(-1,3).astype(np.float32)
                    if len(points)!=env.n_particles:
                        raise ValueError('Write goal and particle counts differ')
                    env.info['goal']=points
                    env.goal_array.assign(points)
                    wp.launch(rasterize_clear_kernel,dim=(64,64),inputs=[env.goal_image,0],device='cuda')
                    wp.launch(rasterize_kernel,dim=len(points),inputs=[env.goal_array,wp.vec3(.105,.105,0.),64/.21,1000,2,64,64,env.goal_image],device='cuda')
                    wp.synchronize()
                    env.goal_image_display_numpy=np.clip(env.goal_image.numpy()[:,::-1],0,255).astype(np.uint8)
                    env._iou=None
                else:
                    env.selected_indices = state['task_state'].astype(np.int64).tolist()
                n = env.n_particles
                if material is None or len(state['x']) != n:
                    raise ValueError('Portable replay requires matching particle topology and materials')
                for key, values in material.items():
                    target = getattr(env.mpm_model.struct, key)
                    full = target.numpy(); full[:n] = values; target.assign(full)
                for buffer in env.mpm_states:
                    fields = dict(x='particle_q', v='particle_qd', F='particle_F', C='particle_C', vc='particle_volume_correction')
                    for key, field in fields.items():
                        target = getattr(buffer.struct, field)
                        full = target.numpy(); full[:n] = state[key]; target.assign(full)
                    full = buffer.struct.particle_vol.numpy(); full[:n] = state.get('vol', material['particle_vol'])
                    buffer.struct.particle_vol.assign(full)
                    buffer.struct.error.zero_()
                for i, joint in enumerate(robot.get_active_joints()):
                    joint.set_drive_target(state['drive_position'][i])
                    joint.set_drive_velocity_target(state['drive_velocity'][i])
            else:
                self.env.set_state(state["sim_state"])
            def numeric(value):
                return {k: numeric(v) for k, v in value.items()} if isinstance(value, dict) else np.asarray(value, dtype=np.float32)
            self.env.agent.controller.set_state(numeric(fixture["controller_state"]))
        return self.snapshot()

    def snapshot(self):
        env = self.env
        state = {k: np.array(v, copy=True) for k, v in env.get_mpm_state().items()}
        n = len(state["x"])
        # Pour exports evolving particle volume instead of volume correction.
        # Keep that actual fluid state and also capture the shared kernel buffer;
        # never rename volume into vc or substitute a fabricated zero array.
        if "vc" not in state:
            state["vc"] = env.mpm_states[0].struct.particle_volume_correction.numpy()[:n].copy()
        struct = env.mpm_model.struct
        state["mass"] = np.array(struct.particle_mass.numpy()[:n], copy=True)
        state["qpos"] = np.array(env.agent.robot.get_qpos(), copy=True)
        state["qvel"] = np.array(env.agent.robot.get_qvel(), copy=True)
        state["sim_state"] = np.array(env.get_state(), copy=True)
        joints = env.agent.robot.get_active_joints()
        state["drive_position"] = np.asarray([j.get_drive_target() for j in joints]).reshape(-1)
        state["drive_velocity"] = np.asarray([j.get_drive_velocity_target() for j in joints]).reshape(-1)
        actors = env._get_coupling_actors()
        actors = [a[0] if isinstance(a, (list, tuple)) else a for a in actors]
        state["rigid_pose"] = np.asarray([np.r_[a.pose.p, a.pose.q] for a in actors])
        state["rigid_velocity"] = np.asarray([np.r_[a.velocity, a.angular_velocity] for a in actors])
        if self.env_id in ('Fill-v0', 'Excavate-v0', 'Hang-v0', 'Pour-v0', 'Write-v0', 'Pinch-v0'):
            robot = env.agent.robot
            root_link = robot.get_links()[0]
            physical = self._physical_actors()
            state.update(root_pose=np.asarray([np.r_[root_link.pose.p, root_link.pose.q]]),
                         root_velocity=np.asarray([np.r_[root_link.velocity, root_link.angular_velocity]]),
                         scene_actor_pose=np.asarray([np.r_[a.pose.p, a.pose.q] for a in physical]),
                         scene_actor_velocity=np.asarray([np.zeros(6, dtype=np.float32) if a.type == 'static'
                                                          else np.r_[a.velocity, a.angular_velocity] for a in physical]),
                         task_state=np.asarray([env.beaker_x, env.beaker_y] if self.env_id == 'Fill-v0'
                                               else [env.target_num] if self.env_id == 'Excavate-v0'
                                               else [env.h1, env.h2] if self.env_id == 'Pour-v0'
                                               else np.r_[env.total_deformed_distance,env.goal_array.numpy().reshape(-1)] if self.env_id == 'Pinch-v0'
                                               else env.goal_array.numpy().reshape(-1) if self.env_id == 'Write-v0'
                                               else env.selected_indices, dtype=np.float64))
        if self.env_id == 'Pinch-v0':
            env._chamfer_dist=None
            state.update({key:np.asarray(env.info[key]).copy() for key in
                          ('goal_depths','goal_rgbs','goal_cam_pos','goal_cam_rot','goal_cam_intrinsic')})
        if self.env_id == 'Write-v0':
            # Reset's legacy API does not initialize this derived cache. Force
            # a fresh raster read for telemetry, without changing physical state.
            env._iou=None
            env._compute_iou()
            state.update(goal_height_mm=env.goal_image.numpy(),current_height_mm=env.current_image.numpy())
        return state

    def description(self):
        import sapien.core as sapien
        env = self.env
        n = env.n_particles
        struct = env.mpm_model.struct
        material = {k: getattr(struct, k).numpy()[:n] for k in (
            "particle_mass", "particle_vol", "particle_mu_lam_ys",
            "particle_friction_cohesion", "particle_type")}
        parameters = {}
        for key in ("dx", "inv_dx", "grid_dim_x", "grid_dim_y", "grid_dim_z",
                    "particle_radius", "body_ke", "body_kd", "body_mu", "body_ka",
                    "body_sticky", "ground_sticky", "static_ke", "static_kd", "static_mu", "static_ka"):
            if hasattr(struct, key):
                parameters[key] = json_value(getattr(struct, key))
        actors = [a[0] if isinstance(a, (list, tuple)) else a for a in env._get_coupling_actors()]
        derived = [i for i, a in enumerate(actors) if isinstance(a, sapien.Link)]
        contract = {"version": 1, "derived_rigid_indices": derived}
        if self.env_id in ('Fill-v0', 'Excavate-v0', 'Hang-v0', 'Pour-v0', 'Write-v0', 'Pinch-v0'):
            physical = self._physical_actors()
            contract = dict(version=2, derived_rigid_indices=derived, root_kind='fixed',
                            scene_actor_names=[a.name for a in physical] if self.env_id not in ('Excavate-v0', 'Write-v0')
                                               else ['ground', 'wall_0', 'wall_1', 'wall_2', 'wall_3'],
                            scene_actor_types=[a.type for a in physical])
        return {"material_sha256": digest_arrays(material),
                "additional_state_fields": ["vol"] if "vol" in env.get_mpm_state() else [],
                "initial_state_contract": contract,
                "controller_state": json_value(env.agent.controller.get_state()),
                "control_mode": self.control_mode,
                "control_dt": float(env.control_timestep),
                "rigid_dt": float(env._scene.get_timestep()),
                "mpm_dt": float(env._mpm_dt),
                "mpm_substeps": int(env._mpm_step_per_sapien_step),
                "parameters": parameters}, material

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        if info.get("crashed") or self.env.sim_crashed:
            raise RuntimeError("MPM simulation crashed")
        return {"reward": json_value(reward), "terminated": bool(terminated),
                "truncated": bool(truncated), **json_value(info)}

    def metrics(self):
        return json_value(self.env.evaluate())

    def close(self):
        self.env.close()


def capture(*, source: Path, output: Path, role: str, env_id: str, seed: int,
            steps: int, control_mode="pd_joint_delta_pos", env_kwargs=None,
            reset_kwargs=None, replay: Path | None = None, actions_path: Path | None = None,
            reference_initial_state_path: Path | None = None, record_mpm_wrenches=False):
    if reference_initial_state_path is not None and (role != 'reference' or replay is not None):
        raise ValueError('Native demonstration initialization is reference-only and cannot replace fixture replay')
    if env_id not in ENVIRONMENTS:
        raise ValueError(f"Unknown reference environment: {env_id}")
    if not doctor()["cuda_reference_ready"]:
        raise RuntimeError("Legacy-compatible MPM capture requires a Linux NVIDIA runner")
    env_kwargs, reset_kwargs = env_kwargs or {}, reset_kwargs or {}
    replay_data = load_fixture(replay) if replay is not None else None
    baseline = replay_data[0] if replay_data else None
    telemetry_fields = [WRENCH_FIELD] if record_mpm_wrenches else []
    if baseline:
        f = baseline["fixture"]
        env_id, seed, control_mode = f["env_id"], f["seed"], f["control_mode"]
        env_kwargs, reset_kwargs = f["env_kwargs"], f["reset_kwargs"]
        steps = baseline["steps"]
        telemetry_fields = f.get('telemetry_fields', [])
    validate_fields(telemetry_fields, env_id)
    if steps < 1:
        raise ValueError("steps must be positive")
    record_provenance = provenance(source, role)
    if role == "reference":
        import mani_skill2
        actual_source = Path(mani_skill2.__file__).resolve().parents[1]
        if actual_source != source.resolve():
            raise RuntimeError("Imported ManiSkill 2 does not match the selected checkout")
        adapter_cls = LegacyAdapter
    else:
        # The fork must implement this adapter. Its absence is a failed port gate.
        module = importlib.import_module("mani_skill.envs.softbody.capture")
        actual_source = Path(module.__file__).resolve().parents[3]
        if actual_source != source.resolve():
            raise RuntimeError("Imported candidate does not match the selected checkout")
        adapter_cls = module.CaptureAdapter
    adapter = adapter_cls(env_id, control_mode=control_mode, env_kwargs=env_kwargs)
    writer = None
    started = time.monotonic()
    try:
        initial = None
        if baseline:
            initial = (replay_data[1], baseline["fixture"], replay_data[3])
        extra_reset = {}
        if reference_initial_state_path is not None:
            extra_reset['native_initial_state'] = load_native_demo_initialization(reference_initial_state_path)
            record_provenance['reference_initial_state_sha256'] = sha256(reference_initial_state_path)
        state = adapter.reset(seed=seed, reset_kwargs=reset_kwargs, replay=initial, **extra_reset)
        state = snapshot_with_telemetry(state, adapter, role, telemetry_fields)
        description, material = adapter.description()
        description['material_sha256'] = digest_arrays(material)
        fixture = {"env_id": env_id, "seed": seed, "env_kwargs": env_kwargs,
                   "reset_kwargs": reset_kwargs, **description}
        if telemetry_fields:
            fixture['telemetry_fields'] = telemetry_fields
        fixture["initial_numeric_sha256"] = digest_arrays(initial_numeric_state(state, fixture))
        if baseline:
            actions = replay_data[2]
        elif actions_path:
            actions = np.load(actions_path, allow_pickle=False)
            if actions.shape[0] != steps:
                raise ValueError("Action count differs from requested steps")
        else:
            space = getattr(adapter.env, 'single_action_space', adapter.env.action_space)
            actions = np.zeros((steps, *space.shape), dtype=np.float32)
        if not np.isfinite(actions).all():
            raise ValueError("Non-finite actions")
        writer = TraceWriter(output, fixture=fixture, provenance=record_provenance, steps=steps)
        np.savez_compressed(output / "material.npz", **material)
        writer.manifest["material_file_sha256"] = sha256(output / "material.npz")
        writer.frame(time_s=0., state=state, metrics=adapter.metrics())
        if baseline and digest_json(fixture) != baseline["fixture_sha256"]:
            raise InvalidArtifact("Reset did not reproduce the reference fixture")
        for i, action in enumerate(actions):
            metrics = adapter.step(action)
            writer.frame(time_s=(i + 1) * fixture["control_dt"],
                         state=snapshot_with_telemetry(adapter.snapshot(), adapter, role, telemetry_fields),
                         metrics=metrics, action=action)
            print(f"captured {env_id} {i + 1}/{steps}", flush=True)
            if metrics.get("truncated") and i + 1 < steps:
                raise RuntimeError("Environment truncated before requested horizon")
        writer.manifest["wall_time_s"] = time.monotonic() - started
        writer.finish()
    except BaseException as exc:
        if writer:
            writer.finish(error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        adapter.close()
    return output


def load_native_demo_initialization(path):
    """Read one bounded numeric initialization from an external demonstration."""
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 64 * 1024**2:
        raise InvalidArtifact('Invalid demonstration initialization file')
    values = np.load(path, allow_pickle=False, mmap_mode='r')
    if not isinstance(values, np.ndarray) or values.ndim != 1 or values.dtype.kind not in 'fiu' or values.nbytes > 64 * 1024**2 or not np.isfinite(values).all():
        raise InvalidArtifact('Invalid numeric demonstration initialization')
    return np.array(values, copy=True)

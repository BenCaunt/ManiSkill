"""Record real controller and checkpoint lifecycle behavior on CPU/GPU tasks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import sapien
from scipy.spatial.transform import Rotation
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(value, prefix=''):
    result = {}
    for key, child in value.items():
        name = prefix + key
        if isinstance(child, dict):
            result.update(flatten(child, name + '/'))
        else:
            result[name] = child.detach().cpu().numpy().copy() if isinstance(child, torch.Tensor) else np.asarray(child).copy()
    return result


def action_for(env, mode):
    arm = env.agent.controller.controllers['arm']
    q = env.agent.robot.get_qpos()[0, :7].detach().cpu().numpy()
    ee = arm.ee_pose_at_base.raw_pose[0].detach().cpu().numpy() if 'ee' in mode else None
    if mode == 'pd_joint_pos':
        action = q + np.array([.002, -.003, .001, -.002, .001, .002, -.001])
    elif mode == 'pd_joint_pos_vel':
        action = np.r_[q + .002, np.full(7, .02)]
    elif mode == 'pd_joint_delta_pos_vel':
        action = np.r_[np.full(7, .01), np.full(7, .02)]
    elif mode == 'pd_ee_pose':
        action = np.r_[ee[:3] + [.001, -.002, .001],
            Rotation.from_quat(ee[[4, 5, 6, 3]]).as_rotvec() + [.002, -.001, .003]]
    elif 'ee' in mode:
        action = np.array([.01, -.02, .01, .03, -.02, .01])[:arm.single_action_space.shape[0]]
    else:
        action = np.full(7, .01)
    if 'gripper' in env.agent.controller.controllers:
        action = np.r_[action, 0.]
    return action.astype(np.float32), ee


def run(args):
    report = dict(case=json.loads(args.request.read_text())['cases'][args.case_index],
                  source_sha256=digest(Path(__file__)), complete=False, files={}, steps=[], replays=[],
                  privileged_state=True,
                  scope='Controller/IK and reset lifecycle; not full task success, reference parity or batching')
    case = report['case']
    adapter = None
    def save(name, arrays):
        if any(a.dtype.kind not in 'fiub' or not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError('Invalid numerical state: ' + name)
        path = args.output / (name + '.npz')
        np.savez_compressed(path, **arrays)
        report['files'][path.name] = digest(path)
    try:
        if case['backend'] == 'gpu':
            sapien.physx.enable_gpu()
        sys.path.insert(0, str(args.source))
        from mani_skill.envs.softbody.capture import CaptureAdapter
        kwargs = dict(sim_backend='physx_cuda' if case['backend'] == 'gpu' else 'physx_cpu', render_backend='gpu')
        options = {}
        if case['task'] in ('Write', 'Pinch'):
            kwargs['level_dir'] = '/levels'
            options['level_file'] = 'line.h5' if case['task'] == 'Write' else 'squeeze_y.h5'
        adapter = CaptureAdapter(case['task'] + '-v0', control_mode=case['control_mode'], env_kwargs=kwargs)
        env = adapter.env
        initial = adapter.reset(seed=101, reset_kwargs=dict(options=options))
        save('initial', {k: v for k, v in initial.items() if k != 'sim_state'})
        action, ee = action_for(env, case['control_mode'])
        report.update(action=action.tolist(), initial_ee_pose_at_base=ee.tolist() if ee is not None else None,
            physics_system=type(env.scene.px).__name__, control_mode=env.agent.control_mode,
            rigid_dt=float(env.scene.px.timestep), control_dt=float(env.control_timestep),
            mpm_substeps=env.mpm_coupler.substeps)
        def step(label):
            outcome = adapter.step(action)
            arm = env.agent.controller.controllers['arm']
            report['steps'].append(dict(label=label, outcome=outcome,
                ik_success=bool(arm.last_ik_success) if 'ee' in case['control_mode'] else None))
            save(label, {k: v for k, v in adapter.snapshot().items() if k != 'sim_state'})
        step('warmup-1'); step('warmup-2')
        checkpoint, flat = env.get_state_dict(), env.get_state().clone()
        save('checkpoint', flatten(checkpoint)); save('checkpoint-flat', {'flat': flat.detach().cpu().numpy()})
        saved_count = int(env.mpm_coupler.model.struct.n_particles)
        step('expected')
        for kind in ('dict', 'flat', 'reconfigure'):
            # Establish a different reset recipe before requesting restoration.
            env.reset(seed=17, options=options)
            fresh_count = int(env.mpm_coupler.model.struct.n_particles)
            old_system = env.scene.px
            env.reset(seed=17, options={**options, 'reconfigure': kind == 'reconfigure',
                'reset_to_env_states': dict(env_states=flat if kind == 'flat' else checkpoint)})
            rebuilt = old_system is not env.scene.px
            del old_system
            save(kind + '-checkpoint', flatten(env.get_state_dict()))
            save(kind + '-before', {k: v for k, v in adapter.snapshot().items() if k != 'sim_state'})
            step(kind + '-after')
            report['replays'].append(dict(kind=kind, fresh_particles=fresh_count, saved_particles=saved_count,
                restored_particles=int(env.mpm_coupler.model.struct.n_particles), rebuilt_physics=rebuilt,
                world_steps=env.mpm_gpu_world.steps if env.gpu_sim_enabled else None))
        try:
            env.set_state_dict(checkpoint)
        except RuntimeError:
            report['reset_guard'] = True
        else:
            raise AssertionError('Physical state assignment outside reset was accepted')
        report['complete'] = True
    except BaseException as exc:
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        (args.output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k not in ('files', 'steps')}, indent=2), flush=True)
        if adapter is not None:
            adapter.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--case-index', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())

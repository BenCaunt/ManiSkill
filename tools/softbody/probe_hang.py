"""Record native Hang dynamics, its recorded grasp, and checkpoint replay."""
import argparse
import json
from pathlib import Path
import sys
import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mani_skill.envs.softbody.hang import HangEnv
from mani_skill.envs.softbody.mpm import wp


def snapshot(env):
    model = env.mpm_coupler.model
    robot = env.agent.robot._objs[0]
    bodies = [env.agent.robot.links_map[r['name']]._objs[0]
              for r in env.reference_pack['robot_parameters']['links']] + [env.rod_body]
    return {**env.mpm_coupler.particle_state(),
        'mass': model.struct.particle_mass.numpy()[:model.struct.n_particles].copy(),
        'qpos': robot.qpos.copy(), 'qvel': robot.qvel.copy(), 'qacc': robot.qacc.copy(),
        'drive_position': np.array([j.drive_target for j in robot.active_joints], dtype=np.float64).reshape(-1),
        'drive_velocity': np.array([j.drive_velocity_target for j in robot.active_joints], dtype=np.float64).reshape(-1),
        'rigid_pose': np.array([np.r_[b.entity_pose.p, b.entity_pose.q] for b in bodies]),
        'rigid_velocity': np.array([np.r_[b.linear_velocity, b.angular_velocity] for b in bodies])}


def run(args):
    wp.config.kernel_cache_dir = str(args.output.parent/'warp-cache')
    env = HangEnv(obs_mode='state_dict', control_mode='pd_joint_delta_pos', render_backend='gpu')
    try:
        obs, _ = env.reset(seed=args.seed)
        initial = snapshot(env)
        np.savez_compressed(args.output/'initial-diagnostic.npz', **initial,
                            requested_qacc=env.rope_starts['robot_qacc'][env.rope_start_index])
        assert len(initial['x']) == 3636
        assert abs(float(initial['mass'].sum()) - .0698112) < 1e-7
        index = env.rope_start_index
        for key in ('x','v','F','C','vc'):
            assert np.array_equal(initial[key], env.rope_starts['mpm_'+key][index]), key
        assert np.array_equal(initial['qpos'], env.rope_starts['robot_qpos'][index])
        assert np.array_equal(initial['qvel'], env.rope_starts['robot_qvel'][index])
        assert obs['extra']['target'].shape == (1, 7)
        def picture(name):
            value = env.render_rgb_array().cpu().numpy()
            imageio.imwrite(args.output/name, value[0] if value.ndim == 4 else value)
        picture('initial.png')
        np.savez_compressed(args.output/'000000.npz', **initial)
        records = []
        for i in range(args.steps):
            action = np.zeros(8, dtype=np.float32)
            _, reward, terminated, truncated, info = env.step(action)
            state = snapshot(env)
            if not all(np.isfinite(v).all() for v in state.values()) or np.max(abs(state['qvel'])) > 100:
                raise RuntimeError('Non-finite or unstable Hang dynamics')
            np.savez_compressed(args.output/f'{i+1:06}.npz', **state)
            records.append(dict(step=i+1, reward=float(reward[0]), success=bool(info['success'][0])))
        picture('final.png')
        checkpoint = env.get_state_dict()
        flat = env.get_state().clone()
        action = np.array([.01, -.02, .01, .02, 0., -.01, 0., -.5], dtype=np.float32)
        env.step(action)
        expected = snapshot(env)
        trials = []
        for kind, state in [('dict', checkpoint), ('flat', flat)]:
            env.reset(seed=args.seed, options={'reset_to_env_states': {'env_states': state}})
            restored = env.get_state_dict()
            for key in ('mpm','mpm_material','mpm_drives','task'):
                assert all(torch.equal(restored[key][k], v) for k,v in checkpoint[key].items()), f'{kind}: {key}'
            env.step(action)
            actual = snapshot(env)
            assert np.max(abs(actual['qpos']-expected['qpos'])) < 1e-5, f'{kind}: joint replay drift'
            trials.append(dict(kind=kind, particle_replay_max_abs_m=float(np.max(abs(actual['x']-expected['x']))),
                               qpos_replay_max_abs=float(np.max(abs(actual['qpos']-expected['qpos'])))))
        return dict(completed=True, parity_validated=False, env_id='Hang-v0', seed=args.seed,
            steps=args.steps, rope_start_index=index, selected_indices=env.selected_indices.tolist(),
            particle_count=len(initial['x']), particle_mass_kg=float(initial['mass'].sum()),
            recorded_particle_and_joint_state_exact=True,
            initial_qacc_readback=initial['qacc'].tolist(), recorded_qacc=env.rope_starts['robot_qacc'][index].tolist(),
            records=records, checkpoint_trials=trials)
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'completed': False, 'parity_validated': False}
    try:
        report = run(args)
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))

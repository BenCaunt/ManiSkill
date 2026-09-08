"""Verify target-controller checkpoints and actual particle camera pixels."""
import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mani_skill.envs.softbody.fill import FillEnv
from mani_skill.envs.softbody.excavate import ExcavateEnv
from mani_skill.envs.softbody.mpm import wp


def run(output, env_id='Fill-v0'):
    wp.config.kernel_cache_dir = str(output.parent / 'warp-cache')
    env_type = {'Fill-v0': FillEnv, 'Excavate-v0': ExcavateEnv}[env_id]
    env = env_type(control_mode='pd_joint_target_delta_pos', obs_mode='state_dict', render_backend='gpu')
    try:
        obs, _ = env.reset(seed=101)
        assert obs['extra']['tcp_pose'].shape == (1, 7)
        assert obs['extra']['target'].shape == (1, 2 if env_id == 'Fill-v0' else 1)
        before = env.get_state_dict()
        env.render_rgb_array()
        camera = env.scene.human_render_cameras['render_camera']
        pixels = {k: v.cpu().numpy() for k, v in camera.get_obs().items()}
        ids = [entity.per_scene_id for entity in env._particle_entities]
        mask = np.isin(pixels['segmentation'], ids)
        count = int(np.count_nonzero(mask))
        assert count > 10, 'Particles missing from actual segmentation pixels'
        depths = pixels['depth'][mask]
        assert np.all(depths > 0) and np.isfinite(depths).all()
        np.savez_compressed(output / 'camera.npz', **pixels)
        imageio.imwrite(output / 'camera.png', pixels['rgb'][0])
        assert torch.equal(before['mpm']['x'], env.get_state_dict()['mpm']['x'])
        assert all(torch.equal(v, env.get_state_dict()['articulations'][k])
                   for k, v in before['articulations'].items()), 'Rendering advanced robot physics'
        first = np.array([.05, -.1, .03, .07, -.02, .08, -.04], dtype=np.float32)
        second = -first * .7
        env.step(first)
        checkpoint = env.get_state_dict()
        flat = env.get_state().clone()
        target = checkpoint['controller']['arm']['target_qpos'].clone()
        assert not torch.equal(target, env.agent.robot.qpos), 'Probe did not exercise controller memory'
        env.step(second)
        expected = env.get_state_dict()
        expected_qpos = env.agent.robot.qpos.clone()
        assert torch.equal(checkpoint['controller']['arm']['target_qpos'], target), 'Checkpoint aliases mutable controller state'
        trials = []
        for kind, state, reconfigure in [('dict', checkpoint, False), ('flat', flat, False), ('reconfigure', checkpoint, True)]:
            obs, _ = env.reset(seed=101, options={'reconfigure': reconfigure,
                        **({'target_num': 601} if env_id == 'Excavate-v0' else {}),
                        'reset_to_env_states': {'env_states': state}})
            restored = env.get_state_dict()
            for key in ('mpm', 'mpm_material', 'mpm_drives', 'task'):
                assert all(torch.equal(restored[key][k], v) for k, v in checkpoint[key].items()), f'{kind}: {key} restore mismatch'
            assert torch.equal(env.agent.controller.get_state()['arm']['target_qpos'], target)
            assert torch.equal(obs['agent']['controller']['arm']['target_qpos'], target)
            env.step(second)
            actual = env.get_state_dict()
            qerror = float(torch.max(abs(env.agent.robot.qpos - expected_qpos)))
            xerror = float(torch.max(abs(actual['mpm']['x'] - expected['mpm']['x'])))
            assert torch.equal(actual['mpm_drives']['position'], expected['mpm_drives']['position'])
            assert qerror < 1e-5, f'{kind}: joint replay error {qerror}'
            trials.append(dict(kind=kind, qpos_replay_max_abs=qerror, particle_replay_max_abs_m=xerror))
        invalid = env.get_state_dict()
        invalid['controller']['arm']['target_qpos'][0, 0] = float('nan')
        rejected = False
        try:
            env.reset(seed=101, options={'reset_to_env_states': {'env_states': invalid}})
        except ValueError:
            rejected = True
        assert rejected, 'Non-finite controller state was accepted'
        return dict(passed=True, env_id=env_id, scope='checkpoint and camera lifecycle, not reference parity',
                    particle_pixels=count, particle_depth_range_mm=[int(depths.min()), int(depths.max())],
                    target_memory_exercised=True, invalid_controller_rejected=rejected, trials=trials)
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--env-id', default='Fill-v0', choices=['Fill-v0', 'Excavate-v0'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'passed': False}
    try:
        report = run(args.output, args.env_id)
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))

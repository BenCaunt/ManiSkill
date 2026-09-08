"""Record native particle pixels and physical state across GPU rendering/reset."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import imageio.v2 as imageio
import numpy as np
import sapien
import torch


def flatten(value, prefix=''):
    result = {}
    for key, child in value.items():
        name = prefix + key
        if isinstance(child, dict):
            result.update(flatten(child, name + '/'))
        else:
            result[name] = child.detach().cpu().numpy().copy() if isinstance(child, torch.Tensor) else np.asarray(child).copy()
    return result


def run(args):
    case = json.loads(args.request.read_text())['cases'][args.case_index]
    report = dict(case=case, complete=False, frames=[],
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Actual RGB/depth/segmentation geometry and render/reset lifecycle; not manipulation success or reference dynamics parity')
    env = None
    try:
        if case['backend'] == 'gpu':
            sapien.physx.enable_gpu()
        sys.path.insert(0, str(args.source))
        from mani_skill.envs.softbody.fill import FillEnv
        from mani_skill.envs.softbody.write import WriteEnv
        task = dict(Fill=FillEnv, Write=WriteEnv)[case['task']]
        kwargs = dict(sim_backend='physx_cuda' if case['backend'] == 'gpu' else 'physx_cpu',
            render_backend='gpu', obs_mode=case.get('obs_mode', 'state_dict'), control_mode=case['control_mode'])
        options = {}
        if case['task'] == 'Write':
            kwargs['level_dir'] = '/levels'; options['level_file'] = 'line.h5'
        env = task(**kwargs)
        env.reset(seed=101, options=options)
        report.update(physics_system=type(env.scene.px).__name__, particle_radius=float(env.mpm_coupler.model.struct.particle_radius))
        def rigid_visual_bounds():
            from mani_skill.envs.softbody.geometry import visual_meshes
            ids, bounds, poses = [], [], []
            for link in env.agent.robot._objs[0].links:
                component = link.entity.find_component_by_type(sapien.render.RenderBodyComponent)
                if component is None or not component.render_shapes:
                    continue
                vertices = np.concatenate([mesh.vertices for mesh in visual_meshes(link)])
                pose = env.rigid_pose(link)
                ids.append(link.entity.per_scene_id)
                bounds.append([vertices.min(axis=0), vertices.max(axis=0)])
                poses.append(pose.to_transformation_matrix())
            return dict(rigid_visual_ids=np.asarray(ids), rigid_visual_bounds=np.asarray(bounds),
                        rigid_visual_poses=np.asarray(poses))
        def frame(label):
            bounds = rigid_visual_bounds() if case.get('rigid_bounds') else {}
            before = flatten(env.get_state_dict())
            native_before = env.scene.px.cuda_rigid_body_data.torch().cpu().numpy().copy() if env.gpu_sim_enabled else np.zeros(0)
            start = time.perf_counter()
            observation_pixels = {}
            if case.get('camera') == 'base_camera':
                observation = env.get_obs()['sensor_data']['base_camera']
                observation_pixels = {'observation_' + k: v.detach().cpu().numpy().copy() for k, v in observation.items()}
                camera = env.scene.sensors['base_camera']
            else:
                env.render_rgb_array()
                camera = env.scene.human_render_cameras['render_camera']
            pixels = {k: v.detach().cpu().numpy().copy() for k, v in camera.get_obs().items()}
            parameters = {k: v.detach().cpu().numpy().copy() for k, v in camera.get_params().items()}
            elapsed = time.perf_counter() - start
            after = flatten(env.get_state_dict())
            native_after = env.scene.px.cuda_rigid_body_data.torch().cpu().numpy().copy() if env.gpu_sim_enabled else np.zeros(0)
            arrays = {**pixels, **parameters, **bounds, **observation_pixels,
                'before_native_rigid': native_before, 'after_native_rigid': native_after,
                'particle_ids': np.array([entity.per_scene_id for entity in env._particle_entities], dtype=np.int64),
                'pool_ids': np.array([entity.per_scene_id for entity in env._particle_visual_pool], dtype=np.int64),
                'particle_x': env.mpm_coupler.particle_state()['x'],
                **{'before/' + k: v for k, v in before.items()}, **{'after/' + k: v for k, v in after.items()}}
            if any(not np.isfinite(value).all() for value in arrays.values()):
                raise ValueError('Nonfinite rendering record')
            path = args.output / (label + '.npz'); np.savez_compressed(path, **arrays)
            imageio.imwrite(args.output / (label + '.png'), pixels['rgb'][0])
            report['frames'].append(dict(label=label, file=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), render_seconds=elapsed,
                shader_pack=camera._shader_config.shader_pack, position_units='mm'))
            if case.get('repeat_frames') and not label.endswith('-repeat'):
                frame(label + '-repeat')
        frame('initial')
        action = np.array([.02, -.04, .01, .03, -.01, .03, -.02], dtype=np.float32)
        report['action'] = action.tolist()
        for _ in range(2):
            env.step(action)
        checkpoint = env.get_state_dict()
        for _ in range(3):
            env.step(action)
        frame('stepped')
        if case.get('resize'):
            # Exercise particle-count restore, retaining each particle's mass
            # and constitutive inputs. This is a lifecycle diagnostic, not a
            # changed benchmark episode or a manipulation-success attempt.
            reduced = {key: ({k: v[:, ::2].clone() for k, v in group.items()}
                       if key in ('mpm', 'mpm_material') else group) for key, group in checkpoint.items()}
            env.reset(seed=17, options={**options, 'reset_to_env_states': dict(env_states=reduced)})
            frame('shrunk')
            env.reset(seed=17, options={**options, 'reset_to_env_states': dict(env_states=checkpoint)})
            frame('regrown')
            expanded = {key: ({k: torch.cat((v, v[:, ::2]), dim=1) for k, v in group.items()}
                        if key in ('mpm', 'mpm_material') else group) for key, group in checkpoint.items()}
            original_count = checkpoint['mpm']['x'].shape[1]
            expanded['mpm']['x'][:, original_count:] += torch.tensor([.06, 0., 0.], device=env.device)
            env.reset(seed=17, options={**options, 'reset_to_env_states': dict(env_states=expanded)})
            frame('expanded')
        else:
            env.reset(seed=17, options={**options, 'reset_to_env_states': dict(env_states=checkpoint)})
            frame('restored')
        env.reset(seed=101, options={**options, 'reconfigure': True})
        frame('reconfigured')
        report['complete'] = True
    except BaseException as exc:
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        (args.output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report, indent=2), flush=True)
        if env is not None:
            env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--case-index', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())

"""Record real shared-world Fill tasks and partial-reset isolation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
import sapien
import torch


def flatten(value, prefix=''):
    result = {}
    for key, child in value.items():
        if isinstance(child, dict):
            result.update(flatten(child, prefix + key + '/'))
        else:
            result[prefix + key] = child.detach().cpu().numpy().copy() if isinstance(child, torch.Tensor) else np.asarray(child).copy()
    return result


def take_rows(value, indices):
    return {k: take_rows(v, indices) for k, v in value.items()} if isinstance(value, dict) else value[indices].clone()


def controller_actions(mode, qpos, scales):
    qpos = qpos.astype(np.float64)
    actions = []
    for q, scale in zip(qpos, scales):
        if mode == 'pd_joint_pos':
            value = q + scale * np.array([.002, -.003, .001, -.002, .001, .002, -.001])
        elif mode == 'pd_joint_pos_vel':
            value = np.r_[q + scale * .002, np.full(7, scale * .02)]
        elif mode == 'pd_joint_delta_pos_vel':
            value = np.r_[np.full(7, scale * .01), np.full(7, scale * .02)]
        elif 'ee' in mode:
            width = 3 if mode.endswith('_pos') else 6
            value = np.array([.01, -.02, .01, .03, -.02, .01])[:width] * scale
        else:
            value = np.full(7, scale * .01)
        actions.append(value)
    return np.asarray(actions, np.float32)


def run(args):
    case = json.loads(args.request.read_text())['cases'][args.case_index]
    report = dict(case=case, complete=False, files={}, steps=[], renders=[],
                  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    env = None
    try:
        sapien.physx.enable_gpu()
        sys.path.insert(0, str(args.source))
        from mani_skill.envs.softbody.fill import FillEnv
        from mani_skill.envs.softbody.geometry import visual_meshes
        count = len(case['seeds'])
        offsets = case.get('scene_offsets')
        if offsets is not None:
            offsets = np.asarray(offsets, dtype=np.float32)
            if offsets.shape != (count, 3) or not np.isfinite(offsets).all():
                raise ValueError('Invalid diagnostic native scene offsets')
        class DiagnosticFillEnv(FillEnv):
            def _setup_scene(self):
                super()._setup_scene()
                if offsets is not None:
                    # Set before rigid objects/GPU buffers exist. This only
                    # translates the native shared-world origins; scene-local
                    # task geometry, particle inputs and physical parameters stay fixed.
                    for scene, offset in zip(self.scene.sub_scenes, offsets):
                        self.scene.px.set_scene_offset(scene, offset)
        kwargs = dict(num_envs=count, sim_backend='physx_cuda', obs_mode='state_dict', control_mode=case['control_mode'])
        if count > 1:
            kwargs['mpm_batch_particle_capacity'] = case['capacity']
        env = DiagnosticFillEnv(**kwargs)
        env.reset(seed=case['seeds'])
        original_system = env.scene.px
        arm = env.agent.controller.controllers['arm']
        ik_calls = []
        original_models = tuple(getattr(arm, 'pmodels', ()))
        if case.get('controller_lifecycle') and 'ee' in case['control_mode']:
            report['ik_models'] = dict(count=len(original_models), distinct=len({id(m) for m in original_models}) == count,
                native_ownership=all(a is b for a, b in zip(arm._native_robots, env.agent.robot._objs)),
                links=list(arm.ee_link_indices), masks=[mask.tolist() for mask in arm.qmasks])
            class RecordedIK:
                def __init__(self, model, index):
                    self.model, self.index = model, index
                def compute_inverse_kinematics(self, link, pose, **kwargs):
                    result = self.model.compute_inverse_kinematics(link, pose, **kwargs)
                    ik_calls.append(dict(index=self.index, link=link, target_pose=np.r_[pose.p, pose.q].tolist(),
                        initial_qpos=kwargs['initial_qpos'].tolist(), active_qmask=kwargs['active_qmask'].tolist(),
                        max_iterations=kwargs['max_iterations'], result=np.asarray(result[0]).tolist(), success=bool(result[1])))
                    return result
            arm.pmodels = [RecordedIK(model, i) for i, model in enumerate(original_models)]
        report.update(num_envs=env.num_envs, physics_system=type(env.scene.px).__name__,
                      scene_offsets=[env.scene.px.get_scene_offset(s).tolist() for s in env.scene.sub_scenes],
                      shared_native_system=all(s.physx_system is env.scene.px for s in env.scene.sub_scenes),
                      world_couplers=len(env.mpm_gpu_world.couplers),
                      mpm_substeps=[c.substeps for c in env.mpm_couplers],
                      rigid_dt=[c.rigid_dt for c in env.mpm_couplers],
                      model_scene_ownership=all(c.scene is s and all(b.entity.scene is s for b in c.bodies)
                                                for c, s in zip(env.mpm_couplers, env.scene.sub_scenes)))
        native_steps = 0
        native_step = env.scene.step
        def counted_step():
            nonlocal native_steps
            native_steps += 1
            native_step()
        env.scene.step = counted_step
        def save(label, arrays):
            if any(v.dtype.kind not in 'fibu' or not np.isfinite(v).all() for v in arrays.values()):
                raise ValueError('Invalid numeric probe record')
            path = args.output / (label + '.npz')
            np.savez_compressed(path, **arrays)
            report['files'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        def dump(label):
            state = flatten(env.get_state_dict(), 'state/')
            if case.get('controller_lifecycle') and 'ee' in case['control_mode']:
                current_arm = env.agent.controller.controllers['arm']
                state.update(ee_pose_at_base=current_arm.ee_pose_at_base.raw_pose.cpu().numpy().copy(),
                             ee_target_pose=current_arm._target_pose.raw_pose.cpu().numpy().copy())
            for i, coupler in enumerate(env.mpm_couplers):
                state.update({f'actual/{i}/' + k: v for k, v in coupler.particle_state().items()})
                state[f'actual/{i}/mass'] = coupler.model.struct.particle_mass.numpy()[:coupler.model.struct.n_particles].copy()
            state.update(qpos=env.agent.robot.get_qpos().cpu().numpy().copy(),
                         qvel=env.agent.robot.get_qvel().cpu().numpy().copy(),
                         drive_position=env.agent.robot.get_drive_targets().cpu().numpy().copy(),
                         drive_velocity=env.agent.robot.get_drive_velocities().cpu().numpy().copy(),
                         counts=np.array([c.model.struct.n_particles for c in env.mpm_couplers]),
                         elapsed_steps=env._elapsed_steps.cpu().numpy().copy(),
                         native_rigid=env.scene.px.cuda_rigid_body_data.torch().cpu().numpy().copy())
            save(label, state)
        def dump_model(label):
            robots = env.agent.robot._objs
            def pose(p):
                return np.r_[p.p, p.q]
            save(label, dict(robot_mass=np.array([[b.mass for b in r.links] for r in robots]),
                             robot_inertia=np.array([[b.inertia for b in r.links] for r in robots]),
                             robot_com=np.array([[pose(b.cmass_local_pose) for b in r.links] for r in robots]),
                             joint_parent_pose=np.array([[pose(j.pose_in_parent) for j in r.joints] for r in robots]),
                             joint_child_pose=np.array([[pose(j.pose_in_child) for j in r.joints] for r in robots])))
        def render(label):
            before = flatten(env.get_state_dict(), 'checkpoint/')
            before_rigid = env.scene.px.cuda_rigid_body_data.torch().cpu().numpy().copy()
            env.render_rgb_array()
            camera = env.scene.human_render_cameras['render_camera']
            pixels = {k: v.cpu().numpy().copy() for k, v in camera.get_obs().items()}
            params = {k: v.cpu().numpy().copy() for k, v in camera.get_params().items()}
            after = flatten(env.get_state_dict(), 'checkpoint/')
            after_rigid = env.scene.px.cuda_rigid_body_data.torch().cpu().numpy().copy()
            for i, coupler in enumerate(env.mpm_couplers):
                pool = env._mpm_batch.pools[i] if count > 1 else env._particle_visuals
                particles = coupler.particle_state()['x']
                ids, bounds, poses = [], [], []
                for link in env.agent.robot._objs[i].links:
                    component = link.entity.find_component_by_type(sapien.render.RenderBodyComponent)
                    if component is None or not component.render_shapes:
                        continue
                    vertices = np.concatenate([m.vertices for m in visual_meshes(link)])
                    ids.append(link.entity.per_scene_id); bounds.append([vertices.min(0), vertices.max(0)])
                    poses.append(env.rigid_pose(link).to_transformation_matrix())
                arrays = {**{k: v[i:i+1] for k, v in pixels.items()}, **{k: v[i:i+1] for k, v in params.items()},
                          'particle_ids': np.array([e.per_scene_id for e in pool.active]),
                          'pool_ids': np.array([e.per_scene_id for e in pool.entities]), 'particle_x': particles,
                          'rigid_visual_ids': np.array(ids), 'rigid_visual_bounds': np.array(bounds), 'rigid_visual_poses': np.array(poses),
                          'before_native_rigid': before_rigid, 'after_native_rigid': after_rigid,
                          'before/mpm/x': particles[None].copy(), 'after/mpm/x': particles[None].copy(),
                          **{'before/' + k: v for k, v in before.items()}, **{'after/' + k: v for k, v in after.items()}}
                name = f'{label}-env{i}'; save(name, arrays)
                imageio.imwrite(args.output / (name + '.png'), pixels['rgb'][i])
                report['renders'].append(dict(label=label, index=i, file=name + '.npz'))
        base_action = np.array([.02, -.04, .01, .03, -.01, .03, -.02], np.float32)
        action = np.array([base_action * scale for scale in case['action_scales']])
        if case.get('controller_lifecycle'):
            action = controller_actions(case['control_mode'], env.agent.robot.get_qpos().cpu().numpy(), case['action_scales'])
        report['action'] = action.tolist()
        def step(label):
            before = native_steps
            controller_record = {}
            if case.get('controller_lifecycle'):
                controller_record['qpos_before'] = env.agent.robot.get_qpos().cpu().tolist()
                if 'ee' in case['control_mode']:
                    controller_record.update(ee_pose_before=arm.ee_pose_at_base.raw_pose.cpu().tolist(),
                                             target_pose_before=arm._target_pose.raw_pose.cpu().tolist())
                ik_calls.clear()
            _, reward, terminated, truncated, _ = env.step(action)
            if case.get('controller_lifecycle'):
                controller_record['ik_calls'] = list(ik_calls)
                if 'ee' in case['control_mode']:
                    controller_record.update(target_pose_after=arm._target_pose.raw_pose.cpu().tolist(),
                        target_qpos_after=arm._target_qpos.cpu().tolist(),
                        ik_success=np.asarray(arm.last_ik_success.cpu() if isinstance(arm.last_ik_success, torch.Tensor)
                                               else [arm.last_ik_success]).tolist())
            report['steps'].append(dict(label=label, native_steps=native_steps-before,
                                       world_models=len(env.mpm_gpu_world.couplers),
                                       **controller_record,
                                       reward=reward.cpu().tolist(), terminated=terminated.cpu().tolist(), truncated=truncated.cpu().tolist()))
            dump(label)
        dump('initial')
        dump_model('model-initial')
        render('camera-initial')
        step('warmup-1'); step('warmup-2')
        checkpoint = env.get_state_dict(); flat = env.get_state().clone()
        save('checkpoint-flat', {'flat': flat.cpu().numpy()})
        step('expected')
        if count == 1:
            step('continued')
        else:
            env.reset(seed=[33], options=dict(env_idx=torch.tensor([0], device=env.device),
                      reset_to_env_states=dict(env_states=take_rows(checkpoint, [0]))))
            dump('partial-restored'); step('partial-stepped')
            env.reset(seed=[17], options=dict(env_idx=torch.tensor([1], device=env.device)))
            dump('partial-fresh')
            reduced = take_rows(checkpoint, [0]); old_count = int(reduced['mpm_meta']['count'][0, 0]); new_count = (old_count + 1) // 2
            for group in ('mpm', 'mpm_material'):
                for key, value in reduced[group].items():
                    target = torch.zeros_like(value); target[:, :new_count] = value[:, :old_count:2]; reduced[group][key] = target
            reduced['mpm_meta']['count'].fill_(new_count)
            reduced['mpm_meta']['mask'][:] = torch.arange(case['capacity'], device=env.device)[None] < new_count
            save('reduced-checkpoint', flatten(reduced, 'state/'))
            env.reset(seed=[34], options=dict(env_idx=torch.tensor([0], device=env.device),
                      reset_to_env_states=dict(env_states=reduced)))
            dump('partial-count'); render('camera-partial-count'); step('count-stepped')
            env.reset(seed=[31, 32], options=dict(reset_to_env_states=dict(env_states=flat)))
            dump('flat-restored'); step('flat-stepped')
        env.reset(seed=case['seeds'], options={'reconfigure': True})
        report['replaced_native_system'] = env.scene.px is not original_system
        if original_models:
            rebuilt = env.agent.controller.controllers['arm']
            report['rebuilt_ik_models'] = (len(rebuilt.pmodels) == count
                and all(all(model is not old for old in original_models) for model in rebuilt.pmodels)
                and all(a is b for a, b in zip(rebuilt._native_robots, env.agent.robot._objs)))
        dump('reconfigured'); dump_model('model-reconfigured'); render('camera-reconfigured')
        report['native_steps'] = native_steps
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
    parser.add_argument('--source', type=Path, required=True); parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--case-index', type=int, required=True); parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())

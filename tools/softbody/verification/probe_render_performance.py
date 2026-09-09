"""Record real shared-world bucket tasks and partial-reset isolation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
import sapien
import torch
from scipy.spatial.transform import Rotation


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


def visual_bounds(component):
    """Body-local bounds from actual render geometry, including native primitives.

    Capsule/cylinder axes follow pinned SAPIEN 3.0.3 render_shape.cpp (local X).
    This capture helper does not change the mesh-only MPM SDF conversion.
    """
    bounds=[]
    for shape in component.render_shapes:
        matrix=shape.local_pose.to_transformation_matrix().astype(np.float64)
        rotation=matrix[:3,:3];center=matrix[:3,3]
        if isinstance(shape,sapien.render.RenderShapeTriangleMesh):
            points=np.concatenate([part.vertices for part in shape.parts])*shape.scale
            points=points@rotation.T+center
            bounds.append(np.array([points.min(0),points.max(0)]));continue
        if isinstance(shape,sapien.render.RenderShapeBox):
            extent=abs(rotation)@np.asarray(shape.half_size)
        elif isinstance(shape,sapien.render.RenderShapeSphere):
            extent=np.full(3,shape.radius)
        elif isinstance(shape,sapien.render.RenderShapeCapsule):
            extent=abs(rotation[:,0])*shape.half_length+shape.radius
        elif isinstance(shape,sapien.render.RenderShapeCylinder):
            extent=abs(rotation[:,0])*shape.half_length+shape.radius*np.linalg.norm(rotation[:,1:],axis=1)
        else:raise NotImplementedError('Unrecognized render primitive: '+type(shape).__name__)
        bounds.append(np.array([center-extent,center+extent]))
    if not bounds:raise ValueError('No native render geometry')
    return np.array([np.min(np.array(bounds)[:,0],axis=0),np.max(np.array(bounds)[:,1],axis=0)])


def rng_snapshot(env):
    data={'rng/main_seed':env._main_seed.copy(),'rng/episode_seed':env._episode_seed.copy()}
    for name,generators in [('main',env._batched_main_rng.rngs),('episode',env._batched_episode_rng.rngs),
                            ('legacy_main',[env._main_rng])]:
        states=[rng.get_state() for rng in generators]
        if any(state[0]!='MT19937' for state in states):raise ValueError('Unexpected RNG algorithm')
        for field,column in [('words',1),('cursor',2),('has_gauss',3),('gaussian',4)]:
            data[f'rng/{name}/{field}']=np.array([state[column] for state in states])
    return data


def controller_actions(mode, qpos, scales, ee_poses=None):
    has_gripper = qpos.shape[1] == 9
    qpos = qpos[:, :7].astype(np.float64)
    actions = []
    for index, (q, scale) in enumerate(zip(qpos, scales)):
        if mode == 'pd_joint_pos':
            value = q + scale * np.array([.002, -.003, .001, -.002, .001, .002, -.001])
        elif mode == 'pd_joint_pos_vel':
            value = np.r_[q + scale * .002, np.full(7, scale * .02)]
        elif mode == 'pd_joint_delta_pos_vel':
            value = np.r_[np.full(7, scale * .01), np.full(7, scale * .02)]
        elif mode == 'pd_ee_pose':
            pose=ee_poses[index]
            value=np.r_[pose[:3]+scale*np.array([.001,-.002,.001]),
                Rotation.from_quat(pose[[4,5,6,3]]).as_rotvec()+scale*np.array([.002,-.001,.003])]
        elif 'ee' in mode:
            width = 3 if mode.endswith('_pos') else 6
            value = np.array([.01, -.02, .01, .03, -.02, .01])[:width] * scale
        else:
            value = np.full(7, scale * .01)
        actions.append(np.r_[value,0.] if has_gripper else value)
    return np.asarray(actions, np.float32)


def run(args):
    request = json.loads(args.request.read_text())
    case = request['cases'][args.case_index]
    report = dict(case=case, complete=False, files={}, steps=[], renders=[],
                  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    env = None
    try:
        sapien.physx.enable_gpu()
        sys.path.insert(0, str(args.source))
        from render_update_probe import install
        install(args)
        from mani_skill.envs.softbody.fill import FillEnv
        from mani_skill.envs.softbody.excavate import ExcavateEnv
        from mani_skill.envs.softbody.hang import HangEnv
        from mani_skill.envs.softbody.pour import PourEnv
        from mani_skill.envs.softbody.write import WriteEnv
        from mani_skill.envs.softbody.pinch import PinchEnv
        from mani_skill.envs.softbody.geometry import visual_meshes, convex_collision_meshes
        if request.get('native_actor_extension') is not None:
            import sapien303_actor_bridge as native_actor
            loaded = Path(native_actor.__file__)
            actual_sha = hashlib.sha256(loaded.read_bytes()).hexdigest()
            if actual_sha != request['native_actor_extension']['sha256']:
                raise ValueError('Loaded actor adapter differs from the frozen binary')
            report['native_actor_extension'] = dict(sha256=actual_sha, path=str(loaded), abi=native_actor.abi())
        if request.get('native_cooked_extension') is not None:
            import sapien303_cooked_bridge as native_cooked
            loaded=Path(native_cooked.__file__)
            actual_sha=hashlib.sha256(loaded.read_bytes()).hexdigest()
            if actual_sha!=request['native_cooked_extension']['sha256']:
                raise ValueError('Loaded cooked-mesh adapter differs from the frozen binary')
            report['native_cooked_extension']=dict(sha256=actual_sha,path=str(loaded),abi=native_cooked.abi())
        task_class = {'Fill': FillEnv, 'Excavate': ExcavateEnv, 'Hang': HangEnv, 'Pour': PourEnv, 'Write': WriteEnv, 'Pinch': PinchEnv}[case['task']]
        count = len(case['seeds'])
        offsets = case.get('scene_offsets')
        if offsets is not None:
            offsets = np.asarray(offsets, dtype=np.float32)
            if offsets.shape != (count, 3) or not np.isfinite(offsets).all():
                raise ValueError('Invalid diagnostic native scene offsets')
        class DiagnosticTaskEnv(task_class):
            def _setup_scene(self):
                super()._setup_scene()
                if offsets is not None:
                    # Set before rigid objects/GPU buffers exist. This only
                    # translates the native shared-world origins; scene-local
                    # task geometry, particle inputs and physical parameters stay fixed.
                    for scene, offset in zip(self.scene.sub_scenes, offsets):
                        self.scene.px.set_scene_offset(scene, offset)
        kwargs = dict(num_envs=count, sim_backend='physx_cuda', obs_mode='state_dict', control_mode=case['control_mode'])
        if 'reward_mode' in case:
            kwargs['reward_mode'] = case['reward_mode']
        if count > 1:
            kwargs['mpm_batch_particle_capacity'] = case['capacity']
        if case['task'] in ('Write','Pinch'):
            kwargs['level_dir'] = '/levels'
        if case['task']=='Pour' and request.get('native_cooked_extension') is not None:
            kwargs['bottle_collision_dir']='/cooked-pack'
        initial_options = dict(level_file=case['level_files']) if case['task'] in ('Write','Pinch') else {}
        fresh_options = dict(level_file=case['fresh_level_file']) if case['task'] in ('Write','Pinch') else {}
        env = DiagnosticTaskEnv(**kwargs)
        env.reset(seed=case['seeds'], options=initial_options)
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
                      reward_mode=env.reward_mode,
                      scene_offsets=[env.scene.px.get_scene_offset(s).tolist() for s in env.scene.sub_scenes],
                      shared_native_system=all(s.physx_system is env.scene.px for s in env.scene.sub_scenes),
                      world_couplers=len(env.mpm_gpu_world.couplers),
                      particle_radius=[float(np.float32(c.model.struct.particle_radius)) for c in env.mpm_couplers],
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
            if request.get('rng_contract') == 1:
                state.update(rng_snapshot(env))
            if case['task'] == 'Pinch':
                info=env.evaluate();state.update(flatten(info,'reported/'))
                state['reported/reward']=env.compute_dense_reward(None,None,info).cpu().numpy().copy()
                extra=env._get_obs_extra(info)
                for key in ('tcp_pose','target_rgb','target_depth','target_points'):
                    state['reported/'+key]=extra[key].cpu().numpy().copy()
                state['reported/chamfer']=np.array([env._compute_chamfer(i) for i in range(count)])
                native_tcps=[next(link for link in robot.links if link.entity.name=='panda_hand_tcp') for robot in env.agent.robot._objs]
                poses=[env.rigid_pose(body) for body in native_tcps]
                state['tcp_pose']=np.array([np.r_[p.p,p.q] for p in poses])
                state['tcp_matrix']=np.array([p.to_transformation_matrix() for p in poses])
                state['tcp_gpu_indices']=np.array([body.gpu_pose_index for body in native_tcps])
                report.setdefault('reset_level_provenance',{})[label]=dict(files=list(env.level_files),sha256=list(env.level_sha256s))
            if case['task'] == 'Write':
                info=env.evaluate();state.update(flatten(info,'reported/'))
                extra=env._get_obs_extra(info)
                state['reported/reward']=env.compute_dense_reward(None,None,info).cpu().numpy().copy()
                state['reported/goal']=extra['goal'].cpu().numpy().copy()
                state['reported/tcp_pose']=extra['tcp_pose'].cpu().numpy().copy()
                state['goal_height_mm']=np.stack([g.image.numpy() for g in env._write_goals])
                state['current_height_mm']=np.stack([g.current.numpy() for g in env._write_goals])
                native_tcps=[next(link for link in robot.links if link.entity.name=='panda_hand_tcp')
                             for robot in env.agent.robot._objs]
                poses=[env.rigid_pose(body) for body in native_tcps]
                state['tcp_pose']=np.array([np.r_[p.p,p.q] for p in poses])
                state['tcp_matrix']=np.array([p.to_transformation_matrix() for p in poses])
                state['tcp_gpu_indices']=np.array([body.gpu_pose_index for body in native_tcps])
                report.setdefault('reset_level_provenance',{})[label]=dict(
                    files=list(env.level_files),sha256=list(env.level_sha256s))
            if case['task'] == 'Pour':
                info=env.evaluate();state.update(flatten(info,'reported/'))
                state['reported/reward']=env.compute_dense_reward(None,None,info).cpu().numpy().copy()
                state['reported/target']=env._get_obs_extra(info)['target'].cpu().numpy().copy()
                state['reported/task_counts']=np.array([env._task_counts(i) for i in range(count)])
                state['reported/grasp']=np.array([env._check_grasp(i) for i in range(count)])
                state['reset_ik_attempts']=env._reset_ik_attempts.copy()
                for name,bodies in [('source',env.source_bodies),('beaker',env.beaker_bodies),('tcp',env.grasp_sites),('leftfinger',env.lfingers),('rightfinger',env.rfingers)]:
                    poses=[env.rigid_pose(body) for body in bodies]
                    state[name+'_pose']=np.array([np.r_[p.p,p.q] for p in poses])
                    state[name+'_matrix']=np.array([p.to_transformation_matrix() for p in poses])
                for name,link in [('left',env.agent.finger1_link),('right',env.agent.finger2_link)]:
                    state['contact/'+name]=env.scene.get_pairwise_contact_impulses(link,env.source_container).cpu().numpy().copy()
            if case['task'] == 'Hang':
                info = env.evaluate()
                state.update(flatten(info, 'reported/'))
                state['reported/reward'] = env.compute_dense_reward(None, None, info).cpu().numpy().copy()
                state['reported/target'] = env._get_obs_extra(info)['target'].cpu().numpy().copy()
                for name, bodies in [('hand',env.hands),('rod',env.rod_bodies),('leftfinger',env.leftfingers),('rightfinger',env.rightfingers)]:
                    poses = [env.rigid_pose(body) for body in bodies]
                    state[name+'_pose'] = np.array([np.r_[p.p,p.q] for p in poses])
                    state[name+'_matrix'] = np.array([p.to_transformation_matrix() for p in poses])
                state['recipe_grasp_index'] = (env.rope_start_indices.copy() if count>1 else np.array([env.rope_start_index]))
            if case['task'] == 'Excavate':
                info = env.evaluate()
                state.update(flatten(info, 'reported/'))
                state['reported/reward'] = env.compute_dense_reward(None, None, info).cpu().numpy().copy()
                state['reported/target'] = env._get_obs_extra(info)['target'].cpu().numpy().copy()
                state['bucket_pose'] = np.array([env.rigid_pose(b).to_transformation_matrix() for b in env.buckets])
                for i in range(count):
                    state[f'reported/inside_bucket/{i}'] = env.particles_inside_bucket(i)
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
            model = dict(robot_mass=np.array([[b.mass for b in r.links] for r in robots]),
                             robot_inertia=np.array([[b.inertia for b in r.links] for r in robots]),
                             robot_com=np.array([[pose(b.cmass_local_pose) for b in r.links] for r in robots]),
                             joint_parent_pose=np.array([[pose(j.pose_in_parent) for j in r.joints] for r in robots]),
                             joint_child_pose=np.array([[pose(j.pose_in_child) for j in r.joints] for r in robots]))
            if case['task'] in ('Excavate','Write'):
                bodies = [[wall._bodies[i] for wall in env.walls] for i in range(count)]
                model.update(wall_mass=np.array([[b.mass for b in row] for row in bodies]),
                    wall_inertia=np.array([[b.inertia for b in row] for row in bodies]),
                    wall_com=np.array([[pose(b.cmass_local_pose) for b in row] for row in bodies]),
                    wall_half_size=np.array([[b.collision_shapes[0].half_size for b in row] for row in bodies]))
            if case['task'] == 'Excavate':
                model.update(bucket_reward_hull=np.array([convex_collision_meshes(b)[0].vertices for b in env.buckets]),
                    stored_reward_hull=np.array([v[:, :3] for v in env.vertices_mats]))
            if case['task'] == 'Hang':
                model.update(rod_mass=np.array([b.mass for b in env.rod_bodies]),
                    rod_inertia=np.array([b.inertia for b in env.rod_bodies]),
                    rod_com=np.array([pose(b.cmass_local_pose) for b in env.rod_bodies]),
                    rod_half_size=np.array([b.collision_shapes[0].half_size for b in env.rod_bodies]),
                    robot_joint_limits=np.array([r.get_qlimits() for r in robots]))
            if case['task'] == 'Pour':
                for name,bodies in [('source',env.source_bodies),('beaker',env.beaker_bodies)]:
                    model[name+'_mass']=np.array([b.mass for b in bodies])
                    model[name+'_inertia']=np.array([b.inertia for b in bodies])
                    model[name+'_com']=np.array([pose(b.cmass_local_pose) for b in bodies])
                    model[name+'_shape_count']=np.array([len(b.collision_shapes) for b in bodies])
                bottle_meshes=[convex_collision_meshes(b) for b in env.source_bodies]
                for index in range(len(env.source_bodies[0].collision_shapes)):
                    meshes=[row[index] for row in bottle_meshes]
                    model[f'bottle_convex_{index}']=np.array([m.vertices for m in meshes])
                if request.get('native_cooked_extension') is not None:
                    report['bottle_cooked_pack']=dict(sha256=env.bottle_collision.manifest_sha256,
                        leaves=list(env.bottle_collision.leaves))
                    for i,body in enumerate(env.source_bodies):
                        for j,shape in enumerate(body.collision_shapes):
                            native=native_cooked.inspect(shape)
                            for key in ('vertices','planes','polygon_indices','polygon_offsets','cached_vertices','cached_triangles','mesh_aabb'):
                                model[f'cooked/{i}/{j}/'+key]=np.asarray(native[key])
                            model[f'cooked/{i}/{j}/flags']=np.array([native['gpu_compatible'],native['cached_and_native_mesh_same']],bool)
                            model[f'cooked/{i}/{j}/properties']=np.array([shape.density,shape.physical_material.static_friction,
                                shape.physical_material.dynamic_friction,shape.physical_material.restitution])
                            model[f'cooked/{i}/{j}/scale']=shape.scale
                            model[f'cooked/{i}/{j}/pose']=np.r_[shape.local_pose.p,shape.local_pose.q]
                for key in ('source_aabb','target_aabb','target_aabc'):
                    model[key]=np.repeat(getattr(env,key)[None],count,axis=0)
                model['target_radius']=np.full(count,env._target_radius)
                model['target_height']=np.full(count,env._target_height)
            save(label, model)
        def capture_render(label, *, diagnostic=False):
            before = flatten(env.get_state_dict(), 'checkpoint/')
            if request.get('rng_contract') == 1:
                before.update(rng_snapshot(env))
            before_rigid = env.scene.px.cuda_rigid_body_data.torch().cpu().numpy().copy()
            env.render_rgb_array()
            camera = env.scene.human_render_cameras['render_camera']
            pixels = {k: v.cpu().numpy().copy() for k, v in camera.get_obs().items()}
            params = {k: v.cpu().numpy().copy() for k, v in camera.get_params().items()}
            after = flatten(env.get_state_dict(), 'checkpoint/')
            if request.get('rng_contract') == 1:
                after.update(rng_snapshot(env))
            after_rigid = env.scene.px.cuda_rigid_body_data.torch().cpu().numpy().copy()
            for i, coupler in enumerate(env.mpm_couplers):
                pool = env._mpm_batch.pools[i] if count > 1 else env._particle_visuals
                particles = coupler.particle_state()['x']
                ids, bounds, poses, visibility = [], [], [], []
                visual_bodies = list(env.agent.robot._objs[i].links)
                if case['task'] == 'Pour':
                    visual_bodies += [env.source_bodies[i],env.beaker_bodies[i]]
                for link in visual_bodies:
                    component = link.entity.find_component_by_type(sapien.render.RenderBodyComponent)
                    if component is None or not component.render_shapes:
                        continue
                    ids.append(link.entity.per_scene_id); bounds.append(visual_bounds(component))
                    poses.append(env.rigid_pose(link).to_transformation_matrix())
                    visibility.append(component.visibility)
                arrays = {**{k: v[i:i+1] for k, v in pixels.items()}, **{k: v[i:i+1] for k, v in params.items()},
                          'particle_ids': np.array([e.per_scene_id for e in pool.active]),
                          'pool_ids': np.array([e.per_scene_id for e in pool.entities]), 'particle_x': particles,
                          'rigid_visual_ids': np.array(ids), 'rigid_visual_bounds': np.array(bounds), 'rigid_visual_poses': np.array(poses),
                          'rigid_visual_visibility': np.array(visibility),
                          'before_native_rigid': before_rigid, 'after_native_rigid': after_rigid,
                          'before/mpm/x': particles[None].copy(), 'after/mpm/x': particles[None].copy(),
                          **{'before/' + k: v for k, v in before.items()}, **{'after/' + k: v for k, v in after.items()}}
                if case['task'] == 'Pour':
                    ring=env._rings[i]
                    arrays.update(target_ring_id=np.array(ring.per_scene_id),
                        task_visual_ids=np.array([env.source_bodies[i].entity.per_scene_id,env.beaker_bodies[i].entity.per_scene_id]),
                        target_ring_pose=ring.pose.to_transformation_matrix(),
                        target_beaker_pose=env.rigid_pose(env.beaker_bodies[i]).to_transformation_matrix(),
                        target_ring_heights=np.array(env._height_pair(i)),target_radius=np.array(env._target_radius))
                name = f'{label}-env{i}'; save(name, arrays)
                imageio.imwrite(args.output / (name + '.png'), pixels['rgb'][i])
                collection = report.setdefault('diagnostic_renders', []) if diagnostic else report['renders']
                collection.append(dict(label=label, index=i, file=name + '.npz'))
        def render(label):
            capture_render(label)
            if request.get('pour_visibility_diagnostic') != 1 or label != 'camera-initial':
                return
            if case['task'] != 'Pour':
                raise ValueError('Pour visibility diagnostics require the Pour task')
            # These explicitly labelled inspection views alter only render-body
            # visibility. Preserve the original default-camera verdict. The
            # captured native buffer and checkpoint must remain byte-identical.
            bodies = [b for robot in env.agent.robot._objs for b in robot.links] + list(env.source_bodies)
            components = [b.entity.find_component_by_type(sapien.render.RenderBodyComponent) for b in bodies]
            components = [c for c in components if c is not None]
            saved = [(c, c.visibility) for c in components]
            try:
                env._invalidate_particle_render_groups()
                for body in env.source_bodies:
                    body.entity.find_component_by_type(sapien.render.RenderBodyComponent).visibility = 0.
                capture_render('inspection-bottle-hidden', diagnostic=True)
                env._invalidate_particle_render_groups()
                for component in components:
                    component.visibility = 0.
                capture_render('inspection-bottle-robot-hidden', diagnostic=True)
            finally:
                env._invalidate_particle_render_groups()
                for component, visibility in saved:
                    component.visibility = visibility
            capture_render('inspection-visibility-restored', diagnostic=True)
        base_action = np.array([.02, -.04, .01, .03, -.01, .03, -.02], np.float32)
        action = np.array([base_action * scale for scale in case['action_scales']])
        if case.get('controller_lifecycle'):
            ee=arm.ee_pose_at_base.raw_pose.cpu().numpy() if case['control_mode']=='pd_ee_pose' else None
            action = controller_actions(case['control_mode'], env.agent.robot.get_qpos().cpu().numpy(), case['action_scales'],ee)
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
            env.reset(seed=[33], options=dict(**fresh_options,env_idx=torch.tensor([0], device=env.device),
                      reset_to_env_states=dict(env_states=take_rows(checkpoint, [0]))))
            dump('partial-restored'); step('partial-stepped')
            env.reset(seed=[17], options=dict(**fresh_options,env_idx=torch.tensor([1], device=env.device)))
            dump('partial-fresh')
            reduced = take_rows(checkpoint, [0]); old_count = int(reduced['mpm_meta']['count'][0, 0])
            retained = np.arange(0,old_count,2)
            if case.get('preserve_task_particle_indices'):
                task_indices = reduced['task']['selected_indices'][0].cpu().numpy()
                retained = np.unique(np.r_[retained,task_indices])
                reduced['task']['selected_indices'][0] = torch.as_tensor(np.searchsorted(retained,task_indices),device=env.device)
            new_count = len(retained)
            retained_tensor = torch.as_tensor(retained,device=env.device)
            groups=['mpm','mpm_material']+(['task_particles'] if case['task']=='Pinch' else [])
            for group in groups:
                for key, value in reduced[group].items():
                    target = torch.zeros_like(value); target[:, :new_count] = value[:, retained_tensor]; reduced[group][key] = target
            reduced['mpm_meta']['count'].fill_(new_count)
            reduced['mpm_meta']['mask'][:] = torch.arange(case['capacity'], device=env.device)[None] < new_count
            save('reduced-checkpoint', flatten(reduced, 'state/'))
            env.reset(seed=[34], options=dict(**fresh_options,env_idx=torch.tensor([0], device=env.device),
                      reset_to_env_states=dict(env_states=reduced)))
            dump('partial-count'); render('camera-partial-count'); step('count-stepped')
            flat_options={}
            restored_flat=flat
            if case.get('reverse_flat_indices'):
                flat_options['env_idx']=torch.arange(count-1,-1,-1,device=env.device)
                restored_flat=flat.flip(0)
            env.reset(seed=list(range(31,31+count)), options=dict(**fresh_options,**flat_options,reset_to_env_states=dict(env_states=restored_flat)))
            dump('flat-restored'); step('flat-stepped')
        env.reset(seed=case['seeds'], options={**initial_options,'reconfigure': True})
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

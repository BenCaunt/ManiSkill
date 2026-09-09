"""Capture all-row GPU batch state, reset isolation, timings and observed memory.

This trusted diagnostic never assigns simulation state outside env.reset().
Numeric snapshots are external verifier inputs, not a task-success claim.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

from .job_archive import atomic_json, file_hash
from .scaling_contract import digest, selection, validate_case


def flatten(tree, prefix='state/'):
    result = {}
    for name, value in tree.items():
        if isinstance(value, dict):
            result.update(flatten(value, prefix+name+'/'))
        else:
            result[prefix+name] = value.detach().cpu().numpy().copy()
    return result


def rows(tree, indices):
    return {k:rows(v,indices) for k,v in tree.items()} if isinstance(tree,dict) else tree[indices].clone()


def run(args):
    import torch
    request = json.loads(args.request.read_text()); case = validate_case(request['scaling_case'])
    n = case['num_envs']; chosen = selection(n)
    record = dict(schema_version=1, complete=False, files={}, steps=[], memory=[],
        provenance=dict(role='scaling', case=case, image=os.environ['SOFTBODY_IMAGE_ID'],
            payload_sha256=os.environ['SOFTBODY_SOURCE_ARCHIVE_SHA256'],
            source_files_digest=digest(request['file_sha256']['source']),
            harness_files_digest=digest(request['file_sha256']['harness']), request_digest=digest(request)),
        selected_indices=chosen, control_mode='pd_joint_pos', timing_scope='Synchronized env.step including observations; no rendering or snapshot I/O',
        memory_scope='Observed whole-device usage at boundaries; torch peaks cover only its allocator; Linux process high-water RSS is separate')
    atomic_json(args.output/'result.json', record)
    env = None
    try:
        sys.path.insert(0,str(args.source))
        from mani_skill.envs.softbody.fill import FillEnv
        from mani_skill.envs.softbody.excavate import ExcavateEnv
        from mani_skill.envs.softbody.mpm import wp
        if request.get('native_actor_extension') is not None:
            import sapien303_actor_bridge as native_actor
            actual = file_hash(Path(native_actor.__file__))
            if actual != request['native_actor_extension']['sha256']:
                raise ValueError('Loaded native actor adapter differs')
            record['native_actor_extension_sha256'] = actual
        def sync():
            wp.synchronize(); torch.cuda.synchronize()
        def memory(label):
            sync(); free,total = torch.cuda.mem_get_info()
            record['memory'].append(dict(label=label,device_used_bytes=total-free,device_total_bytes=total,
                torch_allocated_bytes=torch.cuda.memory_allocated(),torch_reserved_bytes=torch.cuda.memory_reserved(),
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                process_high_water_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        cls = FillEnv if case['env_id']=='Fill-v0' else ExcavateEnv
        started = time.perf_counter()
        env = cls(num_envs=n,sim_backend='physx_cuda',obs_mode='state_dict',control_mode='pd_joint_pos')
        env.reset(seed=case['seeds']); sync()
        record['setup_seconds'] = time.perf_counter()-started
        native_system = env.scene.px
        record.update(actual_backend=env.backend.sim_backend,mpm_device=env.mpm_device,gpu_sim_enabled=bool(env.gpu_sim_enabled),
            actual_num_envs=env.num_envs,capacity=env._mpm_batch.capacity,
            shared_native_system=all(s.physx_system is native_system for s in env.scene.sub_scenes),
            world_models=len(env.mpm_gpu_world.couplers),
            model_scene_ownership=all(c.scene is s and all(b.entity.scene is s for b in c.bodies)
                for c,s in zip(env.mpm_couplers,env.scene.sub_scenes)),
            rigid_dt=[c.rigid_dt for c in env.mpm_couplers],mpm_substeps=[c.substeps for c in env.mpm_couplers])
        native_steps = 0
        original_step = env.scene.step
        def count_step():
            nonlocal native_steps
            native_steps += 1; original_step()
        env.scene.step = count_step
        initial_qpos = env.agent.robot.get_qpos().cpu().numpy().copy()
        scales = .04 + .01*np.arange(n, dtype=np.float64)
        action = (initial_qpos.astype(np.float64) + scales[:,None]*np.array([.002,-.003,.001,-.002,.001,.002,-.001])).astype(np.float32)
        record['action'] = action.tolist()
        action_tensor = torch.as_tensor(action,device=env.device)
        def dump(label):
            sync(); data = flatten(env.get_state_dict())
            data.update(qpos=env.agent.robot.get_qpos().cpu().numpy().copy(),qvel=env.agent.robot.get_qvel().cpu().numpy().copy(),
                drive_position=env.agent.robot.get_drive_targets().cpu().numpy().copy(),
                drive_velocity=env.agent.robot.get_drive_velocities().cpu().numpy().copy(),
                elapsed_steps=env._elapsed_steps.cpu().numpy().copy(),
                counts=np.array([c.model.struct.n_particles for c in env.mpm_couplers]),
                **{'rng/main_seed':env._main_seed.copy(),'rng/episode_seed':env._episode_seed.copy()})
            for name,generators in [('main',env._batched_main_rng.rngs),('episode',env._batched_episode_rng.rngs),('legacy_main',[env._main_rng])]:
                states = [g.get_state() for g in generators]
                if any(s[0]!='MT19937' for s in states):raise ValueError('Unexpected RNG algorithm')
                for key,column in [('words',1),('cursor',2),('has_gauss',3),('gaussian',4)]:
                    data[f'rng/{name}/{key}']=np.array([s[column] for s in states])
            robots=env.agent.robot._objs
            data.update({'model/robot_mass':np.array([[b.mass for b in robot.links] for robot in robots]),
                'model/robot_inertia':np.array([[b.inertia for b in robot.links] for robot in robots]),
                'model/robot_com':np.array([[np.r_[b.cmass_local_pose.p,b.cmass_local_pose.q] for b in robot.links] for robot in robots]),
                'model/shared_system':np.array([s.physx_system is env.scene.px for s in env.scene.sub_scenes]),
                'model/scene_ownership':np.array([c.scene is s and all(b.entity.scene is s for b in c.bodies)
                    for c,s in zip(env.mpm_couplers,env.scene.sub_scenes)])})
            raw = env.scene.px.cuda_rigid_body_data.torch().cpu().numpy().copy()
            for i,c in enumerate(env.mpm_couplers):
                data.update({f'actual/{i}/'+k:v for k,v in c.particle_state().items()})
                data[f'actual/{i}/mass']=c.model.struct.particle_mass.numpy()[:c.model.struct.n_particles].copy()
                robot_indices = np.array([b.gpu_pose_index for b in robots[i].links if b.gpu_pose_index>=0],dtype=np.int64)
                coupler_indices = np.array([b.gpu_pose_index for b in c.bodies if hasattr(b,'gpu_pose_index') and b.gpu_pose_index>=0],dtype=np.int64)
                indices = np.union1d(robot_indices,coupler_indices)
                data[f'native/{i}/indices']=indices
                data[f'native/{i}/robot_indices']=robot_indices
                data[f'native/{i}/coupler_indices']=coupler_indices
                data[f'native/{i}/rigid']=raw[indices].copy()
            if any(v.dtype.kind not in 'fibu' or not np.isfinite(v).all() for v in data.values()):
                raise ValueError('Nonfinite or nonnumeric scaling snapshot')
            target=args.output/(label+'.npz');np.savez_compressed(target,**data)
            record['files'][target.name]=file_hash(target)
            memory(label);atomic_json(args.output/'result.json',record)
            print(json.dumps(dict(snapshot=label,num_envs=n,counts=data['counts'].tolist())),flush=True)
        def step(label, timed=False):
            sync(); before=native_steps; started=time.perf_counter()
            _,reward,terminated,truncated,_=env.step(action_tensor)
            sync(); seconds=time.perf_counter()-started
            record['steps'].append(dict(label=label,timed=timed,seconds=seconds,
                native_steps=native_steps-before,world_models=len(env.mpm_gpu_world.couplers),
                reward=reward.cpu().tolist(),terminated=terminated.cpu().tolist(),truncated=truncated.cpu().tolist()))
        dump('initial')
        for i in range(2):step('warmup-'+str(i))
        dump('warmup'); checkpoint=env.get_state_dict(); flat=env.get_state().clone()
        for i in range(case['timed_controls']):step('timed-'+str(i),True)
        dump('measured')
        env.reset(seed=[700+i for i in chosen],options={'env_idx':torch.tensor(chosen,device=env.device),
            'reset_to_env_states':{'env_states':rows(checkpoint,chosen)}})
        dump('partial-restored');step('partial-step');dump('partial-stepped')
        env.reset(seed=[900+i for i in chosen],options={'env_idx':torch.tensor(chosen,device=env.device)})
        dump('partial-fresh');step('fresh-step');dump('fresh-stepped')
        reversed_indices=list(reversed(range(n)))
        env.reset(seed=[1100+i for i in reversed_indices],options={'env_idx':torch.tensor(reversed_indices,device=env.device),
            'reset_to_env_states':{'env_states':flat.flip(0)}})
        dump('flat-restored');step('flat-step');dump('flat-stepped')
        env.reset(seed=case['seeds'],options={'reconfigure':True});sync()
        record['replaced_native_system']=env.scene.px is not native_system
        dump('reconfigured')
        record['native_steps']=native_steps;record['complete']=True
    except BaseException as exc:
        record['error']=type(exc).__name__+': '+str(exc)
        raise
    finally:
        atomic_json(args.output/'result.json',record)
        print(json.dumps({k:v for k,v in record.items() if k!='files'}),flush=True)
        if env is not None:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())

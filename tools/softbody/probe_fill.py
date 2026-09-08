"""Capture native bucket tasks for independent host-side comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import imageio.v2 as imageio

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from mani_skill.envs.softbody.fill import FillEnv
from mani_skill.envs.softbody.excavate import ExcavateEnv
from mani_skill.envs.softbody.mpm import wp


def snapshot(env):
    c=env.mpm_coupler
    robot=env.agent.robot._objs[0]
    bodies=c.bodies
    return {**c.particle_state(), 'mass':c.model.struct.particle_mass.numpy()[:c.model.struct.n_particles],
            'qpos':robot.qpos.copy(),'qvel':robot.qvel.copy(),
            'drive_position':np.array([j.drive_target for j in robot.active_joints]).reshape(-1),
            'drive_velocity':np.array([j.drive_velocity_target for j in robot.active_joints]).reshape(-1),
            'rigid_pose':np.array([np.r_[b.entity_pose.p,b.entity_pose.q] for b in bodies]),
            'rigid_velocity':np.array([np.r_[b.linear_velocity,b.angular_velocity] for b in bodies]),
            'robot_root_pose':np.r_[robot.root_pose.p,robot.root_pose.q],
            'robot_root_velocity':np.r_[robot.root_linear_velocity,robot.root_angular_velocity]}


def run(args):
    wp.config.kernel_cache_dir=str(args.output.parent/'warp-cache')
    env_type = {'Fill-v0': FillEnv, 'Excavate-v0': ExcavateEnv}[args.env_id]
    env=env_type(mpm_device=args.device,obs_mode='none',reward_mode='dense',render_backend='gpu')
    records=[]
    try:
        env.reset(seed=args.seed)
        initial=snapshot(env)
        expected = {'Fill-v0': (704, .135168), 'Excavate-v0': (11056, 4.146)}
        if args.seed == 101:
            count, mass = expected[args.env_id]
            if len(initial['x']) != count or abs(float(initial['mass'].sum()) - mass) > 1e-6:
                raise RuntimeError('Particle count or mass differs from the seed-101 reference')
        plane = env.ground._bodies[0].get_collision_shapes()[0]
        normal = plane.local_pose.to_transformation_matrix()[:3, 0]
        if not np.allclose(normal, [0., 0., 1.], atol=1e-6):
            raise RuntimeError('Ground plane normal must point out of the floor (+Z)')
        def render_image():
            value=env.render_rgb_array()
            if hasattr(value,'cpu'): value=value.cpu().numpy()
            if value.ndim == 4: value=value[0]
            return value
        imageio.imwrite(args.output/'initial.png',render_image())
        records.append({'step':0,'success':bool(env.evaluate()['success'][0])})
        np.savez_compressed(args.output/'000000.npz',**snapshot(env))
        if hasattr(env, 'vertices_mat'):
            np.savez_compressed(args.output/'reward_bucket_vertices.npz', vertices=env.vertices_mat)
        for i in range(args.steps):
            _,reward,terminated,truncated,info=env.step(np.zeros(7,dtype=np.float32))
            state=snapshot(env)
            if not all(np.isfinite(v).all() for v in state.values()):
                raise RuntimeError('Non-finite task state')
            np.savez_compressed(args.output/f'{i+1:06}.npz',**state)
            # Finite values alone miss catastrophic joint explosions. These
            # broad sanity bounds do not replace the tighter parity protocol.
            if np.max(abs(state['qpos'])) > 10 or np.max(abs(state['qvel'])) > 100:
                raise RuntimeError(f'Robot state unstable at control step {i+1}')
            records.append({'step':i+1,'reward':float(reward[0]),'success':bool(info['success'][0]),
                            **{k:int(info[k][0]) for k in ('contained_particles', 'lifted_particles', 'spilled_particles') if k in info}})
        for i,geometry in enumerate(env.collision_geometry):
            shutil.copyfile(geometry['cache_file'],args.output/f'sdf-{i}.npz')
        imageio.imwrite(args.output/'final.png',render_image())
        body_data=[]
        for b in env.mpm_coupler.bodies:
            body_data.append(dict(name=b.name,mass=float(b.mass),inertia=b.inertia.tolist(),
                             cmass_local_pose=np.r_[b.cmass_local_pose.p,b.cmass_local_pose.q].tolist()))
        return dict(completed=True,parity_validated=False,env_id=args.env_id,seed=args.seed,steps=args.steps,
                    initial_particles=len(initial['x']), initial_mass_kg=float(initial['mass'].sum()),
                    target_num=getattr(env, 'target_num', None),
                    control_dt=env.control_timestep,rigid_dt=env.scene.px.timestep,mpm_dt=env.mpm_dt,
                    records=records,bodies=body_data,geometry=env.collision_geometry,
                    file_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    finally:
        env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    parser.add_argument('--env-id',default='Fill-v0',choices=['Fill-v0','Excavate-v0'])
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--seed',type=int,default=101)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report={'completed':False,'parity_validated':False}
    try:
        report=run(args)
    except BaseException as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))

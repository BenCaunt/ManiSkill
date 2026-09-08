"""Native multi-joint force-transfer experiment with fixed links and branches.

The CPU physical-wrench process never imports the candidate adapter. All robot
state assignments occur in initialization. No collision geometry or renderer is
needed: the experiment isolates articulated force transfer and current-pose use.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import sapien
import torch


def quaternion(axis, angle):
    axis = np.asarray(axis, dtype=float); axis /= np.linalg.norm(axis)
    return np.r_[np.cos(angle/2), axis*np.sin(angle/2)]


def load_adapter(path):
    spec = importlib.util.spec_from_file_location('candidate_gpu_wrenches', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def build(scene, scene_index, gpu):
    builder = scene.create_articulation_builder()
    root = builder.create_link_builder(); root.set_name('root')
    # Parent indices refer to creation order. Two active branches share hinge0;
    # tool/tip are fixed descendants. Masses/COM/inertias are authored inputs.
    recipe = [
        ('hinge0',0,'revolute_unwrapped',[.03,.01,.02],[0,0,1],.2,.4),
        ('hinge1',1,'revolute_unwrapped',[.02,.10,.01],[0,1,0],.6,.25),
        ('tool',2,'fixed',[.03,.09,.04],[1,1,0],-.3,.12),
        ('slider',1,'prismatic',[-.04,.05,-.02],[0,0,1],-.8,.18),
        ('tip',4,'fixed',[.06,.02,.01],[1,0,1],.45,.08),
    ]
    created = [root]
    for n,(name,parent,kind,p,axis,angle,mass) in enumerate(recipe):
        b = builder.create_link_builder(created[parent]); b.set_name(name); b.set_joint_name(name)
        dims = np.array([.06,.09,.11])*(1+.1*n)
        inertia = mass/12*(np.sum(dims**2)-dims**2)
        com = sapien.Pose([.006,-.009,.01], quaternion([1,2,3], .2))
        b.set_mass_and_inertia(mass, com, inertia)
        b.set_joint_properties(kind, [[-10.,10.]] if kind != 'fixed' else [],
            sapien.Pose(p,quaternion(axis,angle)),
            sapien.Pose([-.012,.008,.004],quaternion([1,1,1],-.15)),friction=0.,damping=0.)
        created.append(b)
    builder.set_initial_pose(sapien.Pose([.1,-.2,.3],quaternion([1,2,1],scene_index*.23)))
    robot = builder.build(fix_root_link=True)
    robot.sleep_threshold = 0.
    for joint in robot.active_joints:
        joint.armature = [0.]; joint.friction = 0.; joint.set_drive_properties(0.,0.)
    for body in robot.links:
        body.linear_damping = 0.; body.angular_damping = 0.
    q = {'hinge0':.35+.02*scene_index,'hinge1':-.22-.01*scene_index,'slider':.006+.001*scene_index}
    v = {'hinge0':.06,'hinge1':-.04,'slider':.002} if scene_index != 5 else dict.fromkeys(q,0.)
    q = np.array([q[j.name] for j in robot.active_joints],dtype=np.float32)
    v = np.array([v[j.name] for j in robot.active_joints],dtype=np.float32)
    if not gpu: robot.set_qpos(q); robot.set_qvel(v); robot.set_qf(np.zeros(robot.dof))
    return robot,q,v


def run(mode, adapter_path):
    gpu = mode == 'gpu-joint'
    if gpu: sapien.physx.enable_gpu()
    sapien.physx.set_scene_config(gravity=[0.,0.,0.],enable_pcm=gpu,enable_tgs=False)
    sapien.physx.set_body_config(solver_position_iterations=25,solver_velocity_iterations=1,sleep_threshold=0.)
    px = sapien.physx.PhysxGpuSystem() if gpu else sapien.physx.PhysxCpuSystem()
    px.timestep = .002
    if gpu: px.gpu_set_cuda_stream(torch.cuda.current_stream().cuda_stream)
    scenes, robots, starts = [], [], []
    for i in range(6):
        scene = sapien.Scene([px]); scenes.append(scene)
        if gpu: px.set_scene_offset(scene,[4.*i,0.,0.])
        robot,q,v = build(scene,i,gpu); robots.append(robot); starts.append((q,v))
    if gpu:
        px.gpu_init()
        for robot,(q,v) in zip(robots,starts):
            px.cuda_articulation_qpos.torch()[robot.gpu_index,:robot.dof] = torch.tensor(q,device='cuda')
            px.cuda_articulation_qvel.torch()[robot.gpu_index,:robot.dof] = torch.tensor(v,device='cuda')
        px.cuda_articulation_qf.torch().zero_()
        px.gpu_apply_articulation_qpos(); px.gpu_apply_articulation_qvel(); px.gpu_apply_articulation_qf()
        px.gpu_update_articulation_kinematics()
    adapter_module = load_adapter(adapter_path) if mode != 'cpu-force' else None
    adapter = adapter_module.GPUArticulationForces(px,robots) if gpu else None
    projectors = [adapter_module.ArticulationWrenchProjector(r) for r in robots] if mode == 'cpu-joint' else None
    actual_parameters = []
    for r in robots:
        actual_parameters.append(dict(joint_names=[j.name for j in r.active_joints],
            links=[dict(name=b.name,parent=b.parent.name if b.parent else None,mass=float(b.mass),
                        inertia=b.inertia.tolist(),com=np.r_[b.cmass_local_pose.p,b.cmass_local_pose.q].tolist(),
                        joint_type=b.joint.type,
                        joint_parent_pose=np.r_[b.joint.pose_in_parent.p,b.joint.pose_in_parent.q].tolist()) for b in r.links]))

    def observe():
        if gpu:
            px.gpu_fetch_articulation_qpos();px.gpu_fetch_articulation_qvel()
            px.gpu_fetch_articulation_link_pose();px.gpu_fetch_articulation_link_velocity()
            data=px.cuda_rigid_body_data.torch().cpu().numpy()
            q=px.cuda_articulation_qpos.torch().cpu().numpy();v=px.cuda_articulation_qvel.torch().cpu().numpy()
            return [dict(qpos=q[r.gpu_index,:r.dof].copy(),qvel=v[r.gpu_index,:r.dof].copy(),
                         links=data[[b.gpu_pose_index for b in r.links]].copy()) for r in robots]
        return [dict(qpos=r.qpos.copy(),qvel=r.qvel.copy(),links=np.array([
            np.r_[b.entity_pose.p,b.entity_pose.q,b.linear_velocity,b.angular_velocity] for b in r.links])) for r in robots]

    initial=observe();frames=[initial];applied=[];external=[];baselines=[];powers=[]
    random=np.random.default_rng(8142)
    patterns=[]
    for i,r in enumerate(robots):
        w=random.uniform(-1,1,(len(r.links),6)).astype(np.float32)
        w[:,:3]*=.025;w[:,3:]*=.3;w[0]=0
        if i==2:
            for k,b in enumerate(r.links):
                if b.name!='tip': w[k]=0
        if i==3:
            for k,b in enumerate(r.links):
                if b.name!='tool': w[k]=0
        if i>=4:w[:]=0
        patterns.append(w)
    for step in range(30):
        states=observe();wrenches=[w*(1. if step<10 else 0.) for w in patterns]
        baseline=[]
        for i,r in enumerate(robots):
            base={'hinge0':.02,'hinge1':-.01,'slider':.1}
            baseline.append(np.array([base[j.name] for j in r.active_joints],np.float32)*
                            (np.cos(step*.13) if step<10 and i!=5 else 0.))
        projected=[]
        if gpu:
            base=torch.zeros_like(px.cuda_articulation_qf.torch())
            for r,b in zip(robots,baseline):base[r.gpu_index,:r.dof]=torch.tensor(b,device='cuda')
            total=adapter.apply(wrenches,baseline_qf=base)
            projected=[(total[r.gpu_index,:r.dof]-base[r.gpu_index,:r.dof]).cpu().numpy() for r in robots]
        else:
            for i,(r,state,w,b) in enumerate(zip(robots,states,wrenches,baseline)):
                if mode=='cpu-force':
                    r.set_qf(b)
                    for body,wrench in zip(r.links,w):body.add_force_torque(wrench[3:],wrench[:3],mode='force')
                else:
                    p=projectors[i].project(state['links'][:,:7],w).numpy()
                    projected.append(p);r.set_qf(b+p)
        step_powers=[]
        for i,(state,w) in enumerate(zip(states,wrenches)):
            actual_power=float(np.sum(w[:,:3]*state['links'][:,10:13])+np.sum(w[:,3:]*state['links'][:,7:10]))
            step_powers.append([actual_power,float(np.dot(projected[i],state['qvel'])) if projected else None])
        applied.append([(b+projected[i]).tolist() if projected else None for i,b in enumerate(baseline)])
        external.append([w.tolist() for w in wrenches]);baselines.append([b.tolist() for b in baseline]);powers.append(step_powers)
        px.step();frames.append(observe())
    def lists(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,dict):return {k:lists(v) for k,v in x.items()}
        if isinstance(x,list):return [lists(v) for v in x]
        return x
    return dict(mode=mode,physics_system=type(px).__name__,dt=.002,steps=30,
                frames=lists(frames),applied_qf=applied,external_wrenches=external,baseline_qf=baselines,
                external_power=powers,parameters=actual_parameters,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                adapter_sha256=hashlib.sha256(adapter_path.read_bytes()).hexdigest() if adapter_module else None,
                fixture=dict(creator='sim-infra port work',license='Apache-2.0',units='m, kg, s',
                    collision_geometry='none',inertia_basis='Uniform box estimates with explicit COM rotation; authored diagnostic assumptions',
                    seed=8142,scenes=6,gravity=[0,0,0],enable_pcm=gpu,enable_tgs=False),
                scope='Branched multi-joint force projection from actual native link poses; no MPM or task batching claim')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['cpu-force','cpu-joint','gpu-joint'],required=True)
    parser.add_argument('--adapter',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.mode,args.adapter)
    with args.output.open('x') as f:json.dump(result,f)
    print(json.dumps({k:v for k,v in result.items() if k not in ('frames','applied_qf','external_wrenches','baseline_qf','external_power','parameters')},indent=2))

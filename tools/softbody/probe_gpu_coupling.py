"""Real MPM impacts across free, sliding, hinged, and idle independent scenes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import sapien
import torch


def run(backend, source, output):
    gpu=backend=='gpu'
    if gpu:sapien.physx.enable_gpu()
    sys.path.insert(0,str(source))
    from mani_skill.envs.softbody.mpm import MPMCoupler,MPMModelBuilder,register_collision_body,wp
    from mani_skill.envs.softbody.gpu_coupling import MPMGPUWorld
    wp.config.kernel_cache_dir='/cache/warp';wp.init()
    sapien.physx.set_scene_config(gravity=[0.,0.,0.],enable_pcm=gpu,enable_tgs=False)
    sapien.physx.set_body_config(solver_position_iterations=25,solver_velocity_iterations=1,sleep_threshold=0.)
    px=sapien.physx.PhysxGpuSystem() if gpu else None
    if gpu:px.timestep=.002;px.gpu_set_cuda_stream(torch.cuda.current_stream().cuda_stream)
    fixtures=[]
    for i,kind in enumerate(('free','slider','hinge','idle')):
        system=px if gpu else sapien.physx.PhysxCpuSystem()
        scene=sapien.Scene([system]);scene.set_timestep(.002)
        if gpu:px.set_scene_offset(scene,[4.*i,0.,0.])
        half=np.array([.006,.04,.04],np.float32);mass=.05
        inertia=mass/3*np.array([half[1]**2+half[2]**2,half[0]**2+half[2]**2,half[0]**2+half[1]**2])
        material=sapien.physx.PhysxMaterial(0.,0.,0.)
        lever=.025 if kind=='hinge' else 0.
        robot=None
        if kind in ('slider','hinge'):
            ab=scene.create_articulation_builder();root=ab.create_link_builder();root.set_name('root')
            link=ab.create_link_builder(root);link.set_name('plate');link.add_box_collision(half_size=half,material=material)
            link.set_mass_and_inertia(mass,sapien.Pose(),inertia)
            q=[np.sqrt(.5),0.,-np.sqrt(.5),0.] if kind=='hinge' else [1.,0.,0.,0.]
            link.set_joint_properties('revolute_unwrapped' if kind=='hinge' else 'prismatic',[[-10.,10.]],
                sapien.Pose(q=q),sapien.Pose([0.,-lever,0.],q),friction=0.,damping=0.)
            ab.set_initial_pose(sapien.Pose([0.,0.,.1]));robot=ab.build(fix_root_link=True)
            robot.sleep_threshold=0.;joint=robot.active_joints[0]
            joint.friction=0.;joint.armature=[0.];joint.set_drive_properties(0.,0.)
            if not gpu:robot.set_qpos([0.]);robot.set_qvel([0.]);robot.set_qf([0.])
            body=robot.links[1]
        else:
            entity=sapien.Entity();body=sapien.physx.PhysxRigidDynamicComponent()
            body.attach(sapien.physx.PhysxCollisionShapeBox(half,material));body.mass=mass;body.inertia=inertia
            body.sleep_threshold=0.;entity.add_component(body);entity.pose=sapien.Pose([0.,0.,.1]);scene.add_entity(entity)
        body.linear_damping=0.;body.angular_damping=0.
        builder=MPMModelBuilder();builder.set_mpm_domain([.6,.2,.2],grid_length=.005)
        register_collision_body(builder,body)
        builder.add_mpm_grid(pos=(-.035,lever,.1),vel=(0.,0.,0.) if kind=='idle' else (1.,0.,0.),
            dim_x=3,dim_y=3,dim_z=3,cell_x=.003,cell_y=.003,cell_z=.003,density=1000.,
            mu_lambda_ys=(1000.,1000.,10000.),friction_cohesion=(0.,0.,0.),type=0,jitter=False)
        model=builder.finalize('cuda');model.gravity=np.zeros(3,np.float32)
        model.struct.ground_normal=wp.vec3(0.,0.,1.);model.struct.particle_radius=.0015
        model.struct.body_sticky=1;model.struct.body_mu=0.
        model.grid_contact=True;model.particle_contact=True;model.adaptive_grid=False
        states=[model.state() for _ in range(5)];builder.init_model_state(model,states)
        fixtures.append(dict(scene=scene,kind=kind,body=body,robot=robot,model=model,states=states,
                             inertia=inertia.copy(),effective_inertia=float(inertia[2]+mass*lever**2),lever=lever))
    if gpu:
        px.gpu_init()
        px.cuda_articulation_qpos.torch().zero_();px.cuda_articulation_qvel.torch().zero_();px.cuda_articulation_qf.torch().zero_()
        px.gpu_apply_articulation_qpos();px.gpu_apply_articulation_qvel();px.gpu_apply_articulation_qf()
        px.gpu_update_articulation_kinematics()
        world=MPMGPUWorld(px)
        for f in fixtures:f['coupler']=world.add_model(f['scene'],f['model'],f['states'],[f['body']],mpm_dt=.0005)
    else:
        for f in fixtures:f['coupler']=MPMCoupler(f['scene'],f['model'],f['states'],[f['body']],mpm_dt=.0005)
    def snapshot():
        if gpu:
            px.gpu_fetch_rigid_dynamic_data();px.gpu_fetch_articulation_link_pose();px.gpu_fetch_articulation_link_velocity()
            px.gpu_fetch_articulation_qpos();px.gpu_fetch_articulation_qvel()
            data=px.cuda_rigid_body_data.torch().cpu().numpy()
            q=px.cuda_articulation_qpos.torch().cpu().numpy();v=px.cuda_articulation_qvel.torch().cpu().numpy()
        values=[]
        for f in fixtures:
            b=f['body'];r=f['robot'];state=f['coupler'].particle_state()
            if gpu:
                row=data[b.gpu_pose_index].copy()
                joint=np.array([q[r.gpu_index,0],v[r.gpu_index,0]]) if r else np.zeros(2)
            else:
                row=np.r_[b.entity_pose.p,b.entity_pose.q,b.linear_velocity,b.angular_velocity]
                joint=np.array([r.qpos[0],r.qvel[0]]) if r else np.zeros(2)
            values.append(dict(x=state['x'],v=state['v'],rigid=row,joint=joint))
        return values
    frames=[snapshot()];wrenches=[]
    for step in range(100):
        if gpu:details=world.step(baseline_qf=torch.zeros_like(px.cuda_articulation_qf.torch()))
        else:details=[f['coupler'].step() for f in fixtures]
        wrenches.append(np.array([d.mean_wrench_torque_force[0] for d in details]))
        frames.append(snapshot())
    arrays={key:np.array([[f[key] for f in frame] for frame in frames]) for key in ('x','v','rigid','joint')}
    arrays['wrenches']=np.array(wrenches)
    arrays['mass']=np.array([f['model'].struct.particle_mass.numpy().copy() for f in fixtures])
    np.savez_compressed(output/(backend+'.npz'),**arrays)
    report=dict(backend=backend,physics_system='PhysxGpuSystem' if gpu else 'separate PhysxCpuSystem per scene',
        steps=100,dt=.002,mpm_dt=.0005,scene_offsets=[[4.*i,0,0] for i in range(4)] if gpu else [[0,0,0]]*4,
        fixture=dict(creator='sim-infra port work',license='Apache-2.0 for authored diagnostic; MPM retains its own license',
            units='m, kg, s',mass_kg=.05,dimensions_m=(half*2).tolist(),inertia_kg_m2=fixtures[0]['inertia'].tolist(),
            inertia_basis='Uniform box estimate; same declared diagnostic recipe as probe_coupling.py',
            friction=0.,restitution=0.,material_basis='Uncalibrated diagnostic assumptions',
            kinds=[f['kind'] for f in fixtures],hinge_effective_inertia=fixtures[2]['effective_inertia']),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        arrays_sha256=hashlib.sha256((output/(backend+'.npz')).read_bytes()).hexdigest(),
        scope='Real MPM contact and native force response across four independent scenes; no task-level parity or environment batching claim')
    (output/(backend+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['cpu','gpu'],required=True);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run(args.backend,args.source,args.output)

"""Convert pinned original Pinch contact geometry and explicit levels to NPZ.

Only the reference loads native legacy state or its own SDF pickle. The native
port receives actual physical fields and independent goal data, never a future
rollout. Numeric conversion retains original model/asset licenses.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--levels',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.source=args.source.resolve()
commit='493be36121a9dd06071a57172274babe617b789f'
if subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()!=commit:
    raise ValueError('Pinned reference source required')
subprocess.run(['git','-C',str(args.source),'diff','--exit-code','HEAD','--','mani_skill2','warp_maniskill'],check=True,stdout=subprocess.DEVNULL)
for name,version in [('sapien','2.2.2'),('numpy','1.23.5')]:
    if importlib.metadata.version(name)!=version:raise ValueError(f'Requires {name}=={version}')
args.output.mkdir(parents=True,exist_ok=False)
sys.path[:0]=[str(args.source),str(args.source/'warp_maniskill')]
import numpy as np
import gymnasium as gym
import mani_skill2.envs
from mani_skill2.envs.mpm.utils import actor2meshes

env=gym.make('Pinch-v0',level_dir=str(args.levels),obs_mode='none',control_mode='pd_joint_delta_pos').unwrapped
try:
    level_paths=sorted(args.levels.glob('*.h5'))
    if not level_paths:raise ValueError('No explicit Pinch levels')
    env.reset(seed=101,options={'level_file':level_paths[0].name})
    def pose(value):return np.r_[value.p,value.q].tolist()
    robot=env.agent.robot
    links=[dict(name=b.name,mass=float(b.mass),inertia=b.inertia.tolist(),com=pose(b.cmass_local_pose)) for b in robot.get_links()]
    joints=[dict(name=j.name,parent_pose=pose(j.get_pose_in_parent()),child_pose=pose(j.get_pose_in_child())) for j in robot.get_joints()]
    with open(env.sdf_cache,'rb') as stream:cache=pickle.load(stream)
    geometry=[]
    for i,(actor,sdf) in enumerate(zip(env._coupled_actors,cache['sdfs'])):
        meshes,primitives=actor2meshes(actor,return_primitives=True)
        arrays={} if sdf is None else dict(sdf)
        for j,mesh in enumerate(meshes):arrays.update({f'vertices_{j}':mesh.vertices,f'faces_{j}':mesh.faces})
        name=f'body-{i}.npz';np.savez_compressed(args.output/name,**arrays)
        geometry.append(dict(name=actor.name,file=name,mesh_count=len(meshes),has_sdf=sdf is not None,
            primitives=[dict(kind=kind,size=np.asarray(size).tolist(),pose=pose(local)) for kind,size,local in primitives],
            mass=float(actor.mass),inertia=actor.inertia.tolist(),com=pose(actor.cmass_local_pose)))
    urls=[f'https://github.com/mani-skill/ManiSkill/blob/{commit}/mani_skill2/envs/mpm/pinch_env.py',
          f'https://github.com/mani-skill/ManiSkill/blob/{commit}/mani_skill2/assets/descriptions/panda_pinch.urdf']
    report=dict(source_commit=commit,env_id='Pinch-v0',robot_parameters=dict(links=links,joints=joints),geometry=geometry,
        source_urls=urls,units='m, kg, s',frame='Body-local contact inputs; world-space particles; Z up; wxyz poses',
        terms='Original soft-body and robot asset terms retained; restricted external model data',
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob('*.npz')})
    (args.output/'export.json').write_text(json.dumps(report,indent=2)+'\n')
    levels=args.output/'levels';levels.mkdir();records={}
    for path in level_paths:
        env.reset(seed=101,options={'level_file':path.name});env._chamfer_dist=None
        robot=env.agent.robot;n=env.n_particles
        physical=[a for a in env._actors if a.get_collision_shapes()]
        if len(physical)!=1 or physical[0].name!='ground':raise ValueError('Unexpected Pinch physical actor set')
        state=env.get_mpm_state()
        root=robot.get_links()[0]
        arrays={**state,'root_pose':np.array(pose(robot.pose)),'root_velocity':np.r_[root.velocity,root.angular_velocity],
            'qpos':robot.get_qpos(),'qvel':robot.get_qvel(),'ground_pose':np.array(pose(physical[0].pose)),
            'goal':env.goal_array.numpy(),'deformed_distance':np.asarray(env.total_deformed_distance),
            'goal_points_observation':env.get_goal_points(),
            **{key:np.asarray(env.info[key]) for key in ('goal_depths','goal_rgbs','goal_cam_pos','goal_cam_rot','goal_cam_intrinsic')},
            **{key:getattr(env.mpm_model.struct,key).numpy()[:n].copy() for key in
               ('particle_mass','particle_vol','particle_mu_lam_ys','particle_friction_cohesion','particle_type')}}
        if any(a.dtype.kind not in 'fiu' or not np.isfinite(a).all() for a in arrays.values()):raise ValueError('Non-numeric or non-finite Pinch level')
        filename=path.stem+'.npz';np.savez_compressed(levels/filename,**arrays)
        outcome=env.evaluate()
        records[path.name]=dict(file=filename,sha256=hashlib.sha256((levels/filename).read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),particle_count=n,
            initial_outcome={k:float(v) if k!='success' else bool(v) for k,v in outcome.items()},
            source_fields={key:dict(shape=list(value.shape),dtype=str(value.dtype)) for key,value in arrays.items()})
    (levels/'levels.json').write_text(json.dumps(dict(source_commit=commit,levels=records),indent=2)+'\n')
    for relative,name in [('README.md','UPSTREAM-README.md'),('warp_maniskill/LICENSE.md','UPSTREAM-WARP-LICENSE.md')]:shutil.copyfile(args.source/relative,args.output/name)
    provenance=dict(creator='ManiSkill 2 authors; numeric conversion for Ben Caunt',source_urls=urls,source_commit=commit,
        retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        license='Original NVIDIA Source Code License for Warp and robot asset CC-BY-NC-4.0 terms retained',
        license_evidence_url=f'https://github.com/mani-skill/ManiSkill/blob/{commit}/README.md',
        attribution='ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills (Gu et al., ICLR 2023)',
        physical_basis='Reference benchmark model inputs; no claim of real material or robot calibration',
        units=report['units'],coordinate_frame=report['frame'],transforms='No rescaling; original body-local contacts and world-space initial/goal positions',
        modifications='Reference native states converted to explicit physical fields; original SDF caches converted to numeric NPZ; goal inputs retained',
        files={str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.rglob('*') if p.is_file()})
    (args.output/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(dict(completed=True,manifest_sha256=hashlib.sha256((args.output/'export.json').read_bytes()).hexdigest(),levels=records),indent=2))
finally:env.close()

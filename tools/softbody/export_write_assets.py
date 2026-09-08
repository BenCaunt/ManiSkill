"""Export original Write robot/contact inputs inside the pinned reference runtime.

The supplied level is an explicit goal input, not an expected trajectory. Only
this trusted reference tool unpickles its own generated SDF cache.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import pickle
from pathlib import Path
import shutil
import subprocess
import sys


parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--levels',type=Path,required=True)
args=parser.parse_args()
args.source=args.source.resolve()
commit='493be36121a9dd06071a57172274babe617b789f'
if subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()!=commit:
    raise ValueError('Pinned ManiSkill 2 source required')
subprocess.run(['git','-C',str(args.source),'diff','--exit-code','HEAD','--','mani_skill2','warp_maniskill'],check=True,stdout=subprocess.DEVNULL)
for name,version in [('sapien','2.2.2'),('numpy','1.23.5')]:
    if importlib.metadata.version(name)!=version:
        raise ValueError(f'Requires {name}=={version}')
args.output.mkdir(parents=True,exist_ok=False)
sys.path[:0]=[str(args.source),str(args.source/'warp_maniskill')]
import numpy as np
import gymnasium as gym
import mani_skill2.envs
from mani_skill2.envs.mpm.utils import actor2meshes

# The mount is created by this run. Seed an empty valid cache; the original
# implementation recomputes its SDFs because the geometry signature differs.
cache_path=args.source/'mani_skill2/envs/mpm/WriteEnv.sdf'
if cache_path.stat().st_size==0:
    with cache_path.open('wb') as f:pickle.dump({'signature':None,'sdfs':[]},f)
env=gym.make('Write-v0',obs_mode='none',control_mode='pd_joint_delta_pos',level_dir=str(args.levels)).unwrapped
try:
    env.reset(seed=101,options={'level_file':'line.h5'})
    if len(env.info['goal'])!=env.n_particles:
        raise ValueError('Diagnostic goal point count must match the original particle count')
    def pose(p):return np.r_[p.p,p.q].tolist()
    robot=env.agent.robot
    links=[dict(name=b.name,mass=float(b.mass),inertia=b.inertia.tolist(),com=pose(b.cmass_local_pose)) for b in robot.get_links()]
    joints=[dict(name=j.name,parent_pose=pose(j.get_pose_in_parent()),child_pose=pose(j.get_pose_in_child())) for j in robot.get_joints()]
    with cache_path.open('rb') as f:cache=pickle.load(f)
    geometry=[]
    for i,(actor,sdf) in enumerate(zip(env._coupled_actors,cache['sdfs'])):
        meshes,primitives=actor2meshes(actor,return_primitives=True)
        arrays={} if sdf is None else dict(sdf)
        for j,mesh in enumerate(meshes):arrays.update({f'vertices_{j}':mesh.vertices,f'faces_{j}':mesh.faces})
        filename=f'body-{i}.npz';np.savez_compressed(args.output/filename,**arrays)
        geometry.append(dict(name=actor.name if i==0 else f'wall_{i-1}',source_name=actor.name,
            file=filename,mesh_count=len(meshes),has_sdf=sdf is not None,
            primitives=[dict(kind=kind,size=np.asarray(size).tolist(),pose=pose(local)) for kind,size,local in primitives],
            mass=float(actor.mass),inertia=actor.inertia.tolist(),com=pose(actor.cmass_local_pose)))
    source_url=f'https://github.com/mani-skill/ManiSkill/blob/{commit}/mani_skill2/envs/mpm/write_env.py'
    report=dict(source_commit=commit,env_id='Write-v0',robot_parameters=dict(links=links,joints=joints),geometry=geometry,
        source_urls=[source_url,f'https://github.com/mani-skill/ManiSkill/blob/{commit}/mani_skill2/assets/descriptions/panda_stick.urdf'],
        units='m, kg, s',frame='Body-local collision geometry; world particle positions; Z up; wxyz poses',
        terms='Original soft-body and asset terms retained; restricted external model data, not shared-library assets',
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob('*.npz')})
    (args.output/'export.json').write_text(json.dumps(report,indent=2)+'\n')
    state=env.get_mpm_state()
    np.savez_compressed(args.output/'initial-readback.npz',**state,qpos=robot.get_qpos(),qvel=robot.get_qvel(),
        goal=np.asarray(env.info['goal'],dtype=np.float32),goal_image=env.goal_image.numpy(),
        mass=env.mpm_model.struct.particle_mass.numpy()[:env.n_particles])
    for relative,name in [('README.md','UPSTREAM-README.md'),('warp_maniskill/LICENSE.md','UPSTREAM-WARP-LICENSE.md')]:
        shutil.copyfile(args.source/relative,args.output/name)
    provenance=dict(creator='ManiSkill 2 authors; numeric conversion for Ben Caunt',
        attribution='ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills (Gu et al., ICLR 2023)',
        source_urls=report['source_urls'],source_commit=commit,retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        license='Original NVIDIA Source Code License for Warp and asset CC-BY-NC-4.0 terms retained',
        license_evidence_url=f'https://github.com/mani-skill/ManiSkill/blob/{commit}/README.md',
        usage='Restricted reference model data; excluded from the shared asset catalog',units=report['units'],coordinate_frame=report['frame'],
        transforms='Original metric SDF grids and primitive poses; identity scale',
        modifications='Exported native robot mass/COM/inertia/joint frames and SDF cache to numeric NPZ without retuning',
        physical_basis='Legacy simulation inputs and loader estimates, not measured material calibration',
        goal_scope='Explicit authored diagnostic level line.h5; no official Write level verification',
        goal_file_sha256=hashlib.sha256((args.levels/'line.h5').read_bytes()).hexdigest(),
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir() if p.is_file()})
    (args.output/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(dict(manifest_sha256=hashlib.sha256((args.output/'export.json').read_bytes()).hexdigest(),
        n_particles=env.n_particles,mass_kg=float(env.mpm_model.struct.particle_mass.numpy()[:env.n_particles].sum()),
        goal_scope=provenance['goal_scope']),indent=2))
finally:
    env.close()

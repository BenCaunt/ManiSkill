"""Export Hang model inputs inside the trusted ManiSkill 2 reference environment.

Run with pinned SAPIEN 2.2.2 and NumPy 1.23.5. This tool loads the original
RopeInit pickle only after verifying the source revision and checksum; the native
ManiSkill 3 task loads only the resulting bounded numeric pack. No rollout is
recorded. The reference SDF cache must be writable when absent or stale.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.source = args.source.resolve()
args.output.mkdir(parents=True, exist_ok=False)
commit = '493be36121a9dd06071a57172274babe617b789f'
if subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip() != commit:
    raise ValueError('A pinned ManiSkill 2 reference checkout is required')
subprocess.run(['git', '-C', str(args.source), 'diff', '--exit-code', 'HEAD', '--', 'mani_skill2', 'warp_maniskill'], check=True, stdout=subprocess.DEVNULL)
for name, expected in [('sapien', '2.2.2'), ('numpy', '1.23.5')]:
    if importlib.metadata.version(name) != expected:
        raise ValueError(f'Reference export requires {name}=={expected}')
sys.path[:0] = [str(args.source), str(args.source/'warp_maniskill')]
import numpy as np

source=args.source/'mani_skill2/envs/mpm/RopeInit.pkl'
source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
if source_sha != '1a3f1374b6682f5ea6708be8b80f6eb4eff9a61fe5afe1fdbdda438b467209e7':
    raise ValueError('Original RopeInit checksum mismatch')
import gymnasium as gym
import mani_skill2.envs
from mani_skill2.envs.mpm.utils import actor2meshes

env=gym.make('Hang-v0',obs_mode='none',control_mode='pd_joint_delta_pos').unwrapped
try:
    env.reset(seed=101)
    # Only the isolated, trusted pinned reference unpickles its own bundled
    # start states and SDF cache. Export numeric arrays with no object dtype.
    starts=env.rope_init
    arrays={}
    for key in ('x','v','F','C','vc'):
        arrays['mpm_'+key]=np.stack([a[key] for a in starts['mpm_states']])
    for key in ('robot_root_vel','robot_root_qvel','robot_qpos','robot_qvel','robot_qacc'):
        arrays[key]=np.stack([a[key] for a in starts['agent_states']])
    arrays['robot_root_pose']=np.asarray([np.r_[a['robot_root_pose'].p,a['robot_root_pose'].q] for a in starts['agent_states']])
    assert all(a.dtype.kind in 'fiu' and np.isfinite(a).all() for a in arrays.values())
    np.savez_compressed(args.output/'initial-states.npz',**arrays)
    def pose(p): return np.r_[p.p,p.q].tolist()
    robot=env.agent.robot
    actual_initial=dict(qpos=robot.get_qpos().tolist(),qvel=robot.get_qvel().tolist(),qacc=robot.get_qacc().tolist(),
        drive_position=[j.get_drive_target() for j in robot.get_active_joints()],
        drive_velocity=[j.get_drive_velocity_target() for j in robot.get_active_joints()])
    links=[dict(name=b.name,mass=float(b.mass),inertia=b.inertia.tolist(),com=pose(b.cmass_local_pose)) for b in robot.get_links()]
    joints=[dict(name=j.name,parent_pose=pose(j.get_pose_in_parent()),child_pose=pose(j.get_pose_in_child())) for j in robot.get_joints()]
    with open(env.sdf_cache,'rb') as f: cache=pickle.load(f)
    geometry=[]
    for i,(actor,sdf) in enumerate(zip(env._coupled_actors,cache['sdfs'])):
        meshes,primitives=actor2meshes(actor,return_primitives=True)
        data={} if sdf is None else dict(sdf)
        for j,mesh in enumerate(meshes): data.update({f'vertices_{j}':mesh.vertices,f'faces_{j}':mesh.faces})
        file=f'body-{i}.npz';np.savez_compressed(args.output/file,**data)
        geometry.append(dict(name=actor.name,file=file,mesh_count=len(meshes),has_sdf=sdf is not None,
            primitives=[dict(kind=kind,size=np.asarray(size).tolist(),pose=pose(local)) for kind,size,local in primitives],
            mass=float(actor.mass),inertia=actor.inertia.tolist(),com=pose(actor.cmass_local_pose)))
    out=args.output
    report=dict(source_commit='493be36121a9dd06071a57172274babe617b789f',env_id='Hang-v0',seed=101,
        rope_source_sha256=source_sha,robot_parameters=dict(links=links,joints=joints),geometry=geometry,
        selected_indices=np.asarray(env.selected_indices).tolist(),initial_shapes={k:list(v.shape) for k,v in arrays.items()},
        source_urls=['https://github.com/mani-skill/ManiSkill/blob/493be36121a9dd06071a57172274babe617b789f/mani_skill2/envs/mpm/RopeInit.pkl',
                     'https://github.com/mani-skill/ManiSkill/blob/493be36121a9dd06071a57172274babe617b789f/mani_skill2/envs/mpm/hang_env.py'],
        units='m, kg, s',frame='body-local collision geometry; world particle positions; Z up; wxyz poses',
        terms='Original ManiSkill 2 soft-body and asset terms retained; isolated reference data, not shared-library assets',
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.npz')})
    (out/'export.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'initial-readback.json').write_text(json.dumps(actual_initial,indent=2)+'\n')
    for relative, name in [('README.md', 'UPSTREAM-README.md'), ('warp_maniskill/LICENSE.md', 'UPSTREAM-WARP-LICENSE.md')]:
        shutil.copyfile(args.source/relative, out/name)
    provenance = dict(
        creator='ManiSkill 2 authors; numeric conversion for Ben Caunt',
        attribution='ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills (Gu et al., ICLR 2023)',
        source_urls=report['source_urls'], source_commit=commit,
        retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        license='Original soft-body NVIDIA Source Code License for Warp; original asset CC-BY-NC-4.0 terms retained',
        license_evidence_url=f'https://github.com/mani-skill/ManiSkill/blob/{commit}/README.md',
        usage='Restricted reference model inputs; excluded from shipped shared asset catalog',
        units=report['units'], coordinate_frame=report['frame'],
        transforms='Original metric body-local SDF grid positions/scales and primitive poses; world-space recorded starts; no rescaling',
        modifications='Converted original recorded starts and generated collision cache to numeric NPZ; SAPIEN loader mass, COM, inertia and joint frames exported without retuning',
        physical_basis='Legacy benchmark simulation inputs, not measured physical material or robot calibration',
        source_rope_sha256=source_sha,
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()})
    (out/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(dict(output=str(out), manifest_sha256=hashlib.sha256((out/'export.json').read_bytes()).hexdigest()),indent=2))
finally:
    env.close()

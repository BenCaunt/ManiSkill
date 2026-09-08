"""Export fixed Pour model inputs in the isolated pinned ManiSkill 2 runtime.

The reference alone reads its generated SDF pickle. The port consumes bounded
numeric arrays and original assets under their retained source terms.
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
import gymnasium as gym
import mani_skill2.envs
from mani_skill2.envs.mpm.utils import actor2meshes

def pose(p):
    return np.r_[p.p, p.q].tolist()

env = gym.make('Pour-v0', obs_mode='none', control_mode='pd_joint_delta_pos').unwrapped
try:
    env.reset(seed=101)
    robot = env.agent.robot
    parameters = dict(
        links=[dict(name=b.name, mass=float(b.mass), inertia=b.inertia.tolist(), com=pose(b.cmass_local_pose)) for b in robot.get_links()],
        joints=[dict(name=j.name, parent_pose=pose(j.get_pose_in_parent()), child_pose=pose(j.get_pose_in_child())) for j in robot.get_joints()])
    with open(env.sdf_cache, 'rb') as f:
        cache = pickle.load(f)
    geometry = []
    for i, (actor, sdf) in enumerate(zip(env._coupled_actors, cache['sdfs'])):
        meshes = actor2meshes(actor, visual=True)
        hulls = actor2meshes(actor, visual=False)
        data = dict(sdf)
        for j, mesh in enumerate(meshes):
            data.update({f'vertices_{j}': mesh.vertices, f'faces_{j}': mesh.faces})
        for j, hull in enumerate(hulls):
            data.update({f'rigid_vertices_{j}': hull.vertices, f'rigid_faces_{j}': hull.faces})
        file = f'body-{i}.npz'
        np.savez_compressed(args.output/file, **data)
        geometry.append(dict(name=actor.name, file=file, mesh_count=len(meshes), rigid_hull_count=len(hulls),
            has_sdf=True, primitives=[], mass=float(actor.mass), inertia=actor.inertia.tolist(), com=pose(actor.cmass_local_pose)))
    assets = args.source/'mani_skill2/assets'
    source_files = ['deformable_manipulation/bottle.glb', 'deformable_manipulation/beaker.glb', 'descriptions/panda_v2.urdf']
    report = dict(source_commit=commit, env_id='Pour-v0', robot_parameters=parameters, geometry=geometry,
        source_aabb=np.asarray(env.source_aabb).tolist(), target_aabb=np.asarray(env.target_aabb).tolist(),
        target_aabc=np.asarray(env.target_aabc).tolist(), target_height=float(env._target_height), target_radius=float(env._target_radius),
        units='m, kg, s', frame='body-local geometry and SDF; Z up; wxyz poses',
        asset_sha256={name:hashlib.sha256((assets/name).read_bytes()).hexdigest() for name in source_files},
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob('*.npz')})
    (args.output/'export.json').write_text(json.dumps(report, indent=2)+'\n')
    # Diagnostics are separate from the immutable model pack and are never used
    # to set the candidate's freshly sampled initialization.
    initial = dict(seed=101, qpos=robot.get_qpos().tolist(), qvel=robot.get_qvel().tolist(),
        source_pose=pose(env.source_container.pose), target_pose=pose(env.target_beaker.pose), h1=env.h1, h2=env.h2,
        particle_count=env.n_particles, particle_mass_kg=float(env.mpm_model.struct.particle_mass.numpy()[:env.n_particles].sum()))
    (args.output/'initial-readback.json').write_text(json.dumps(initial, indent=2)+'\n')
    np.savez_compressed(args.output/'initial-diagnostic.npz', **env.get_mpm_state())
    for relative, name in [('README.md', 'UPSTREAM-README.md'), ('warp_maniskill/LICENSE.md', 'UPSTREAM-WARP-LICENSE.md')]:
        shutil.copyfile(args.source/relative, args.output/name)
    provenance = dict(creator='ManiSkill 2 authors; numeric conversion for Ben Caunt',
        attribution='ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills (Gu et al., ICLR 2023)',
        source_urls=[f'https://github.com/mani-skill/ManiSkill/blob/{commit}/mani_skill2/envs/mpm/pour_env.py'] +
                    [f'https://github.com/mani-skill/ManiSkill/blob/{commit}/mani_skill2/assets/{name}' for name in source_files],
        retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_commit=commit,
        license='Original soft-body NVIDIA Source Code License for Warp; original asset CC-BY-NC-4.0 terms retained',
        license_evidence_url=f'https://github.com/mani-skill/ManiSkill/blob/{commit}/README.md',
        usage='Restricted external reference data; excluded from shipped shared asset catalog',
        units=report['units'], coordinate_frame=report['frame'],
        transforms='Original mesh scale bottle .025 and beaker .04; local SDF sampling positions/scales unchanged',
        modifications='Numeric export of fixed model inputs, actual reference cooked hulls, mass/COM/inertia and joint frames; separate initialization diagnostics',
        physical_basis='Legacy simulation assumptions, not measured physical calibration',
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir() if p.is_file()})
    (args.output/'PROVENANCE.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(dict(manifest_sha256=hashlib.sha256((args.output/'export.json').read_bytes()).hexdigest(), initial=initial), indent=2))
finally:
    env.close()

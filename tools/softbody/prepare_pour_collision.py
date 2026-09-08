"""Prepare an open rigid bottle cavity from the pinned original visual mesh.

CoACD is a preparation dependency only. The fluid uses the exact reference SDF.
Retain original reference mass/COM/inertia; do not infer lighter mass from this
decomposition. This intentionally replaces the reference's closed rigid hull.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path

import coacd
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

def prepare(pack, output):
    if importlib.metadata.version('coacd') != '1.0.14':
        raise ValueError('Preparation requires coacd==1.0.14')
    manifest = pack/'export.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != '54d4c40bdfe3184788d5e1f1841c8806611cff1249f0aad8b5c9a150ab982710':
        raise ValueError('Expected the pinned Pour reference model export')
    record = json.loads(manifest.read_text())
    source = pack/'body-0.npz'
    if hashlib.sha256(source.read_bytes()).hexdigest() != record['files']['body-0.npz']:
        raise ValueError('Bottle geometry checksum mismatch')
    with np.load(source, allow_pickle=False) as data:
        mesh = trimesh.Trimesh(data['vertices_0'], data['faces_0'])
    if not mesh.is_watertight or mesh.volume <= 0:
        raise ValueError('Original bottle walls must form a consistently oriented solid')
    params = dict(threshold=.0005, real_metric=True, max_convex_hull=128,
        preprocess_mode='off', resolution=2000, mcts_nodes=20, mcts_iterations=150,
        mcts_max_depth=3, pca=False, merge=True, seed=0)
    # A finite-radius path along the neck must remain clear, not merely an
    # infinitesimal center ray. The model is centered on its Z axis.
    z = np.linspace(.012, .17, 400)
    angles = np.linspace(0, 2*np.pi, 17)[:-1]
    points = np.concatenate([np.c_[np.full_like(z, r*np.cos(a)), np.full_like(z, r*np.sin(a)), z]
                             for r in (0., .003, .006) for a in angles])
    if mesh.contains(points).any():
        raise ValueError('Requested cavity probe intersects the original bottle')
    coacd.set_log_level('warn')
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), **params)
    for vertices, _ in parts:
        equations = ConvexHull(vertices).equations
        if np.any(np.all(points @ equations[:, :3].T + equations[:, 3] <= 1e-8, axis=1)):
            raise ValueError('Decomposition obstructs the original bottle cavity')
    output.mkdir(parents=True, exist_ok=False)
    data = {f'vertices_{i}': vertices for i, (vertices, _) in enumerate(parts)}
    np.savez_compressed(output/'bottle-collision.npz', **data)
    report = dict(creator='ManiSkill 2 authors; decomposition for Ben Caunt',
        source_export_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        source_geometry_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_asset_sha256=record['asset_sha256']['deformable_manipulation/bottle.glb'],
        source_url='https://github.com/mani-skill/ManiSkill/blob/493be36121a9dd06071a57172274babe617b789f/mani_skill2/assets/deformable_manipulation/bottle.glb',
        attribution='ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills (Gu et al., ICLR 2023)',
        license='Original asset CC-BY-NC-4.0 terms retained; external research inputs only',
        license_evidence_url='https://github.com/mani-skill/ManiSkill/blob/493be36121a9dd06071a57172274babe617b789f/README.md',
        retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        tool='coacd==1.0.14', tool_url='https://github.com/SarahWeiii/CoACD', parameters=params,
        units='m', frame='Original body-local Z-up vertices; no rescaling or pose adjustment',
        modifications='Open-cavity convex decomposition replaces original closed rigid hull; original mass/COM/inertia retained separately',
        reached_hull_limit=len(parts)==params['max_convex_hull'],
        accuracy_note='Requested concavity is not guaranteed when the hull limit is reached; cavity probe does not certify exterior contact accuracy',
        collision_count=len(parts), cavity_probe_radius_m=.006, cavity_probe_z_range_m=[.012,.17], cavity_probe_count=len(points),
        files={'bottle-collision.npz':hashlib.sha256((output/'bottle-collision.npz').read_bytes()).hexdigest()})
    (output/'collision.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.pack, args.output)

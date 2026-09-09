"""Freeze convex cells derived from the pinned original visual bottle walls."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .planar import merge,gpu_limits
from .profile import material_profile,convex_partition,wedge_vertices

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare(structure,output):
    report=json.loads((structure/'report.json').read_text())
    if sha(structure/'structure.npz')!=report['array_sha256']:raise ValueError('Structure hash mismatch')
    with np.load(structure/'structure.npz',allow_pickle=False) as z:profile=material_profile(z['profile'])
    cells=convex_partition(profile);sectors=report['vertices_per_ring'];phase=report['phase_radians']
    arrays={'material_profile':profile};manifest=dict(scope='Convex representation of original visual walls; differs from original closed rigid hull',
        structure_report_sha256=sha(structure/'report.json'),structure_array_sha256=report['array_sha256'],
        source_geometry_sha256=report['source_sha256'],source_surface_correspondence_bound_m=report['boundary_hausdorff_bound_m'],
        units='metres',sectors=sectors,phase_radians=phase,cells=[c.tolist() for c in cells],pieces=[],
        recipes={p.name:sha(p) for p in [Path(__file__),*[Path(__file__).with_name(n) for n in ('profile.py','planar.py','partition.py')]]})
    for sector in range(sectors):
        for index,cell in enumerate(cells):
            name=f's{sector:02d}_c{index:02d}';ideal=wedge_vertices(profile,cell,sector,sectors,phase)
            data=merge(ideal)
            if not gpu_limits(data):raise ValueError('Native convex limits exceeded: '+name)
            arrays[name+'/ideal_vertices']=ideal
            for key in ['vertices','poly_planes','poly_indices','poly_offsets']:arrays[name+'/'+key]=data[key]
            manifest['pieces'].append(dict(name=name,sector=sector,cell=index,volume=data['volume']))
    output.mkdir(exist_ok=False);np.savez_compressed(output/'cells.npz',**arrays)
    manifest['arrays_sha256']=sha(output/'cells.npz');(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    protocol=dict(scope=manifest['scope']+'; native geometry only, not physics acceptance',manifest_sha256=sha(output/'manifest.json'),
        solid_union_hausdorff_limit_m=1e-6,native_plane_hull_hausdorff_limit_m=1e-6,total_volume_relative_limit=1e-5,
        profile_symmetric_difference_area_limit_m2=1e-16,profile_overlap_area_limit_m2=1e-16,
        require_all_gpu_compatible=True,cavity_radius_m=.006,cavity_z_range_m=[.012,.17],cases=['cpu','gpu-data'])
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print('convex profile cells',len(cells),'3Dpieces',len(manifest['pieces']))


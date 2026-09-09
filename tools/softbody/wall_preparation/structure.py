"""Recover the pinned original bottle's rotational structure without editing it."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SOURCE_SHA='b3b3c1fddeaa84784a759b0b4d4bdd6525bc9a9b65b1ef53a3dafba6aa070edf'

def inspect(source):
    if hashlib.sha256(source.read_bytes()).hexdigest()!=SOURCE_SHA:
        raise ValueError('Expected the pinned original Pour visual/rigid geometry export')
    with np.load(source,allow_pickle=False) as z:vertices=z['vertices_0'];faces=z['faces_0']
    rz=np.c_[np.linalg.norm(vertices[:,:2],axis=1),vertices[:,2]]
    groups=[];mapping=np.full(len(vertices),-1)
    for i,point in enumerate(rz):
        if mapping[i]>=0:continue
        ids=np.flatnonzero(np.linalg.norm(rz-point,axis=1)<1e-7)
        if np.any(mapping[ids]>=0):raise ValueError('Overlapping ring clusters')
        mapping[ids]=len(groups);groups.append(ids)
    if len(groups)!=25 or any(len(ids)!=32 for ids in groups):raise ValueError('Unexpected ring structure')
    edges=set();caps=[]
    for face in faces:
        rings=np.unique(mapping[face])
        if len(rings)==1:caps.append(int(rings[0]))
        elif len(rings)==2:edges.add(tuple(int(i) for i in rings))
        else:raise ValueError('Unexpected cross-ring triangle')
    neighbors={i:set() for i in range(len(groups))}
    for i,j in edges:neighbors[i].add(j);neighbors[j].add(i)
    ends=[i for i in neighbors if len(neighbors[i])==1]
    if len(ends)!=2 or any(len(n) not in [1,2] for n in neighbors.values()):raise ValueError('Ring graph is not a path')
    order=[min(ends,key=lambda i:rz[groups[i],1].mean())]
    while len(order)<25:
        options=neighbors[order[-1]]-set(order)
        if len(options)!=1:raise ValueError('Disconnected ring path')
        order.append(options.pop())
    angles=np.arctan2(vertices[:,1],vertices[:,0]);phase=float(np.angle(np.exp(32j*angles).mean())/32)
    bins=np.mod(np.rint((angles-phase)/(2*np.pi/32)).astype(int),32)
    ideal=np.empty_like(vertices);profile=[];ring_vertices=[]
    for ring in order:
        ids=groups[ring]
        if len(set(bins[ids]))!=32:raise ValueError('Duplicate angular vertex within ring')
        radius,height=rz[ids].mean(0);profile.append([float(radius),float(height)])
        theta=phase+bins[ids]*(2*np.pi/32)
        ideal[ids]=np.c_[radius*np.cos(theta),radius*np.sin(theta),np.full(32,height)]
        ring_vertices.append(ids[np.argsort(bins[ids])])
    bound=float(np.linalg.norm(ideal-vertices,axis=1).max())
    report=dict(source=str(source),source_sha256=SOURCE_SHA,rings=25,vertices_per_ring=32,
                triangles=len(faces),profile_ring_order=order,profile_r_z_m=profile,
                phase_radians=phase,maximum_vertex_correspondence_error_m=bound,
                boundary_hausdorff_bound_m=bound,
                proof='Identical triangle connectivity; barycentric correspondence bounds all surface points by maximum paired-vertex distance.',
                scope='Structure/ideal-surface correspondence only. No convex wall decomposition or physics acceptance.')
    return report,dict(original_vertices=vertices,ideal_vertices=ideal,faces=faces,
                       profile=np.asarray(profile),ring_vertex_indices=np.asarray(ring_vertices))

"""Merge adjacent convex facets only within a declared metric plane tolerance."""
from collections import Counter
import numpy as np
from scipy.spatial import ConvexHull
from .partition import descriptor

PLANE_LIMIT=2e-8

def boundary(faces):
    edges={}
    for face in faces:
        for a,b in zip(face,np.roll(face,-1)):
            a,b=int(a),int(b)
            if (b,a) in edges:del edges[(b,a)]
            else:edges[(a,b)]=True
    following={}
    for a,b in edges:
        if a in following:return None
        following[a]=b
    if not following:return None
    start=min(following);cycle=[start];current=following[start]
    while current!=start:
        if current in cycle or current not in following:return None
        cycle.append(current);current=following[current]
    return np.asarray(cycle) if len(cycle)==len(edges) else None

def stable_convex_fan(vertices,cycle,normal):
    if cycle is None:return None
    points=vertices[cycle].astype(float)
    edges=np.roll(points,-1,axis=0)-points
    if np.min(np.cross(edges,np.roll(edges,-1,axis=0))@normal)<-1e-18:return None
    best=None
    for shift in range(len(cycle)):
        order=np.roll(cycle,-shift);p=vertices[order].astype(float)
        area=np.cross(p[1:-1]-p[0],p[2:]-p[0])@normal
        if area.min()>0 and (best is None or area.min()>best[0]):best=(float(area.min()),order)
    return None if best is None else best[1]

def merge(points,cleanup_limit=None):
    data=descriptor(points,cleanup_limit=cleanup_limit,maximum=64)
    vertices=data['vertices'];faces=data['faces'];hull=ConvexHull(vertices.astype(float))
    # Recompute each oriented triangle's precise outward plane.
    tri=vertices[faces].astype(float);normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    areas=np.linalg.norm(normal,axis=1);normal/=areas[:,None]
    planes=np.c_[normal,-np.einsum('ij,ij->i',normal,tri[:,0])]
    edge_faces={}
    for i,face in enumerate(faces):
        for a,b in zip(face,np.roll(face,-1)):edge_faces.setdefault(tuple(sorted((int(a),int(b)))),[]).append(i)
    neighbors=[set() for _ in faces]
    for attached in edge_faces.values():
        if len(attached)!=2:raise ValueError('Input hull is not a closed triangle manifold')
        a,b=attached;neighbors[a].add(b);neighbors[b].add(a)
    remaining=set(range(len(faces)));polygons=[];polyplanes=[]
    for seed in np.argsort(-areas,kind='stable'):
        seed=int(seed)
        if seed not in remaining:continue
        group={seed};remaining.remove(seed);queue=[seed];plane=planes[seed]
        while queue:
            current=queue.pop()
            for candidate in sorted(neighbors[current]&remaining):
                candidate_points=vertices[faces[candidate]].astype(float)
                if plane[:3]@planes[candidate,:3]>.99999 and np.max(abs(candidate_points@plane[:3]+plane[3]))<=PLANE_LIMIT:
                    group.add(candidate);remaining.remove(candidate);queue.append(candidate)
        cycle=stable_convex_fan(vertices,boundary(faces[sorted(group)]),plane[:3])
        if cycle is None:
            for i in sorted(group):polygons.append(faces[i]);polyplanes.append(planes[i])
        else:polygons.append(cycle);polyplanes.append(plane)
    # Merging facets can expose a redundant point on an edge shared by only
    # two polygons. PhysX requires at least three incident faces per vertex.
    # Remove such a point from both boundary loops only within the same metric
    # tolerance; the independent whole-union verifier still checks the result.
    while True:
        counts=Counter(int(v) for p in polygons for v in p)
        redundant=[v for v,count in counts.items() if count<3]
        if not redundant:break
        for vertex in redundant:
            incident=[i for i,p in enumerate(polygons) if vertex in p]
            if len(incident)!=2:raise ValueError('Unexpected polygon incidence')
            for i in incident:
                p=polygons[i];j=list(p).index(vertex)
                if len(p)<4:raise ValueError('Cannot remove a triangle corner')
                a,b=vertices[p[j-1]].astype(float),vertices[p[(j+1)%len(p)]].astype(float)
                edge=b-a;length2=edge@edge
                if length2<=0:raise ValueError('Degenerate polygon edge')
                t=np.clip((vertices[vertex]-a)@edge/length2,0.,1.)
                if np.linalg.norm(vertices[vertex]-(a+t*edge))>PLANE_LIMIT:
                    raise ValueError('Redundant vertex exceeds metric removal limit')
            for i in incident:polygons[i]=polygons[i][polygons[i]!=vertex]
    used=np.unique(np.concatenate(polygons));remap=np.full(len(vertices),-1);remap[used]=np.arange(len(used))
    vertices=np.ascontiguousarray(vertices[used]);indices=np.concatenate([remap[p] for p in polygons]).astype(np.uint32)
    offsets=np.r_[0,np.cumsum([len(p) for p in polygons])].astype(np.uint32)
    return dict(vertices=vertices,poly_indices=indices,poly_offsets=offsets,poly_planes=np.asarray(polyplanes,np.float32),
                volume=float(ConvexHull(vertices.astype(float)).volume))

def gpu_limits(data):
    incidence=Counter(int(v) for v in data['poly_indices'])
    return len(data['vertices'])<=64 and len(data['poly_planes'])<=64 and max(incidence.values())<=32

def split(points,path=''):
    data=merge(points,cleanup_limit=2e-8 if path else None)
    if gpu_limits(data):return dict(path=path,leaf=True,**data)
    if len(path)>20:raise ValueError('Planar partition did not converge')
    v=data['vertices'];hull=ConvexHull(v.astype(float))
    axis=int(np.argmax(np.ptp(v,axis=0)));coordinate=float((float(v[:,axis].min())+float(v[:,axis].max()))*.5)
    edges={tuple(sorted((int(f[i]),int(f[(i+1)%3])))) for f in hull.simplices for i in range(3)}
    cuts=[]
    for a,b in sorted(edges):
        va,vb=v[a].astype(float),v[b].astype(float);da,db=va[axis]-coordinate,vb[axis]-coordinate
        if da*db<0:
            cut=va+(vb-va)*(-da/(db-da));cut[axis]=coordinate;cuts.append(cut)
    children=[]
    for side,mask in [('L',v[:,axis]<=coordinate),('R',v[:,axis]>=coordinate)]:
        children.append(split(np.vstack([v[mask],np.asarray(cuts)]),path+side))
    return dict(path=path,leaf=False,axis=axis,coordinate=coordinate,children=children,**data)

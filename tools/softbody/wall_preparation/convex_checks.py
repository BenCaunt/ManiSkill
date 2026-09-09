"""Independent convex-set coverage proof; does not import native or partition code."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull,HalfspaceIntersection

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def triangle_distance(point,triangles):
    """Distance via valid face/edge points, without an absolute mesh tolerance.

    Each candidate is a convex combination of triangle vertices. This also
    guarantees a distance no larger than the nearest vertex witness.
    """
    triangles=np.asarray(triangles,dtype=np.float64);point=np.asarray(point,dtype=np.float64)
    a,b,c=triangles[:,0],triangles[:,1],triangles[:,2]
    best=float('inf')
    for start,end in [(a,b),(b,c),(c,a)]:
        edge=end-start;length2=np.einsum('ij,ij->i',edge,edge)
        t=np.divide(np.einsum('ij,ij->i',point-start,edge),length2,out=np.zeros(len(edge)),where=length2>0)
        closest=start+np.clip(t,0,1)[:,None]*edge
        best=min(best,float(np.linalg.norm(closest-point,axis=1).min()))
    normal=np.cross(b-a,c-a);size=np.linalg.norm(normal,axis=1);valid=size>0
    normal=normal[valid]/size[valid,None];a,b,c=a[valid],b[valid],c[valid]
    projected=point-np.einsum('ij,ij->i',point-a,normal)[:,None]*normal
    weights=np.column_stack([np.einsum('ij,ij->i',np.cross(b-projected,c-projected),normal),
        np.einsum('ij,ij->i',np.cross(c-projected,a-projected),normal),
        np.einsum('ij,ij->i',np.cross(a-projected,b-projected),normal)])
    inside=(weights>=0).all(axis=1)&(weights.sum(axis=1)>0)
    if inside.any():
        weights=weights[inside]/weights[inside].sum(axis=1)[:,None]
        closest=weights[:,0,None]*a[inside]+weights[:,1,None]*b[inside]+weights[:,2,None]*c[inside]
        best=min(best,float(np.linalg.norm(closest-point,axis=1).min()))
    return best

def directed_distance(points,target):
    points=np.asarray(points,dtype=np.float64);target=np.asarray(target,dtype=np.float64)
    hull=ConvexHull(target)
    outside=np.max(points@hull.equations[:,:3].T+hull.equations[:,3],axis=1)>0
    if not outside.any():return 0.
    distance=np.array([triangle_distance(p,target[hull.simplices]) for p in points[outside]])
    if not np.isfinite(distance).all():raise ValueError('Nonfinite point-to-convex distance')
    return float(distance.max())

def hausdorff(a,b):return max(directed_distance(a,b),directed_distance(b,a))

def clipped(parent,axis,coordinate,side):
    parent=np.asarray(parent,dtype=np.float64);hull=ConvexHull(parent)
    plane=np.zeros(4);plane[axis]=1 if side==0 else -1;plane[3]=-coordinate*plane[axis]
    center=parent.mean(axis=0);distance=plane[:3]@center+plane[3]
    if distance>=-1e-12:
        best=parent[np.argmin(parent@plane[:3]+plane[3])]
        inward=float(plane[:3]@best+plane[3])
        if inward>=0:raise ValueError('Empty reference halfspace intersection')
        crossing=distance/(distance-inward)
        center=center+(best-center)*((crossing+1)*.5)
    spaces=np.vstack([hull.equations,plane])
    if np.max(spaces[:,:3]@center+spaces[:,3])>=0:raise ValueError('No strict interior for independent clipping')
    vertices=HalfspaceIntersection(spaces,center).intersections
    if not np.isfinite(vertices).all() or np.max(vertices@spaces[:,:3].T+spaces[:,3])>1e-8:
        raise ValueError('Invalid independent halfspace intersection')
    return vertices

def native_faces(data,limit):
    vertices=data['vertices'].astype(float);planes=data['planes'].astype(float)
    indices=data['polygon_indices'];offsets=data['polygon_offsets'];failures=[]
    if planes.ndim!=2 or planes.shape[1]!=4 or len(offsets)!=len(planes)+1 or offsets[0]!=0 or offsets[-1]!=len(indices) or np.any(np.diff(offsets.astype(int))<3):
        raise ValueError('Invalid native polygon layout')
    if np.any(indices>=len(vertices)):raise ValueError('Invalid native polygon index')
    if np.max(abs(np.linalg.norm(planes[:,:3],axis=1)-1))>1e-5:failures.append('Nonunit native plane')
    if np.max(vertices@planes[:,:3].T+planes[:,3])>limit:failures.append('Native planes are not a convex enclosure')
    triangles=[];edges={}
    for i,plane in enumerate(planes):
        poly=indices[offsets[i]:offsets[i+1]].astype(int)
        if len(set(poly))!=len(poly):failures.append('Repeated polygon vertex')
        if np.max(abs(vertices[poly]@plane[:3]+plane[3]))>limit:failures.append('Native polygon leaves its plane')
        for a,b in zip(poly,np.roll(poly,-1)):
            pair=tuple(sorted((int(a),int(b))));edges.setdefault(pair,[]).append(1 if a<b else -1)
        for j in range(1,len(poly)-1):
            face=poly[[0,j,j+1]];a,b,c=vertices[face]
            if np.dot(np.cross(b-a,c-a),plane[:3])<=0:failures.append('Degenerate or inward native face')
            triangles.append(face)
    if any(len(signs)!=2 or sum(signs)!=0 for signs in edges.values()):failures.append('Native polygons are not a closed oriented manifold')
    triangles=vertices[np.asarray(triangles)]-vertices.mean(axis=0)
    signed_volume=float(np.sum(np.einsum('ij,ij->i',triangles[:,0],np.cross(triangles[:,1],triangles[:,2])))/6)
    volume=float(ConvexHull(vertices).volume)
    if abs(signed_volume-volume)>volume*1e-5:failures.append('Native polygon volume differs from vertex hull')
    return failures

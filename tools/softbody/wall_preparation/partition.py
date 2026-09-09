"""Represent a convex hull as bounded convex pieces without changing its union."""
import numpy as np
from scipy.spatial import ConvexHull


def descriptor(points,cleanup_limit=None,maximum=32):
    points=np.unique(np.asarray(points,dtype=np.float32),axis=0)
    hull=ConvexHull(points.astype(np.float64))
    vertices=points[hull.vertices]
    original=vertices.astype(np.float64)
    # Intersecting triangulated faces can create almost-collinear cut vertices.
    # Bound removal against the entire initial child, not just the last removal.
    # The final native union must separately satisfy the original 1e-6 m gate.
    if cleanup_limit is not None:
        while len(vertices)>maximum:
            best=None
            for index in range(len(vertices)):
                remaining=np.delete(vertices,index,axis=0)
                try:candidate=ConvexHull(remaining.astype(np.float64))
                except Exception:continue
                error=max(0.,float(np.max(original@candidate.equations[:,:3].T+candidate.equations[:,3])))
                if error<=cleanup_limit and (best is None or error<best[0]):
                    best=(error,remaining[candidate.vertices])
            if best is None:break
            vertices=best[1]
    hull=ConvexHull(vertices.astype(np.float64))
    faces=hull.simplices.copy()
    for i,face in enumerate(faces):
        a,b,c=vertices[face].astype(np.float64)
        if np.dot(np.cross(b-a,c-a),hull.equations[i,:3])<0:faces[i]=face[[0,2,1]]
    return dict(vertices=np.ascontiguousarray(vertices),faces=faces.astype(np.uint32),
                planes=hull.equations.astype(np.float32),volume=float(hull.volume))


def split(points,maximum=32,path=''):
    data=descriptor(points,cleanup_limit=2e-8 if path else None,maximum=maximum)
    v=data['vertices'];faces=data['faces']
    if len(v)<=maximum:
        return dict(path=path,leaf=True,**data)
    if len(path)>20:raise ValueError('Convex partition did not converge')
    axis=int(np.argmax(np.ptp(v,axis=0)));coordinate=float((v[:,axis].min()+v[:,axis].max())*.5)
    edges={tuple(sorted((int(f[i]),int(f[(i+1)%3])))) for f in faces for i in range(3)}
    cuts=[]
    for a,b in sorted(edges):
        va,vb=v[a].astype(np.float64),v[b].astype(np.float64)
        da,db=va[axis]-coordinate,vb[axis]-coordinate
        if da*db<0:
            cut=va+(vb-va)*(-da/(db-da));cut[axis]=coordinate;cuts.append(cut)
    children=[]
    for side,mask in [('L',v[:,axis]<=coordinate),('R',v[:,axis]>=coordinate)]:
        child=np.vstack([v[mask],np.asarray(cuts)])
        children.append(split(child,maximum,path+side))
    return dict(path=path,leaf=False,vertices=v,faces=faces,planes=data['planes'],volume=data['volume'],
                axis=axis,coordinate=coordinate,children=children)


def leaves(node):
    if node['leaf']:return [node]
    return [leaf for child in node['children'] for leaf in leaves(child)]

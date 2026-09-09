"""Partition a simple material meridian into convex polygons, then wedge it."""
import numpy as np

AREA_EPS=1e-20

def cross(a,b):return float(a[0]*b[1]-a[1]*b[0])
def area(points):return sum(cross(a,b) for a,b in zip(points,np.roll(points,-1,axis=0)))*.5

def material_profile(rings):
    rings=np.asarray(rings,dtype=float)
    if rings.ndim!=2 or rings.shape[1]!=2 or np.any(rings[:,0]<=0):raise ValueError('Invalid positive-radius profile')
    polygon=np.vstack([rings,[0.,rings[-1,1]],[0.,rings[0,1]]])
    if area(polygon)<=0:raise ValueError('Expected an outward-upward CCW material profile')
    return polygon

def triangulate(points):
    points=np.asarray(points,dtype=float);remaining=list(range(len(points)));triangles=[]
    # Exact collinear boundary points do not change the polygon. Keep their
    # source indices in the profile, but omit them from its triangulation.
    changed=True
    while changed and len(remaining)>3:
        changed=False
        for j,i in enumerate(remaining):
            a,b,c=points[[remaining[j-1],i,remaining[(j+1)%len(remaining)]]]
            if abs(cross(b-a,c-b))<=AREA_EPS and np.dot(b-a,b-c)<=0:
                del remaining[j];changed=True;break
    while len(remaining)>3:
        ears=[]
        for j,b in enumerate(remaining):
            ids=[remaining[j-1],b,remaining[(j+1)%len(remaining)]];a,p,c=points[ids]
            twice_area=cross(p-a,c-p)
            if twice_area<=AREA_EPS:continue
            other=[i for i in remaining if i not in ids]
            if any(cross(p-a,points[i]-a)>=-AREA_EPS and cross(c-p,points[i]-p)>=-AREA_EPS and cross(a-c,points[i]-c)>=-AREA_EPS for i in other):continue
            scale=float(np.sum((p-a)**2)+np.sum((c-p)**2)+np.sum((a-c)**2))
            ears.append((twice_area/scale,j,ids))
        if not ears:raise ValueError('Profile is nonsimple or cannot be triangulated robustly')
        _,j,ids=max(ears);triangles.append(ids);del remaining[j]
    if area(points[remaining])<=AREA_EPS:raise ValueError('Degenerate final profile triangle')
    triangles.append(remaining)
    return [np.asarray(p,dtype=int) for p in triangles]

def joined_boundary(left,right):
    edges=set()
    for polygon in [left,right]:
        for a,b in zip(polygon,np.roll(polygon,-1)):
            pair=(int(a),int(b));reverse=(int(b),int(a))
            if reverse in edges:edges.remove(reverse)
            elif pair in edges:return None
            else:edges.add(pair)
    following={}
    for a,b in edges:
        if a in following:return None
        following[a]=b
    if len(following)!=len(edges):return None
    start=min(following);cycle=[start];current=following[start]
    while current!=start:
        if current in cycle or current not in following:return None
        cycle.append(current);current=following[current]
    return np.asarray(cycle) if len(cycle)==len(edges) else None

def convex_partition(points):
    points=np.asarray(points,dtype=float);cells=triangulate(points)
    while True:
        candidates=[]
        for i in range(len(cells)):
            for j in range(i+1,len(cells)):
                if len(set(cells[i])&set(cells[j]))<2:continue
                polygon=joined_boundary(cells[i],cells[j])
                if polygon is None:continue
                p=points[polygon];edges=np.roll(p,-1,axis=0)-p
                if min(cross(a,b) for a,b in zip(edges,np.roll(edges,-1,axis=0)))<-AREA_EPS:continue
                if abs(area(p)-area(points[cells[i]])-area(points[cells[j]]))>1e-16:continue
                candidates.append((area(p),i,j,polygon))
        if not candidates:break
        _,i,j,polygon=max(candidates,key=lambda p:(p[0],-p[1],-p[2]))
        cells[i]=polygon;del cells[j]
    return cells

def wedge_vertices(profile,cell,sector,sectors,phase):
    values=np.asarray(profile)[cell];angles=phase+np.array([sector,sector+1])*(2*np.pi/sectors)
    return np.array([[r*np.cos(a),r*np.sin(a),z] for a in angles for r,z in values])

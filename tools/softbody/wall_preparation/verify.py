"""Independent wall-union proof using GEOS, halfspaces and native readbacks.

Does not import the candidate profile partition, wedge or facet-merging code.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull,HalfspaceIntersection
from shapely.geometry import Polygon,MultiPoint,Point
from shapely.ops import unary_union
from .convex_checks import hausdorff,native_faces,clipped

SOURCE_SHA='b3b3c1fddeaa84784a759b0b4d4bdd6525bc9a9b65b1ef53a3dafba6aa070edf'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def cross2(a,b):return a[0]*b[1]-a[1]*b[0]
def signed_volume(v,faces):
    t=v[faces]-v.mean(axis=0)
    return float(np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2])).sum()/6)
def radial_moment(poly):
    following=np.roll(poly,-1,axis=0)
    return float(np.sum((poly[:,0]+following[:,0])*(poly[:,0]*following[:,1]-following[:,0]*poly[:,1]))/6)

def profile_cell_halfspaces(poly,sector,count,phase):
    a,b=phase+np.array([sector,sector+1])*2*np.pi/count;mid=(a+b)*.5;c=np.cos((b-a)*.5)
    planes=[]
    for start,end in zip(poly,np.roll(poly,-1,axis=0)):
        dr,dz=end-start;n=np.array([dz*np.cos(mid)/c,dz*np.sin(mid)/c,-dr])
        d=-(dz*start[0]-dr*start[1]);norm=np.linalg.norm(n)
        if norm<=0:raise ValueError('Degenerate profile edge')
        planes.append(np.r_[n,d]/norm)
    planes.extend([[np.sin(a),-np.cos(a),0,0],[-np.sin(b),np.cos(b),0,0]])
    r,z=poly.mean(axis=0);center=np.array([r*c*np.cos(mid),r*c*np.sin(mid),z])
    planes=np.asarray(planes)
    if np.max(planes[:,:3]@center+planes[:,3])>=0:raise ValueError('Non-interior analytic wedge center')
    return planes,center

def verify_source_surface(structure,source,protocol):
    if sha(source)!=SOURCE_SHA:raise ValueError('Unpinned original source geometry')
    sr=json.loads((structure/'report.json').read_text())
    if sha(structure/'structure.npz')!=sr['array_sha256']:raise ValueError('Source structure changed')
    data=load(structure/'structure.npz');original=load(source)['vertices_0'];faces=load(source)['faces_0']
    if not np.array_equal(data['original_vertices'],original) or not np.array_equal(data['faces'],faces):raise ValueError('Source mesh modified')
    rings=data['ring_vertex_indices'];count=sr['vertices_per_ring']
    if rings.shape!=(25,32) or count!=32 or not np.array_equal(np.sort(rings.ravel()),np.arange(len(original))):
        raise ValueError('Invalid source ring partition')
    profile=np.array([[np.linalg.norm(original[row,:2],axis=1).mean(),original[row,2].mean()] for row in rings])
    if np.max(abs(profile-data['profile']))>1e-15:raise ValueError('Profile differs from source ring means')
    theta=sr['phase_radians']+np.arange(count)*2*np.pi/count;ideal=np.empty_like(original)
    for row,(r,z) in zip(rings,profile):ideal[row]=np.c_[r*np.cos(theta),r*np.sin(theta),np.full(count,z)]
    if np.max(abs(ideal-data['ideal_vertices']))>1e-15:raise ValueError('Ideal surface not derived from source profile')
    correspondence=float(np.linalg.norm(ideal-original,axis=1).max())
    # Prove all side triangles cover exactly the expected planar ring quads.
    ring_id=np.empty(len(original),int);sector_id=np.empty(len(original),int)
    for i,row in enumerate(rings):ring_id[row]=i;sector_id[row]=np.arange(count)
    sides={};caps={0:[],24:[]}
    for face in faces:
        rows=np.unique(ring_id[face]);sectors=np.unique(sector_id[face])
        if len(rows)==1:
            if rows[0] not in caps:raise ValueError('Unexpected cap in source profile')
            caps[int(rows[0])].append(face);continue
        if len(rows)!=2 or rows[1]-rows[0]!=1 or len(sectors)!=2:raise ValueError('Unexpected wall triangle connectivity')
        start=next((int(s) for s in sectors if (s+1)%count in sectors),None)
        if start is None:raise ValueError('Wall triangle skips angular sectors')
        sides.setdefault((int(rows[0]),start),[]).append(face)
    if set(sides)!={(i,j) for i in range(24) for j in range(count)}:raise ValueError('Missing source wall quads')
    for (i,j),triangles in sides.items():
        if len(triangles)!=2:raise ValueError('Incorrect source wall quad triangulation')
        quad=[int(rings[i,j]),int(rings[i,(j+1)%count]),int(rings[i+1,(j+1)%count]),int(rings[i+1,j])]
        edges=Counter(tuple(sorted((int(a),int(b)))) for face in triangles for a,b in zip(face,np.roll(face,-1)))
        expected={tuple(sorted((a,b))) for a,b in zip(quad,quad[1:]+quad[:1])}
        if {e for e,n in edges.items() if n==1}!=expected or sorted(edges.values())!=[1,1,1,1,2]:raise ValueError('Source quad overlap or gap')
        dr,dz=profile[i+1]-profile[i];mid=sr['phase_radians']+(j+.5)*2*np.pi/count
        normal=np.array([dz*np.cos(mid)/np.cos(np.pi/count),dz*np.sin(mid)/np.cos(np.pi/count),-dr])
        for face in triangles:
            a,b,c=ideal[face]
            if np.cross(b-a,c-a)@normal<=0:raise ValueError('Reversed source wall face')
    for i,triangles in caps.items():
        if len(triangles)!=count-2:raise ValueError('Unexpected cap triangle count')
        shapes=[Polygon(ideal[face,:2]) for face in triangles];union=unary_union(shapes);target=Polygon(ideal[rings[i],:2])
        if union.symmetric_difference(target).area>1e-16 or sum(p.area for p in shapes)-union.area>1e-16:raise ValueError('Source cap overlap/gap')
        for face in triangles:
            a,b,c=ideal[face]
            if np.cross(b-a,c-a)[2]*(1 if i==24 else -1)<=0:raise ValueError('Reversed cap')
    material=np.vstack([profile,[0,profile[-1,1]],[0,profile[0,1]]])
    mesh_volume=signed_volume(ideal,faces);analytic_volume=count*np.sin(2*np.pi/count)*radial_moment(material)
    if mesh_volume<=0 or abs(mesh_volume/analytic_volume-1)>1e-10:raise ValueError('Lathe solid volume differs from source mesh boundary')
    if correspondence>protocol['solid_union_hausdorff_limit_m']:raise ValueError('Source regularization exceeds geometry limit')
    return material,correspondence,analytic_volume,sr

def cavity_clearance(vertices,low,high):
    if vertices[:,2].max()<=low or vertices[:,2].min()>=high:return None
    if vertices[:,2].min()<low:vertices=clipped(vertices,2,low,1)
    if vertices[:,2].max()>high:vertices=clipped(vertices,2,high,0)
    return float(MultiPoint(vertices[:,:2]).convex_hull.distance(Point(0,0)))

def verify_profile_partition(profile,cells,protocol):
    if profile.ndim!=2 or profile.shape[1]!=2 or not np.isfinite(profile).all() or np.any(profile[:,0]<0):
        raise ValueError('Invalid material profile')
    target=Polygon(profile)
    if not target.is_valid or target.area<=0 or not target.exterior.is_ccw:raise ValueError('Invalid material polygon')
    polygons=[]
    for cell in cells:
        if len(cell)<3 or any(type(i) is not int for i in cell) or len(set(cell))!=len(cell) or min(cell)<0 or max(cell)>=len(profile):
            raise ValueError('Invalid profile cell indices')
        poly=Polygon(profile[cell])
        if not poly.is_valid or poly.area<=0 or not poly.exterior.is_ccw or poly.symmetric_difference(poly.convex_hull).area>1e-18:
            raise ValueError('Nonconvex or reversed profile cell')
        polygons.append(poly)
    union=unary_union(polygons)
    difference=union.symmetric_difference(target).area;overlap=sum(p.area for p in polygons)-union.area
    if difference>protocol['profile_symmetric_difference_area_limit_m2'] or overlap>protocol['profile_overlap_area_limit_m2']:
        raise ValueError('Material profile coverage has gaps or overlaps')
    return polygons,difference,overlap

def verify_inventory(manifest,execution,protocol):
    pieces=manifest['pieces'];ids={(p['sector'],p['cell']) for p in pieces}
    if len(ids)!=len(pieces) or ids!={(s,c) for s in range(manifest['sectors']) for c in range(len(manifest['cells']))}:
        raise ValueError('Angular/profile coverage incomplete or duplicated')
    if any(p['name']!=f"s{p['sector']:02d}_c{p['cell']:02d}" for p in pieces):raise ValueError('Invalid native piece name')
    cases=[trial['case'] for trial in execution['cases']]
    if len(cases)!=len(set(cases)) or set(cases)!=set(protocol['cases']):raise ValueError('Missing or duplicate native cases')
    if execution['input_manifest_sha256']!=protocol['manifest_sha256']:raise ValueError('Native input hash mismatch')

def verify_native_piece(actual,ideal,source_bound,protocol,gpu_compatible):
    if any(not np.isfinite(v).all() for v in actual.values()):raise ValueError('Nonfinite native geometry')
    failures=native_faces(actual,protocol['solid_union_hausdorff_limit_m'])
    if gpu_compatible is False:failures.append('GPU incompatible')
    vertex_error=hausdorff(ideal,actual['vertices'])
    native_planes=actual['planes'].astype(float);native_center=actual['vertices'].mean(axis=0).astype(float)
    if np.max(native_planes[:,:3]@native_center+native_planes[:,3])>=0:raise ValueError('Native planes lack expected strict interior')
    plane_hull=HalfspaceIntersection(native_planes,native_center).intersections
    plane_error=hausdorff(ideal,plane_hull);coherence=hausdorff(actual['vertices'],plane_hull)
    if coherence>protocol['native_plane_hull_hausdorff_limit_m']:failures.append('Native plane/vertex disagreement')
    bound=source_bound+max(vertex_error,plane_error)
    if bound>protocol['solid_union_hausdorff_limit_m']:failures.append('Solid union geometry bound exceeded')
    clearances=[cavity_clearance(v,*protocol['cavity_z_range_m']) for v in [actual['vertices'].astype(float),plane_hull]]
    row=dict(vertex_hausdorff_error_m=vertex_error,plane_hull_hausdorff_error_m=plane_error,
             native_plane_vertex_disagreement_m=coherence,source_solid_union_bound_m=bound)
    return row,failures,ConvexHull(actual['vertices']).volume,ConvexHull(plane_hull).volume,[v for v in clearances if v is not None]

def evaluate(inputs,output,structure,source):
    manifest=json.loads((inputs/'manifest.json').read_text());protocol=json.loads((inputs/'protocol.json').read_text())
    if sha(inputs/'manifest.json')!=protocol['manifest_sha256'] or sha(inputs/'cells.npz')!=manifest['arrays_sha256']:
        raise ValueError('Candidate input changed')
    if sha(structure/'report.json')!=manifest['structure_report_sha256']:raise ValueError('Source structure report changed')
    profile,source_bound,volume,sr=verify_source_surface(structure,source,protocol)
    arrays=load(inputs/'cells.npz')
    if np.max(abs(profile-arrays['material_profile']))>1e-15:raise ValueError('Candidate changed material profile')
    polygons,difference,overlap=verify_profile_partition(profile,manifest['cells'],protocol)
    if manifest['sectors']!=32 or manifest['phase_radians']!=sr['phase_radians']:raise ValueError('Angular frame changed')
    execution=json.loads((output/'report.json').read_text())
    verify_inventory(manifest,execution,protocol)
    results={}
    for trial in execution['cases']:
        case=trial['case'];folder=output/case;failures=[];rows=[];sum_vertex_volume=0.;sum_plane_volume=0.;clearances=[]
        if trial['exit_code']!=0 or trial['timed_out']:failures.append('Native cooking did not complete')
        if trial['completed']!=len(manifest['pieces']) or trial['errors']!=0:failures.append('Native execution count mismatch or reported errors')
        events=[json.loads(s) for s in (folder/'progress.jsonl').read_text().splitlines()]
        complete=[e for e in events if e['event']=='complete'];records={e['name']:e for e in complete}
        if any(e['event']=='error' for e in events):failures.append('Native cooking reported an error')
        if len(records)!=len(complete) or set(records)!={p['name'] for p in manifest['pieces']}:failures.append('Missing/duplicate native meshes')
        for piece in manifest['pieces']:
            name=piece['name']
            if name not in records:continue
            event=records[name];file=folder/(name+'.npz')
            if sha(file)!=event['readback_sha256'] or sha(folder/(name+'.bin'))!=event['blob_sha256']:raise ValueError('Native record hash mismatch')
            actual=load(file)
            if event['build_gpu_data']!=(case=='gpu-data') or event['status']!=0:failures.append(name+': native cooking configuration or status mismatch')
            planes,center=profile_cell_halfspaces(profile[manifest['cells'][piece['cell']]],piece['sector'],manifest['sectors'],manifest['phase_radians'])
            ideal=HalfspaceIntersection(planes,center).intersections
            if hausdorff(ideal,arrays[name+'/ideal_vertices'])>1e-10:failures.append(name+': candidate wedge differs from independent halfspace cell')
            gpu=event['gpu_compatible'] if case=='gpu-data' and protocol['require_all_gpu_compatible'] else None
            row,errors,v_volume,p_volume,cavity=verify_native_piece(actual,ideal,source_bound,protocol,gpu)
            failures.extend(name+': '+f for f in errors);sum_vertex_volume+=v_volume;sum_plane_volume+=p_volume
            clearances.extend(cavity);rows.append(dict(name=name,**row))
        volume_errors=[abs(sum_vertex_volume/volume-1),abs(sum_plane_volume/volume-1)]
        if max(volume_errors)>protocol['total_volume_relative_limit']:failures.append('Summed cell volume differs from original wall solid')
        if not clearances or min(clearances)<protocol['cavity_radius_m']:failures.append('Continuous finite-radius cavity is obstructed')
        results[case]=dict(passed=not failures,failures=failures,pieces=rows,maximum_source_solid_union_bound_m=max((r['source_solid_union_bound_m'] for r in rows),default=None),
            vertex_volume_relative_error=volume_errors[0],plane_volume_relative_error=volume_errors[1],minimum_continuous_cavity_clearance_m=min(clearances,default=None))
    return dict(scope=protocol['scope'],cases=results,profile_cell_count=len(polygons),piece_count=len(manifest['pieces']),
                original_ideal_wall_volume_m3=volume,source_surface_correspondence_bound_m=source_bound,
                profile_symmetric_difference_m2=difference,profile_overlap_m2=overlap,
                gpu_geometry_passed=results.get('gpu-data',{}).get('passed',False),
                proof='GEOS proves meridian coverage. Sector maps are bijections away from the axis; independently reconstructed halfspaces equal each ideal convex cell. Verified source ring quads/caps equal the ideal lathe boundary. Piece Hausdorff errors bound their union; source triangle correspondence adds the original-surface bound. Native vertex and implicit plane hulls are both checked. Cavity clearance uses projections of cells clipped to the entire declared height interval.')

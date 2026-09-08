"""Independent NumPy reconstruction of legacy Write height images and outcomes.

Equations are from ManiSkill 2 v0.5.3 height_rasterizer.py and write_env.py;
LEGACY-SOURCE-LICENSE.md applies. No simulator or candidate code is imported.

The pinned sm86 reference PTX uses separate float32 add/multiply/subtract,
div.approx.ftz.f32 and sqrt.approx.ftz.f32. CPU division/sqrt are not an exact
oracle for those instructions. height_bounds propagates their published error
bounds; a score whose success depends on an uncertain pixel is rejected.
Arithmetic contract: https://docs.nvidia.com/cuda/parallel-thread-execution/
#floating-point-instructions-div and #floating-point-instructions-sqrt.
"""
import numpy as np
from .artifacts import InvalidArtifact


def height_image(points):
    points=np.asarray(points,dtype=np.float32)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise InvalidArtifact('Invalid Write raster points')
    image=np.zeros((64,64),dtype=np.int32)
    scale=np.float32(64/.21)
    q=points+np.array([.105,.105,0.],dtype=np.float32)
    start=q[:,:2]*scale-np.float32(2.)
    # Warp roundf rounds halves away from zero, unlike NumPy rint.
    start=np.sign(start)*np.floor(abs(start.astype(np.float64))+.5)
    if np.any(abs(start)>2**30):
        raise InvalidArtifact('Write raster coordinates exceed integer range')
    start=start.astype(np.int32)
    radius=np.float32(2.)/scale
    radius_squared=radius*radius
    for i in range(5):
        for j in range(5):
            gx,gy=start[:,0]+i,start[:,1]+j
            dx=gx.astype(np.float32)/scale-q[:,0]
            dy=gy.astype(np.float32)/scale-q[:,1]
            z_squared=radius_squared-dx*dx-dy*dy
            mask=(gx>=0)&(gx<64)&(gy>=0)&(gy<64)&(z_squared>0)
            values=(np.sqrt(z_squared[mask])+q[mask,2])*np.float32(1000.)
            if np.any(abs(values)>2**30):
                raise InvalidArtifact('Write raster heights exceed integer range')
            np.maximum.at(image,(gy[mask],gx[mask]),values.astype(np.int32))
    return image


def compare_images(goal,current):
    if goal.shape!=(64,64) or current.shape!=(64,64):
        raise InvalidArtifact('Write height images must be 64 by 64')
    a,an=goal<40,goal<50
    b,bn=current<40,current<50
    intersection=int(np.count_nonzero((an&b)|(a&bn)))
    union=int(np.count_nonzero(a|b))
    if not union:
        raise InvalidArtifact('Write IoU is undefined for an empty mask union')
    iou=intersection/union
    return dict(success=iou>.8,iou=iou,intersection_pixels=intersection,union_pixels=union)


def _f32(value):
    value=np.asarray(value,dtype=np.float32)
    return np.where(abs(value)<np.finfo(np.float32).tiny,np.float32(0),value)


def _outward(value,up):
    """Round a float64 endpoint outward to float32, including FTZ zero."""
    rounded=np.asarray(value,dtype=np.float32)
    wrong=(rounded<value) if up else (rounded>value)
    result=np.where(wrong,np.nextafter(rounded,np.float32(np.inf if up else -np.inf)),rounded)
    return _f32(result)


def _divide_bounds(numerator):
    # Fixed, normal denominator. PTX permits at most 2 ULP error; use the
    # larger neighboring spacing at exponent boundaries, with outward rounding.
    exact=np.asarray(numerator,dtype=np.float64)/float(np.float32(64/.21))
    rounded=exact.astype(np.float32)
    magnitude=abs(rounded)
    ulp=(np.nextafter(magnitude,np.float32(np.inf))-magnitude).astype(np.float64)
    error=2*ulp
    return _outward(exact-error,False),_outward(exact+error,True)


def _square_bounds(lo,hi):
    near=np.where((lo<=0)&(hi>=0),np.float32(0),np.minimum(abs(lo),abs(hi)))
    far=np.maximum(abs(lo),abs(hi))
    return _f32(near*near),_f32(far*far)


def height_bounds(points):
    """Conservative integer image endpoints, not a per-pixel mm tolerance.

    All ordinary operations are monotonic float32/FTZ operations on interval
    endpoints. Squaring handles intervals crossing zero. A sphere contributes
    to the lower image only when z_squared is certainly positive, and to the
    upper image whenever it may be positive. Atomic maximum preserves bounds.
    """
    points=np.asarray(points,dtype=np.float32)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all() or np.any(abs(points)>10):
        raise InvalidArtifact('Invalid or out-of-range Write raster points')
    q=_f32(_f32(points)+np.array([.105,.105,0.],dtype=np.float32))
    start=_f32(_f32(q[:,:2]*np.float32(64/.21))-np.float32(2.))
    # Use double only for this exact half-away-from-zero integer conversion.
    start=(np.sign(start)*np.floor(abs(start.astype(np.float64))+.5)).astype(np.int32)
    rlo,rhi=_divide_bounds(2.)
    r2lo,r2hi=_square_bounds(rlo,rhi)
    gridlo,gridhi=_divide_bounds(np.arange(64))
    lower=np.zeros((64,64),dtype=np.int32);upper=lower.copy()
    for i in range(5):
        for j in range(5):
            gx,gy=start[:,0]+i,start[:,1]+j
            valid=(gx>=0)&(gx<64)&(gy>=0)&(gy<64)
            gx,gy=gx[valid],gy[valid];p=q[valid]
            x2lo,x2hi=_square_bounds(_f32(gridlo[gx]-p[:,0]),_f32(gridhi[gx]-p[:,0]))
            y2lo,y2hi=_square_bounds(_f32(gridlo[gy]-p[:,1]),_f32(gridhi[gy]-p[:,1]))
            zlo=_f32(_f32(r2lo-x2hi)-y2hi)
            zhi=_f32(_f32(r2hi-x2lo)-y2lo)
            for image,z2,up in [(lower,zlo,False),(upper,zhi,True)]:
                contributes=z2>0
                # Published maximum relative error is 2^-23. Float64 evaluates
                # the bound; outward rounding includes all permitted f32 values.
                root=np.sqrt(z2[contributes].astype(np.float64))
                root=_outward(root*(1+(1 if up else -1)*2**-23),up)
                height=_f32(_f32(root+p[contributes,2])*np.float32(1000.))
                np.maximum.at(image,(gy[contributes],gx[contributes]),height.astype(np.int32))
    return lower,upper


def compare_bounds(goal_bounds,current_bounds):
    glo,ghi=goal_bounds;clo,chi=current_bounds
    a,an=ghi<40,ghi<50;b,bn=chi<40,chi<50
    ap,anp=glo<40,glo<50;bp,bnp=clo<40,clo<50
    imin=int(np.count_nonzero((an&b)|(a&bn)))
    imax=int(np.count_nonzero((anp&bp)|(ap&bnp)))
    umin=int(np.count_nonzero(a|b));umax=int(np.count_nonzero(ap|bp))
    if not umin:
        raise InvalidArtifact('Write IoU union may be empty under GPU arithmetic bounds')
    lo,hi=imin/umax,min(1.,imax/umin)
    if lo<=.8<hi:
        raise InvalidArtifact(f'Write success is ambiguous under GPU arithmetic bounds: [{lo}, {hi}]')
    return dict(success=lo>.8,iou_bounds=[lo,hi])


def outcome(state):
    goal=height_bounds(np.asarray(state['task_state']).reshape(-1,3))
    current=height_bounds(state['x'])
    uncertain={}
    for key,(lo,hi) in [('goal_height_mm',goal),('current_height_mm',current)]:
        image=np.asarray(state[key])
        if image.shape!=(64,64) or not np.isfinite(image).all() or np.any(image!=np.floor(image)) or np.any(image<lo) or np.any(image>hi):
            raise InvalidArtifact(f'Write {key} disagrees with independent particle rasterization bounds')
        uncertain[key]=int(np.count_nonzero(lo!=hi))
    bounded=compare_bounds(goal,current)
    actual=compare_images(state['goal_height_mm'],state['current_height_mm'])
    if actual['success']!=bounded['success']:
        raise InvalidArtifact('Write recorded image score contradicts independent score bounds')
    return {**actual,**bounded,'rasterization_verified':True,
            'rasterization_verification':'PTX arithmetic interval inclusion; success must be unambiguous',
            'rasterization_exact':not any(uncertain.values()),'uncertain_height_pixels':uncertain}

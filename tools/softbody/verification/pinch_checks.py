"""Independent Pinch task audit from particle/goal arrays; no simulator imports.

The original CUDA kernel returns squared nearest-point distance. The task then
squares those values, averages in float32 and takes a float64 fourth root.
Bounds below cover ordinary IEEE float32 arithmetic and even sequential (rather
than NumPy's more accurate pairwise) accumulation. These bounds decide only the
task label; they are not tolerances for physical trajectory comparisons.
"""
import numpy as np
from scipy.spatial import cKDTree
class InvalidArtifact(ValueError):
    """Malformed numeric verification artifact."""


def directed_fourth_norm_bounds(a, b):
    a=np.asarray(a);b=np.asarray(b)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1:]!=(3,) or b.shape[1:]!=(3,) or not len(a) or not len(b):
        raise InvalidArtifact('Pinch point clouds must be nonempty Nx3 arrays')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or max(np.max(abs(a)),np.max(abs(b)))>1000:
        raise InvalidArtifact('Pinch audit requires finite metric coordinates within 1000m')
    if not np.array_equal(a,a.astype(np.float32)) or not np.array_equal(b,b.astype(np.float32)):
        raise InvalidArtifact('Pinch points must be exactly representable as float32')
    a=a.astype(np.float64);b=b.astype(np.float64)
    indices=cKDTree(b).query(a,k=1,eps=0)[1]
    squared=np.sum((a-b[indices])**2,axis=1)
    if np.any((squared>0)&(squared<1e-16)):
        raise InvalidArtifact('Pinch audit does not certify distances in the quartic underflow range')
    fourth_mean=np.mean(squared*squared,dtype=np.float64)
    exact=float(fourth_mean**.25)
    # At most 3 subtractions, 3 products, 2 additions for squared norm;
    # square, n-1 reduction additions and one divide. 32 extra rounding
    # factors cover the repeated use of the norm error and float64 host work.
    u=2.**-24;operations=len(a)+32
    if operations*u>=.01:raise InvalidArtifact('Pinch audit particle count exceeds arithmetic bound')
    gamma=operations*u/(1-operations*u)
    lower=np.nextafter(exact*(1-gamma)**.25,-np.inf)
    upper=np.nextafter(exact*(1+gamma)**.25,np.inf)
    if exact==0:lower=upper=0.
    return dict(value=exact,lower=float(lower),upper=float(upper),rounding_factor=gamma)


def outcome(state):
    x=state['x'];task=state['task_state'];n=len(x)
    if task.shape!=(2+3*n,) or not np.isfinite(task).all():
        raise InvalidArtifact('Pinch goal and deformation-distance state required')
    initial=task[:2];goal=task[2:].reshape(n,3)
    if np.any(initial<0) or initial.sum()<=0:raise InvalidArtifact('Invalid initial deformation distance')
    directed=[directed_fourth_norm_bounds(x,goal),directed_fourth_norm_bounds(goal,x)]
    lo=sum(v['lower'] for v in directed);hi=sum(v['upper'] for v in directed)
    total=float(sum(initial));target=.3*total
    if hi<target:success=True
    elif lo>=target:success=False
    else:raise InvalidArtifact('Pinch success threshold falls inside the arithmetic uncertainty interval')
    return dict(success=success,directed_fourth_norms=directed,distance_lower_m=lo,distance_upper_m=hi,
        target_distance_m=target,progress_lower=1-hi/total,progress_upper=1-lo/total,
        metric='Sum of directed fourth norms of nearest-point distances; strict threshold',
        arithmetic_verified=True,calibrated_physics_parity=False)

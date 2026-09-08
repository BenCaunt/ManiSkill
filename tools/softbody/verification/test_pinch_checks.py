import numpy as np
import pytest
from .pinch_checks import InvalidArtifact
from .pinch_checks import directed_fourth_norm_bounds,outcome


def state(x,goal,total):
    return dict(x=np.asarray(x,dtype=np.float32),task_state=np.r_[[total/2,total/2],np.asarray(goal,dtype=np.float32).reshape(-1)])


def test_fourth_norm_is_not_mean_or_rms_distance():
    x=np.array([[0,0,0],[3,4,0]],dtype=np.float32)
    b=directed_fourth_norm_bounds(x,np.zeros((1,3),dtype=np.float32))
    expected=(625/2)**.25
    assert b['lower']<expected<b['upper']
    assert b['lower']>np.sqrt(25/2)>2.5


def test_success_and_failure_clear_of_threshold():
    assert outcome(state([[0,0,0]],[[0,0,0]],1))['success']
    assert not outcome(state([[0,0,0]],[[3,4,0]],1))['success']
    assert outcome(state([[0,0,0]],[[3,4,0]],40))['success']


def test_threshold_ambiguity_is_rejected_not_rounded_to_success():
    with pytest.raises(InvalidArtifact,match='uncertainty'):
        outcome(state([[0,0,0]],[[3,4,0]],10/.3))


def test_points_need_exact_float32_representation():
    with pytest.raises(InvalidArtifact,match='float32'):
        directed_fourth_norm_bounds(np.array([[.1,0,0]]),np.zeros((1,3)))


def test_nonfinite_and_tiny_distance_are_not_certified():
    with pytest.raises(InvalidArtifact,match='finite'):
        directed_fourth_norm_bounds(np.array([[np.nan,0,0]]),np.zeros((1,3)))
    with pytest.raises(InvalidArtifact,match='underflow'):
        directed_fourth_norm_bounds(np.array([[1e-10,0,0]],dtype=np.float32),np.zeros((1,3),dtype=np.float32))


def test_l4_bounds_cover_explicit_float32_reduction():
    rng=np.random.RandomState(15)
    a=rng.uniform(-.1,.1,(73,3)).astype(np.float32);b=rng.uniform(-.1,.1,(47,3)).astype(np.float32)
    d=a[:,None,:]-b[None,:,:]
    squared=np.sum(d*d,axis=-1,dtype=np.float32).min(1)
    actual=float(np.mean(squared*squared,dtype=np.float32))**.25
    bound=directed_fourth_norm_bounds(a,b)
    assert bound['lower']<=actual<=bound['upper']


# The host trace-audit integration test stays in softbody_lab. This independent
# package tests forged reported progress in test_gpu_pinch_batch_checks.py.

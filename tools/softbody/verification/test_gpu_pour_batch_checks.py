"""Synthetic task/geometry faults; not native Pour evidence."""
import itertools
import numpy as np
import pytest
from tools.softbody.verification.gpu_pour_batch_checks import task_metrics,grasped,hull_error,check_ring


def test_original_fill_and_quiet_joint_predicates_and_dense_reward():
    x=np.tile([0.,0.,.012],(101,1));identity=np.eye(4)
    tcp=identity.copy();tcp[:3,3]=[.2,0,.05]
    model=dict(target_radius=.04,target_height=.08,source_aabb=np.array([[-.02,-.02,0],[.02,.02,.1]]),target_aabc=[0,0,.04,0,.08])
    args=[x,np.zeros(9),identity,identity,tcp,identity,identity,[np.zeros(3),np.zeros(3)],[.01,.014],model]
    out=task_metrics(*args);assert out['success'] and out['reward']==15
    np.testing.assert_array_equal(out['counts'],[101,0,0,101])
    args[1][0]=.1;out=task_metrics(*args)
    assert not out['success']
    assert out['reward']==pytest.approx(.101-np.linalg.norm([.2,0,.017]))
    args[0]=np.r_[x,[[.1,0,-.01]]];out=task_metrics(*args)
    assert out['counts'][2]==1  # Source predicate intentionally has no floor lower bound.


def test_grasp_requires_both_contacts_in_their_finger_directions():
    matrices=[np.eye(4),np.eye(4)]
    assert grasped([np.array([0,.01,0]),np.array([0,-.01,0])],matrices)
    assert not grasped([np.array([.01,0,0]),np.array([0,-.01,0])],matrices)
    assert not grasped([np.zeros(3),np.array([0,-.01,0])],matrices)


def test_native_convex_check_ignores_vertex_order_but_rejects_changed_geometry():
    cube=np.array(list(itertools.product([-1.,1.],repeat=3)))*.01
    assert hull_error(cube[::-1],cube)==0
    assert hull_error(cube+[.005,0,0],cube)==pytest.approx(.005)


def test_ring_pixels_and_pose_must_follow_selected_physical_target():
    frame={'target_ring_id':np.array(301),'task_visual_ids':np.array([401,402]),
        'before/checkpoint/task/fill_heights':np.array([[.02,.024],[.01,.014]]),
        'before/checkpoint/actors/target_beaker':np.array([[0.,0,0],[1.,0,0]]),
        'target_ring_heights':np.array([.01,.014]),'target_ring_pose':np.eye(4),
        'segmentation':np.array([[[[301],[401],[402]]]]),'position':np.array([[[[1040,0,12],[0,0,0],[0,0,0]]]]),
        'cam2world_gl':np.eye(4)[None],'target_radius':np.array(.04)}
    frame['target_ring_pose'][:3,3]=[1,0,.01]
    protocol=dict(initial_pose_limit=1e-6,minimum_ring_pixels=1,minimum_container_pixels=1,ring_surface_limit_m=.003)
    assert not check_ring(frame,1,protocol)['failures']
    frame['position'][0,0,0,0]=40
    assert 'Ring pixels differ' in check_ring(frame,1,protocol)['failures'][0]
    frame['target_ring_pose'][0,3]=0
    assert any('Ring pose differs' in f for f in check_ring(frame,1,protocol)['failures'])

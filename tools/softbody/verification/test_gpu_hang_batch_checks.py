"""Synthetic faults for the host verifier; no native physics claim."""
from copy import deepcopy
import numpy as np
import pytest

from tools.softbody.verification.gpu_hang_batch_checks import check_task_snapshot, rod_recipe


@pytest.fixture
def records():
    x=np.array([[0,-.06,.15],[0,-.03,.19],[0,0,.23],[0,.03,.19],[0,.06,.15]],np.float32)
    poses=np.array([[0,0,.2,1,0,0,0],[1,0,.2,1,0,0,0]],np.float32)
    matrices=np.repeat(np.eye(4,dtype=np.float32)[None],2,axis=0);matrices[:,:3,3]=poses[:,:3]
    data={'counts':np.array([5,5]),'state/task/selected_indices':np.tile(np.arange(5),(2,1)),
        'rod_pose':poses,'rod_matrix':matrices,'reported/target':poses.copy(),
        'hand_pose':poses.copy(),'leftfinger_pose':poses.copy(),'rightfinger_pose':poses.copy(),
        'qpos':np.zeros((2,9),np.float32),'reported/success':np.array([True,False]),
        'reported/reward':np.array([6.,4.45],np.float32)}
    data['hand_pose'][:,2]=.23
    data['leftfinger_pose'][:,0]-=[.04,.01];data['rightfinger_pose'][:,0]+=[.04,.01]
    data['qpos'][:,-2:]=[[.04,.04],[.01,.01]]
    for i in range(2):
        data[f'actual/{i}/x']=x+np.array([i,0,0],np.float32)
        data[f'actual/{i}/v']=np.zeros_like(x)
    model={'robot_joint_limits':np.broadcast_to(np.array([-.04,.04],np.float32),(2,9,2))}
    return data,model


def test_independent_release_outcome_and_reward(records):
    assert not check_task_snapshot(*records,1e-6)


@pytest.mark.parametrize('fault,expected',[('particles','reward'),('rod','Target observation'),
    ('label','success'),('reward','reward'),('gripper','reward')])
def test_cross_environment_or_corrupt_task_telemetry_is_rejected(records,fault,expected):
    data,model=deepcopy(records)
    if fault=='particles':data['actual/1/x']=data['actual/0/x'].copy()
    if fault=='rod':data['rod_pose'][1]=data['rod_pose'][0]
    if fault=='label':data['reported/success'][1]=True
    if fault=='reward':data['reported/reward'][1]=6
    if fault=='gripper':data['qpos'][1,-2:]=data['qpos'][0,-2:]
    assert any(expected in f for f in check_task_snapshot(data,model,1e-6))


def test_invalid_goal_particle_cannot_be_silently_ignored(records):
    data,model=records;data['state/task/selected_indices'][1,4]=5
    with pytest.raises(ValueError,match='evaluation indices'):check_task_snapshot(data,model,1e-6)


def test_seeded_rods_are_distinct_and_use_original_geometry():
    poses=np.array([rod_recipe(seed) for seed in (101,17)])
    assert not np.array_equal(*poses)
    assert np.all((np.linalg.norm(poses[:,:2],axis=1)>=.2)&(np.linalg.norm(poses[:,:2],axis=1)<.23))
    assert np.all((poses[:,2]>=.2)&(poses[:,2]<.3))
    np.testing.assert_allclose(np.linalg.norm(poses[:,3:],axis=1),1,atol=1e-7)

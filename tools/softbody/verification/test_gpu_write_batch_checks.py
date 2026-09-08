"""Fault controls for independent Write score, reward and reference checks."""
import copy

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from .write_checks import InvalidArtifact
from .gpu_write_batch_checks import check_reference_initial,check_task_snapshot,load_goals
from .job_archive import file_hash
from .write_checks import compare_images,height_image


@pytest.fixture
def snapshot():
    xy=np.stack(np.meshgrid(np.linspace(-.105,.102,64),np.linspace(-.105,.102,64)),axis=-1).reshape(-1,2)
    goals=np.stack([np.resize(np.column_stack((xy,np.where(abs(xy[:,i])<.018,.025,.055))),(19404,3)) for i in range(2)]).astype(np.float32)
    current=[goals[0].copy(),goals[0,:4096:3].copy()]
    pose=np.array([[0.,0.,.08,1.,0.,0.,0.],[.05,.03,.11,1.,0.,0.,0.]],np.float32)
    pose[1,3:]=Rotation.from_euler('y',.3).as_quat()[[3,0,1,2]]
    matrices=np.repeat(np.eye(4,dtype=np.float32)[None],2,axis=0)
    matrices[:,:3,3]=pose[:,:3];matrices[:,:3,:3]=Rotation.from_quat(pose[:,[4,5,6,3]]).as_matrix()
    data={'counts':np.array([len(x) for x in current]),'state/task/goal_points':goals,
        'goal_height_mm':np.stack([height_image(g) for g in goals]),
        'current_height_mm':np.stack([height_image(x) for x in current]),
        'tcp_pose':pose,'tcp_matrix':matrices,'reported/tcp_pose':pose.copy(),
        'tcp_gpu_indices':np.array([1,3]),'native_rigid':np.zeros((5,13),np.float32)}
    data['native_rigid'][[1,3],:7]=pose
    values=[compare_images(g,c) for g,c in zip(data['goal_height_mm'],data['current_height_mm'])]
    data.update({'reported/iou':np.array([v['iou'] for v in values],np.float32),
        'reported/success':np.array([v['success'] for v in values]),
        'reported/goal':np.clip(data['goal_height_mm'][:,:,::-1],0,255).astype(np.uint8)})
    rewards=[]
    for i,x in enumerate(current):
        data[f'actual/{i}/x']=x
        end=matrices[i,:3,3]+.02*matrices[i,:3,2]
        distance=np.sqrt(np.sum((x-end)**2,axis=1)).min()
        # Independent fixture construction for these known Y-axis rotations.
        angle=np.arcsin(abs(matrices[i,0,2]))
        rewards.append(values[i]['iou']+.1*(1-np.tanh(10*distance))+.1*(1-angle))
    data['reported/reward']=np.asarray(rewards,np.float32)
    return data


def check(data):return check_task_snapshot(data,2e-6,1e-6)


def test_distinct_goals_and_reduced_live_particles_verify(snapshot):
    result=check(snapshot)
    assert not result['failures']
    assert [m['success'] for m in result['measurements']]==[True,False]
    assert all(m['rasterization_verified'] for m in result['measurements'])


@pytest.mark.parametrize('field',['reported/reward','reported/iou','reported/success','reported/goal','reported/tcp_pose'])
def test_swapped_reported_rows_fail(snapshot,field):
    snapshot[field]=snapshot[field][::-1].copy()
    assert check(snapshot)['failures']


@pytest.mark.parametrize('field',['goal_height_mm','current_height_mm','state/task/goal_points'])
def test_swapped_raster_or_goal_rows_fail(snapshot,field):
    snapshot[field]=snapshot[field][::-1].copy()
    with pytest.raises(InvalidArtifact):check(snapshot)


def test_one_millimeter_raster_change_has_no_blanket_tolerance(snapshot):
    snapshot['current_height_mm'][0,0,0]+=1
    with pytest.raises(InvalidArtifact):check(snapshot)


def test_padding_cannot_stand_in_for_live_particles(snapshot):
    snapshot['actual/1/x']=np.r_[snapshot['actual/1/x'],np.zeros((500,3),np.float32)]
    with pytest.raises(ValueError,match='live Write particle count'):check(snapshot)


def test_forged_tcp_matrix_fails(snapshot):
    snapshot['tcp_matrix'][0,0,3]+=.01
    assert any('matrix' in f for f in check(snapshot)['failures'])


def test_wrong_native_tcp_row_fails(snapshot):
    snapshot['tcp_gpu_indices']=snapshot['tcp_gpu_indices'][::-1].copy()
    assert any('GPU buffer' in f for f in check(snapshot)['failures'])


def test_duplicate_native_tcp_ownership_fails(snapshot):
    snapshot['tcp_gpu_indices'][:]=1
    with pytest.raises(ValueError,match='ownership'):check(snapshot)


def test_reference_check_detects_shared_implementation_reset_changes(snapshot):
    old={'x':snapshot['actual/0/x'].copy(),'task_state':snapshot['state/task/goal_points'][0].reshape(-1).copy()}
    n=len(old['x'])
    for key,shape in [('v',(n,3)),('F',(n,3,3)),('C',(n,3,3)),('vc',(n,)),('mass',(n,))]:
        old[key]=np.ones(shape,np.float32)
        snapshot['actual/0/'+key]=old[key].copy()
    for key in ('qpos','qvel'):
        old[key]=np.zeros(7,np.float32);snapshot[key]=np.zeros((2,7),np.float32)
    material={'particle_mass':np.ones(n,np.float32)}
    snapshot['state/mpm_material/particle_mass']=np.ones((2,n+100),np.float32)
    assert not check_reference_initial(snapshot,0,old,material)
    for key in ('actual/0/x','qpos','state/task/goal_points','state/mpm_material/particle_mass'):
        broken=copy.deepcopy(snapshot);broken[key].flat[0]+=.001
        assert check_reference_initial(broken,0,old,material),key


def test_pinned_goal_directory_can_include_notices(tmp_path,snapshot):
    path=tmp_path/'goal.h5'
    with h5py.File(path,'w') as f:f['goal']=snapshot['state/task/goal_points'][0]
    notice=tmp_path/'NOTICE.txt';notice.write_text('Unit fixture provenance')
    files={p.name:file_hash(p) for p in (path,notice)}
    goals=load_goals(tmp_path,files)
    assert set(goals)=={'goal.h5'}
    assert np.array_equal(goals['goal.h5'],snapshot['state/task/goal_points'][0])
    with h5py.File(path,'r+') as f:f['goal'][0,0]=.003
    with pytest.raises(ValueError,match='changed'):load_goals(tmp_path,files)


def test_hashed_hdf5_cannot_redirect_to_unpinned_goal(tmp_path):
    path=tmp_path/'goal.h5'
    with h5py.File(path,'w') as f:f['goal']=h5py.ExternalLink('untracked.h5','/goal')
    with pytest.raises(ValueError,match='inline'):load_goals(tmp_path,{path.name:file_hash(path)})

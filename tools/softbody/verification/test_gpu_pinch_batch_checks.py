import copy
import json

import numpy as np
import pytest

from tools.softbody.verification.gpu_pinch_batch_checks import CAMERA_SHAPES, check_task_snapshot, check_level_goal, load_reference_inputs, check_initial_root
from tools.softbody.verification.job_archive import file_hash


@pytest.fixture
def snapshot():
    counts=np.array([8,12]);capacity=16
    data={'counts':counts,'state/task_particles/goal':np.zeros((2,capacity,3),np.float32),
        'state/task/deformed_distance':np.array([[.02,.03],[.02,.03]]),
        'state/task/goal_depths':np.zeros((2,4,128,128),np.float32),
        'state/task/goal_rgbs':np.zeros((2,4,128,128,3),np.uint8),
        'state/task/goal_cam_pos':np.zeros((2,4,3)),
        'state/task/goal_cam_rot':np.tile([1.,0.,0.,0.],(2,4,1)),
        'state/task/goal_cam_intrinsic':np.tile([[100.,0.,64.],[0.,100.,64.],[0.,0.,1.]],(2,1,1)),
        'reported/target_points':np.zeros((2,65536,4),np.float32),
        'tcp_pose':np.array([[0.,0.,.1,1.,0.,0.,0.],[.08,0.,.2,1.,0.,0.,0.]],np.float32),
        'tcp_matrix':np.repeat(np.eye(4,dtype=np.float32)[None],2,axis=0),
        'tcp_gpu_indices':np.array([1,3]),'native_rigid':np.zeros((5,13),np.float32)}
    data['tcp_matrix'][:,:3,3]=data['tcp_pose'][:,:3]
    data['native_rigid'][[1,3],:7]=data['tcp_pose'];data['reported/tcp_pose']=data['tcp_pose'].copy()
    data['state/task/goal_depths'][:,:,64,64]=.25
    data['state/task/goal_rgbs'][1]=127
    data['state/task/goal_cam_pos'][1,:,0]=.1
    for i,n in enumerate(counts):
        x=np.column_stack([np.arange(n)*.05,np.zeros(n),np.full(n,.05)]).astype(np.float32)
        data[f'actual/{i}/x']=x
        data['state/task_particles/goal'][i,:n]=x+[0.,i*.01,0.]
        data['reported/target_points'][i,:4]=np.array([.25+float(np.float32(i*.1)),-.00125,-.00125,1.],np.float32)
    distance=float(np.float32(.01))
    data['reported/chamfer']=np.array([[0.,0.],[distance,distance]])
    data['reported/success']=np.array([True,False]);data['reported/progress']=1-data['reported/chamfer'].sum(1)/.05
    data['reported/reward']=np.array([-100*sum(data['reported/chamfer'][i])+.1*(1-np.tanh(
        10*np.linalg.norm(data[f'actual/{i}/x']-(data['tcp_pose'][i,:3]+np.array([0,0,.02],np.float32)),axis=1).min()))+.1 for i in range(2)],np.float32)
    data['reported/target_rgb']=data['state/task/goal_rgbs'].copy()
    data['reported/target_depth']=data['state/task/goal_depths'].copy()
    return data


def test_distinct_goals_and_heterogeneous_counts_verify(snapshot):
    result=check_task_snapshot(snapshot)
    assert not result['failures']
    assert [m['success'] for m in result['measurements']]==[True,False]


@pytest.mark.parametrize('field',['reported/progress','reported/success','reported/chamfer','reported/reward',
                                 'reported/target_rgb','reported/target_points','reported/tcp_pose'])
def test_swapped_rows_cannot_pass(snapshot,field):
    snapshot[field]=snapshot[field][::-1].copy()
    assert check_task_snapshot(snapshot)['failures']


def test_goal_padding_is_not_scored(snapshot):
    snapshot['state/task_particles/goal'][0,-1]=[.1,0,0]
    with pytest.raises(ValueError,match='zero padding'):check_task_snapshot(snapshot)


def test_wrong_live_count_fails(snapshot):
    snapshot['counts'][1]-=1
    with pytest.raises(ValueError,match='live particle counts'):check_task_snapshot(snapshot)


def test_wrong_native_tcp_ownership_is_detected(snapshot):
    snapshot['tcp_gpu_indices']=snapshot['tcp_gpu_indices'][::-1].copy()
    assert any('GPU buffer' in f for f in check_task_snapshot(snapshot)['failures'])


def test_camera_pose_change_cannot_preserve_old_projected_goal(snapshot):
    snapshot['state/task/goal_cam_pos'][0,0,0]+=.001
    assert any('projection' in f for f in check_task_snapshot(snapshot)['failures'])


def test_l2_sum_cannot_replace_the_two_directed_fourth_norms(snapshot):
    snapshot['reported/chamfer'][1]*=np.sqrt(12)
    assert any('arithmetic bounds' in f for f in check_task_snapshot(snapshot)['failures'])


def test_forged_progress_fails_even_when_success_remains_correct(snapshot):
    snapshot['reported/progress'][1]+=.01
    failures=check_task_snapshot(snapshot)['failures']
    assert any('progress' in f for f in failures)
    assert not any('success' in f for f in failures)


@pytest.fixture
def roots():
    physical=np.zeros((2,31),np.float32)
    physical[:,3]=1;physical[1,0]=.1;physical[1,7]=.02
    data={'counts':np.array([8,12]),'state/articulations/panda':physical,
          'controller/root':physical.copy()}
    old={'root_pose':physical[1:2,:7].copy(),'root_velocity':physical[1:2,7:13].copy()}
    return data,old


def test_initial_root_matches_own_physical_row(roots):
    data,old=roots
    assert not check_initial_root(data,1,old)
    assert len(check_initial_root(data,0,old))==2


@pytest.mark.parametrize('component',range(13))
def test_initial_root_rejects_one_bit_change_despite_unchanged_controller(roots,component):
    data,old=roots;array=data['state/articulations/panda']
    array[1,component]=np.nextafter(array[1,component],np.float32(1)) if array[1,component]!=1 else np.nextafter(np.float32(1),np.float32(2))
    assert check_initial_root(data,1,old)


@pytest.mark.parametrize('bad',['missing','multiple','shape','dtype','nonfinite','reference'])
def test_invalid_initial_root_evidence_is_rejected(roots,bad):
    data,old=roots;key='state/articulations/panda'
    if bad=='missing':del data[key]
    if bad=='multiple':data['state/articulations/another']=data[key].copy()
    if bad=='shape':data[key]=data[key][:,:30]
    if bad=='dtype':data[key]=data[key].astype(np.float64)
    if bad=='nonfinite':data[key][1,0]=np.nan
    if bad=='reference':old['root_pose']=old['root_pose'][0]
    with pytest.raises(ValueError):check_initial_root(data,1,old)


@pytest.mark.parametrize('bad',['depth','intrinsic','quaternion'])
def test_invalid_camera_rejected_before_projection(snapshot,bad):
    if bad=='depth':snapshot['state/task/goal_depths'][0,0,0,0]=-1
    if bad=='intrinsic':snapshot['state/task/goal_cam_intrinsic'][0]=0
    if bad=='quaternion':snapshot['state/task/goal_cam_rot'][0,0]=0
    with pytest.raises(ValueError,match='camera goal'):check_task_snapshot(snapshot)


@pytest.mark.parametrize('limit',[float('nan'),float('inf'),-1])
def test_invalid_projection_limit_rejected(snapshot,limit):
    with pytest.raises(ValueError,match='verification limits'):check_task_snapshot(snapshot,projection_limit=limit)


def test_pinned_goal_check_detects_changed_particles_and_metadata(snapshot):
    level={k:snapshot['state/task/'+k][1].copy() for k in CAMERA_SHAPES}
    level['goal']=snapshot['state/task_particles/goal'][1,:12].copy()
    level['goal_points_observation']=snapshot['reported/target_points'][1].copy()
    assert not check_level_goal(snapshot,1,level,1e-7)
    level['goal'][0,0]+=.01;level['goal_rgbs'][0,0,0,0]+=1
    failures=check_level_goal(snapshot,1,level,1e-7)
    assert len(failures)==2


@pytest.fixture
def pinned_inputs(tmp_path):
    pack=tmp_path/'pack';baseline=tmp_path/'baseline';(pack/'levels').mkdir(parents=True);baseline.mkdir()
    (pack/'export.json').write_text('{}')
    np.savez(pack/'levels/a.npz',goal=np.zeros((2,3),np.float32))
    digest=file_hash(pack/'levels/a.npz')
    (pack/'levels/levels.json').write_text(json.dumps({'levels':{'a.h5':{'file':'a.npz','sha256':digest,'source_sha256':'old'}}}))
    np.savez(baseline/'snapshot.npz',x=np.zeros((2,3),np.float32))
    np.savez(baseline/'material.npz',particle_mass=np.ones(2,np.float32))
    config={'pack_files':{str(p.relative_to(pack)):file_hash(p) for p in pack.rglob('*') if p.is_file()},
        'baseline':{'files':{p.name:file_hash(p) for p in baseline.iterdir()},'cases':[
            dict(seed=101,level_file='a.h5',control_mode='pd_joint_delta_pos',snapshot='snapshot.npz',material='material.npz')]}}
    return config,baseline,pack


def test_all_consumed_baselines_and_levels_are_pinned(pinned_inputs):
    _,records,levels,refs=load_reference_inputs(*pinned_inputs)
    assert list(records)==list(levels)==['a.h5']
    assert list(refs)==[(101,'a.h5','pd_joint_delta_pos')]


@pytest.mark.parametrize('missing',['snapshot','material','export','manifest','level'])
def test_unpinned_input_is_rejected_even_if_it_exists(pinned_inputs,missing):
    config,baseline,pack=pinned_inputs
    if missing in ('snapshot','material'):del config['baseline']['files'][missing+'.npz']
    else:del config['pack_files'][{'export':'export.json','manifest':'levels/levels.json','level':'levels/a.npz'}[missing]]
    with pytest.raises(ValueError,match='Unpinned'):load_reference_inputs(config,baseline,pack)


def test_reference_controller_identity_must_be_unique(pinned_inputs):
    config,baseline,pack=pinned_inputs
    config['baseline']['cases'].append(copy.deepcopy(config['baseline']['cases'][0]))
    with pytest.raises(ValueError,match='Duplicate'):load_reference_inputs(config,baseline,pack)


def test_pinned_file_mutation_is_rejected(pinned_inputs):
    config,baseline,pack=pinned_inputs;(baseline/'material.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='input changed'):load_reference_inputs(config,baseline,pack)

"""Synthetic task-oracle faults; not evidence of native dynamics."""
from copy import deepcopy

import numpy as np
import pytest

from tools.softbody.verification.gpu_excavate_batch_checks import check_task_snapshot, task_metrics, check_old_recipe


@pytest.fixture
def snapshot():
    x = [np.tile([0.,0.,.25],(100,1)),
         np.r_[np.tile([0.,0.,.05],(250,1)),np.tile([.2,0.,.05],(30,1))]]
    hull = np.array([[-.05,-.04,0.],[.05,.04,.1]])
    model = {'bucket_reward_hull':np.array([hull,hull])}
    poses = np.array([np.eye(4),np.eye(4)]); poses[:,:3,3] = [1.,0.,0.]
    data = dict(counts=np.array([100,280]),bucket_pose=poses)
    data.update({'state/task/target_num':np.array([[199.],[250.]]),
                 'reported/target':np.array([[199.],[250.]],np.float32),
                 'reported/success':np.array([True,False]),
                 'reported/lifted_particles':np.array([100,0]),
                 'reported/spilled_particles':np.array([0,30]),
                 'reported/reward':np.array([6.,-.3],np.float32)})
    for i in range(2):
        data[f'actual/{i}/x'] = x[i]; data[f'actual/{i}/v'] = np.zeros_like(x[i])
        data[f'reported/inside_bucket/{i}'] = np.empty((0,3))
    return data, model


def test_independent_outcomes_and_known_rewards(snapshot):
    assert not check_task_snapshot(*snapshot,1e-6)


@pytest.mark.parametrize('fault,expected',[
    ('label','predicate'),('target','Target observation'),('reward','dense reward'),('membership','bucket membership')])
def test_corrupted_task_claims_are_rejected(snapshot,fault,expected):
    data,model = deepcopy(snapshot)
    if fault == 'label': data['reported/success'][1] = True
    if fault == 'target': data['reported/target'][1] = 199
    if fault == 'reward': data['reported/reward'][1] = 6
    if fault == 'membership': data['reported/inside_bucket/1'] = data['actual/1/x'][:1]
    assert any(expected in f for f in check_task_snapshot(data,model,1e-6))


def test_bucket_planes_hull_and_translated_pose_have_independent_effects():
    points = np.array([[0.,0.,.02],[0.,-.02,.02],[.04,0.,.02],[0.,0.,.09]])
    hull = np.array([[-.05,-.04,0.],[.05,.04,.1]])
    result = task_metrics(points,np.zeros_like(points),250,np.eye(4),hull)
    assert np.array_equal(result['inside'],points[[0,2]])
    narrow = hull.copy(); narrow[:,0] *= .5
    assert np.array_equal(task_metrics(points,np.zeros_like(points),250,np.eye(4),narrow)['inside'],points[[0]])
    pose = np.eye(4); pose[:3,3] = [1.,0.,0.]
    shifted = points + [1.,0.,0.]
    assert np.array_equal(task_metrics(shifted,np.zeros_like(shifted),250,pose,hull)['inside'],shifted[[0,2]])


def test_old_source_comparison_checks_actual_particles_materials_and_target():
    old={k:np.zeros((2,3)) for k in ('x','v','F','C','vc','mass')}
    old.update(task_state=np.array([250]),qpos=np.zeros(7))
    checkpoint={'mpm_material/'+k:np.ones((1,2)) for k in ('particle_mass','particle_vol','particle_type','particle_mu_lam_ys','particle_friction_cohesion')}
    new={'actual/0/'+k:v.copy() for k,v in old.items() if k not in ('task_state','qpos')}
    new.update({'state/'+k:v.copy() for k,v in checkpoint.items()})
    new['state/task/target_num']=old['task_state'][None].copy();new['qpos']=old['qpos'][None].copy()
    assert not check_old_recipe(new,old,checkpoint)
    for field in ('actual/0/x','actual/0/mass','state/mpm_material/particle_vol','state/task/target_num'):
        changed=deepcopy(new); changed[field].flat[0] += 1
        assert check_old_recipe(changed,old,checkpoint)

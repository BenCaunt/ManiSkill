"""Synthetic verifier faults, not evidence of native IK or dynamics."""
from copy import deepcopy

import numpy as np
import pytest

from tools.softbody.verification.gpu_batch_checks import expected_actions
from tools.softbody.verification.gpu_controller_batch_checks import check_ik_step


@pytest.fixture
def records():
    case = dict(seeds=[101,17],control_mode='pd_ee_target_delta_pos',action_scales=[1.,-.5],controller_lifecycle=True)
    q = np.array([[.1]*7,[.3]*7],np.float32)
    action = expected_actions(case,q)
    previous = np.array([[.2,0.,.3,1.,0.,0.,0.],[-.1,.2,.4,1.,0.,0.,0.]])
    after = previous.copy(); after[:,:3] += action.astype(float)*.1
    model = dict(links=[8,9],masks=[[True]*7]*2)
    step = dict(qpos_before=q.tolist(),target_pose_before=previous.tolist(),ee_pose_before=previous.tolist(),
                target_pose_after=after.tolist(),target_qpos_after=q.tolist(),ik_success=[True,True],ik_calls=[])
    for i in range(2):
        step['ik_calls'].append(dict(index=i,link=model['links'][i],active_qmask=[True]*7,max_iterations=100,
            target_pose=after[i].tolist(),initial_qpos=q[i].tolist(),result=q[i].tolist(),success=True))
    return case, step, model, action


def test_independent_valid_rows_are_accepted(records):
    assert not check_ik_step(*records,1e-6)


@pytest.mark.parametrize('fault,expected', [
    ('initial','another environment'),('target','did not receive'),('result','own native IK solution'),
    ('missing','exactly once'),('failed','Native IK failed'),('nonfinite','Invalid batched IK telemetry')])
def test_native_ik_telemetry_faults_are_rejected(records, fault, expected):
    case, step, model, action = deepcopy(records)
    if fault == 'initial': step['ik_calls'][1]['initial_qpos'] = step['ik_calls'][0]['initial_qpos']
    if fault == 'target': step['ik_calls'][1]['target_pose'] = step['ik_calls'][0]['target_pose']
    if fault == 'result': step['ik_calls'][1]['result'] = step['ik_calls'][0]['result']
    if fault == 'missing': step['ik_calls'].pop()
    if fault == 'failed': step['ik_calls'][1]['success'] = False
    if fault == 'nonfinite': step['target_pose_after'][1][0] = float('nan')
    assert any(expected in f for f in check_ik_step(case,step,model,action,1e-6))


def test_absolute_joint_actions_use_each_actual_initial_configuration():
    case = dict(control_mode='pd_joint_pos',action_scales=[1.,-.5],controller_lifecycle=True)
    q = np.array([[.1]*7,[.3]*7],np.float32)
    action = expected_actions(case,q)
    assert action.shape == (2,7)
    np.testing.assert_allclose(action[:,0]-q[:,0],[.002,-.001],atol=2e-8)

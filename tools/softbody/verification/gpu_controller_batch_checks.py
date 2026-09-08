"""Independent batched controller/IK and reset lifecycle checks."""
import argparse
import json
from pathlib import Path

import numpy as np

from .gpu_batch_checks import evaluate, particle_errors


def quat_product(a, b):
    return np.r_[a[0]*b[0]-a[1:]@b[1:], a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:], b[1:])]


def rotate(q, p):
    q = q / np.linalg.norm(q)
    return quat_product(quat_product(q, np.r_[0., p]), np.r_[q[0], -q[1:]])[1:]


def target_pose(previous, action, mode):
    absolute = mode == 'pd_ee_pose'
    translation = action[:3] if absolute else np.clip(action[:3], -1., 1.) * .1
    vector = action[3:] if absolute else action[3:] * .1 / max(np.linalg.norm(action[3:]), 1.) if len(action) == 6 else np.zeros(3)
    angle = np.linalg.norm(vector)
    q = np.r_[np.cos(angle/2), vector * (np.sin(angle/2)/angle if angle else .5)]
    if absolute:return np.r_[translation,q]
    if mode.endswith('_align'):
        return np.r_[previous[:3]+translation, quat_product(q, previous[3:])]
    return np.r_[previous[:3]+rotate(previous[3:],translation), quat_product(previous[3:],q)]


def check_ik_step(case, step, model_info, action, tolerance):
    count = len(case['seeds']); failures = []
    dof = 9 if case.get('task') in ('Hang','Pour','Pinch') else 7
    before = np.asarray(step['qpos_before'])
    if before.shape != (count, dof) or not np.isfinite(before).all():
        return ['Invalid pre-action native joint state']
    if 'ee' not in case['control_mode']:
        return [] if not step['ik_calls'] else ['Joint controller unexpectedly ran IK']
    calls = step['ik_calls']
    if len(calls) != count or [c['index'] for c in calls] != list(range(count)):
        return ['Each environment must call its own native IK model exactly once']
    for key, shape in [('target_pose_before',(count,7)),('ee_pose_before',(count,7)),
                       ('target_pose_after',(count,7)),('target_qpos_after',(count,7)),('ik_success',(count,))]:
        value = np.asarray(step[key])
        if value.shape != shape or not np.isfinite(value).all():
            return ['Invalid batched IK telemetry: '+key]
    for index, call in enumerate(calls):
        if (call['link'] != model_info['links'][index] or call['active_qmask'] != model_info['masks'][index]
                or call['active_qmask'] != [True]*7+[False]*(dof-7) or call['max_iterations'] != 100):
            failures.append('Native IK link/mask/iteration contract changed')
        if not np.array_equal(call['initial_qpos'], step['qpos_before'][index]):
            failures.append('Native IK used another environment or stale initial qpos')
        previous = np.asarray(step['target_pose_before' if '_target_' in case['control_mode'] else 'ee_pose_before'][index])
        arm_width = 3 if case['control_mode'].endswith('_pos') else 6
        expected = target_pose(previous, action[index,:arm_width].astype(float), case['control_mode'])
        target = np.asarray(step['target_pose_after'][index])
        if np.max(np.abs(target-expected)) > tolerance:
            failures.append('EE target disagrees with independent legacy frame composition')
        native_target = np.asarray(call['target_pose'])
        if native_target.shape != (7,) or not np.isfinite(native_target).all() or np.max(np.abs(target-native_target)) > tolerance:
            failures.append('Native IK did not receive this environment target')
        if not call['success'] or not step['ik_success'][index]:
            failures.append('Native IK failed for a declared reachable action')
        result = np.asarray(call['result'], np.float32)
        if result.shape != (dof,) or not np.isfinite(result).all() or not np.array_equal(result[:7], np.asarray(step['target_qpos_after'][index],np.float32)):
            failures.append('Controller did not use its own native IK solution')
    return failures


def evaluate_controllers(root, protocol):
    root = Path(root)
    base = evaluate(root, protocol); failures = list(base['failures']); reports = {}
    for case in protocol['cases']:
        name = case['name']; result = json.loads((root/name/'result.json').read_text())
        if not result['complete']: continue
        count = len(case['seeds']); action = np.asarray(result['action'], np.float32)
        model_info = result.get('ik_models')
        if 'ee' in case['control_mode']:
            if (model_info['count'] != count or not model_info['distinct'] or not model_info['native_ownership']
                    or not result['rebuilt_ik_models']):
                failures.append(name+': missing independent native models or rebuilt model ownership')
        for step in result['steps']:
            failures.extend(name+'/'+step['label']+': '+f for f in check_ik_step(case,step,model_info,action,protocol['target_pose_limit']))
        def load(label):
            with np.load(root/name/(label+'.npz'),allow_pickle=False) as z: return {k:z[k] for k in z.files}
        predecessors = {'warmup-1':'initial','warmup-2':'warmup-1','expected':'warmup-2','continued':'expected',
                        'partial-stepped':'partial-restored','count-stepped':'partial-count','flat-stepped':'flat-restored'}
        for step in result['steps']:
            before = load(predecessors[step['label']])
            if not np.array_equal(before['qpos'], step['qpos_before']):
                failures.append(name+'/'+step['label']+': pre-action qpos differs from actual native checkpoint')
            if 'ee' in case['control_mode']:
                for key, field in [('ee_pose_before','ee_pose_at_base'),('target_pose_before','ee_target_pose')]:
                    if not np.array_equal(before[field], step[key]):
                        failures.append(name+'/'+step['label']+': '+key+' differs from independently hashed snapshot')
        replays = {}
        if count > 1:
            expected = load('expected')
            for label, indices in [('partial-stepped',[0]),('flat-stepped',list(range(count)))]:
                actual = load(label)
                for index in indices:
                    errors = particle_errors(expected,index,actual,index); replays[f'{label}/env{index}']=errors
                    if errors['qpos'] > protocol['joint_replay_limit']:
                        failures.append(name+f'/{label}/env{index}: joint replay exceeds existing lifecycle limit')
        reports[name] = dict(ik_requests=sum(len(s['ik_calls']) for s in result['steps']),replay_errors=replays)
    return dict(passed=not failures,failures=failures,reports=reports,lifecycle_and_rendering=base,
                scope=protocol['scope'],full_port_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('protocol',type=Path);parser.add_argument('output',type=Path)
    args = parser.parse_args(); result=evaluate_controllers(args.root,json.loads(args.protocol.read_text()))
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)

"""Independent Hang batch task equations, grasp and initial-state checks.

Task equations follow ManiSkill2 v0.5.3 hang_env.py; the source terms in
LEGACY-SOURCE-LICENSE.md apply. This verifier imports no simulator or candidate.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .gpu_controller_batch_checks import evaluate_controllers
from .job_archive import file_hash


def task_metrics(x, v, indices, rod, normal, hand, fingers, finger_qpos, width):
    directions = x[indices] - rod
    signs = np.sign(directions @ normal)
    opposite = signs[0] != signs[3]
    draped = opposite and signs[0] == signs[1] and signs[3] == signs[4]
    high_v = v[x[:,2] > rod[2]-.03]
    quiet = np.count_nonzero((high_v > -.05) & (high_v < .05))/(high_v.size+.001)
    success = bool(draped and rod[2] < x[:,2].max() < rod[2]+.05
        and directions[0,2] < 0 and directions[4,2] < 0 and x[:,2].min() > .03
        and quiet > .99 and np.linalg.norm(fingers[0]-fingers[1]) > .07)
    if success: return dict(success=True,reward=6.)
    reach = 1-np.tanh(10*np.linalg.norm(x-hand,axis=1).min())
    middle = x[indices[2]]
    center = .5*(1-np.tanh(10*np.linalg.norm(rod[:2]-middle[:2])))
    center += .5*(1-np.tanh(10*(rod[2]-middle[2]))) if rod[2]>=middle[2] else .5
    side = top = bottom = release = 0.
    if rod[2] < middle[2]:
        bottom = 1-np.tanh(10*max(0,.04-x[:,2].min()))
        side = .5*(int(opposite)+int(draped))
        if draped:
            top = .25*np.count_nonzero(directions[[0,1,3,4],2] < 0)
            reach = 1.
            if top > .9: release = finger_qpos.sum()/width
    return dict(success=False,reward=float(reach+center+side+top+release+.2*bottom))


def rod_recipe(seed):
    rng = np.random.RandomState(seed)
    radius = .2+rng.rand()*.03; angle = np.pi/4+rng.rand()*np.pi/2
    height = .2+rng.rand()*.1
    return np.array([np.sin(angle)*radius,np.cos(angle)*radius,height,
                     np.cos(angle/2),0.,0.,-np.sin(angle/2)],np.float32)


def check_task_snapshot(data, model, reward_limit):
    failures = []; count = len(data['counts'])
    indices = data['state/task/selected_indices']
    if (indices.shape != (count,5) or not np.isfinite(indices).all()
            or np.any(indices != np.floor(indices)) or np.any(indices < 0)
            or np.any(indices >= data['counts'][:,None])):
        raise ValueError('Invalid rope evaluation indices')
    if not np.array_equal(data['reported/target'],data['rod_pose']):
        failures.append('Target observation differs from own native rod pose')
    for i in range(count):
        values = task_metrics(data[f'actual/{i}/x'],data[f'actual/{i}/v'],indices[i].astype(int),
            data['rod_pose'][i,:3],data['rod_matrix'][i,:3,1],data['hand_pose'][i,:3],
            [data['leftfinger_pose'][i,:3],data['rightfinger_pose'][i,:3]],
            data['qpos'][i,-2:],model['robot_joint_limits'][i,-1,1]*2)
        if data['reported/success'].shape != (count,) or data['reported/success'][i] != values['success']:
            failures.append(f'env{i}: success differs from independent rope predicates')
        if data['reported/reward'].shape != (count,) or abs(data['reported/reward'][i]-values['reward'])>reward_limit:
            failures.append(f'env{i}: reward differs from independent rope equations')
    return failures


def check_old_recipe(new, old, checkpoint):
    failures = []
    for key in ('x','v','F','C','vc','mass'):
        if not np.array_equal(new['actual/0/'+key],old[key]):
            failures.append('N1 initial particles differ from frozen old source: '+key)
    for key in ('qpos','qvel','drive_position','drive_velocity'):
        if not np.array_equal(new[key][0],old[key]):
            failures.append('N1 initial robot/drives differ from frozen old source: '+key)
    if not np.array_equal(new['state/task/selected_indices'][0],old['task_state']):
        failures.append('N1 evaluation indices differ from frozen old source')
    for key,value in checkpoint.items():
        if key.startswith('mpm_material/') and not np.array_equal(new['state/'+key],value):
            failures.append('N1 materials differ from frozen old source: '+key)
    return failures


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def evaluate_hang(root, protocol, baseline_root, pack_root):
    root, baseline_root, pack_root = map(Path,(root,baseline_root,pack_root))
    base = evaluate_controllers(root,protocol)
    failures, reports, cases = list(base['failures']), {}, {}
    baseline = protocol['baseline']
    for name,digest in baseline['files'].items():
        if file_hash(baseline_root/name) != digest:raise ValueError('Old-source baseline hash mismatch: '+name)
    for name,digest in protocol['pack_files'].items():
        if file_hash(pack_root/name) != digest:raise ValueError('Pinned Hang pack hash mismatch: '+name)
    pack = json.loads((pack_root/'export.json').read_text()); starts=load(pack_root/'initial-states.npz')
    rod = next(x for x in pack['geometry'] if x['name']=='rod')
    for case in protocol['cases']:
        if case['task'] != 'Hang':continue
        name=case['name']; result=json.loads((root/name/'result.json').read_text())
        if not result['complete']:continue
        if result['reward_mode'] != 'dense':failures.append(name+': wrong native reward mode')
        data={f[:-4]:load(root/name/f) for f in result['files']};cases[name]=data
        model=data['model-initial'];n=len(case['seeds'])
        for field,value in [('rod_mass',rod['mass']),('rod_inertia',rod['inertia']),
            ('rod_com',rod['com']),('rod_half_size',[.3,.01,.01])]:
            expected=np.broadcast_to(np.asarray(value,np.float32),model[field].shape)
            if not np.array_equal(model[field],expected):failures.append(name+': original rod model changed: '+field)
        labels=[label for label,x in data.items() if 'counts' in x]
        for label in labels:
            failures.extend(name+'/'+label+': '+f for f in check_task_snapshot(data[label],model,protocol['reward_limit']))
        for step in result['steps']:
            if not np.array_equal(step['reward'],data[step['label']]['reported/reward']):
                failures.append(name+'/'+step['label']+': returned reward differs from measured state')
        for label in ('initial','reconfigured'):
            value=data[label]
            for i,seed in enumerate(case['seeds']):
                if np.max(np.abs(value['rod_pose'][i]-rod_recipe(seed)))>protocol['initial_pose_limit']:
                    failures.append(name+f'/{label}/env{i}: rod differs from original seeded recipe')
                grasp=int(value['recipe_grasp_index'][i])
                if not 0<=grasp<len(starts['mpm_x']):raise ValueError('Invalid recorded grasp index')
                for key in ('x','v','F','C','vc'):
                    if not np.array_equal(value[f'actual/{i}/'+key],starts['mpm_'+key][grasp]):
                        failures.append(name+f'/{label}/env{i}: initial rope differs from pinned recorded grasp: '+key)
                for key in ('qpos','qvel'):
                    if not np.array_equal(value[key][i],starts['robot_'+key][grasp]):
                        failures.append(name+f'/{label}/env{i}: initial robot differs from pinned recorded grasp: '+key)
        reports[name]=dict(task_snapshots=len(labels),initial_counts=data['initial']['counts'].tolist(),
                          grasp_indices=data['initial']['recipe_grasp_index'].tolist())
    for batch_name,index,single_name in protocol['initial_comparisons']:
        if batch_name not in cases or single_name not in cases:continue
        batch,single=cases[batch_name],cases[single_name]
        for key,value in single['model-initial'].items():
            if not np.array_equal(value[0],batch['model-initial'][key][index]):
                failures.append(batch_name+f'/env{index}: native model differs from N1: '+key)
        for label in ('initial','reconfigured'):
            for key,value in single[label].items():
                if key.startswith(('state/mpm/','state/mpm_material/')):
                    same=np.array_equal(value[0],batch[label][key][index,:single[label]['counts'][0]])
                elif key in ('qpos','qvel','drive_position','drive_velocity','counts','recipe_grasp_index','state/task/selected_indices'):
                    same=np.array_equal(value[0],batch[label][key][index])
                else:continue
                if not same:failures.append(batch_name+f'/{label}/env{index}: initial recipe differs from N1: '+key)
    if baseline['new_case'] in cases:
        failures.extend(check_old_recipe(cases[baseline['new_case']]['initial'],load(baseline_root/baseline['snapshot']),load(baseline_root/baseline['checkpoint'])))
    return dict(passed=not failures,failures=failures,reports=reports,controller_lifecycle=base,
                scope=protocol['scope'],full_port_complete=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','protocol','baseline','pack','output'):p.add_argument(name,type=Path)
    a=p.parse_args();result=evaluate_hang(a.root,json.loads(a.protocol.read_text()),a.baseline,a.pack)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)

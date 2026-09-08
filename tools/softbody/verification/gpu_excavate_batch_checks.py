"""Independent Excavate batch geometry, outcomes and seed-recipe checks.

Task predicates and reward geometry follow ManiSkill2 v0.5.3 excavate_env.py.
The source terms in LEGACY-SOURCE-LICENSE.md apply. No simulator imports.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .gpu_controller_batch_checks import evaluate_controllers
from .job_archive import file_hash


def task_metrics(x, v, target, matrix, hull):
    """Reconstruct task outcome and reward from physical numeric observations."""
    lift = int(np.count_nonzero(x[:, 2] > .2))
    in_terrain = np.all((x[:, :2] > -.12) & (x[:, :2] < .12), axis=1)
    spill = int(np.count_nonzero(~in_terrain))
    quiet = np.count_nonzero((v > -.05) & (v < .05)) / v.size
    success = bool(target - 100 < lift < target + 150 and spill < 20 and quiet > .99)
    local = np.array([[0., -.01, .045, 1.], [-.03, -.01, .01, 1.],
                      [.03, -.01, .01, 1.], [-.03, .02, .08, 1.], [.03, .02, .08, 1.]])
    base, tl, tr, bl, br = np.array([(matrix @ p)[:3] for p in local], np.float32)
    top = np.cross(base-tr, tr-tl); top /= np.linalg.norm(top)
    bottom = np.cross(base-bl, bl-br); bottom /= np.linalg.norm(bottom)
    bounds = (matrix @ np.column_stack([hull, np.ones(len(hull))]).T).T[:, :3]
    bounded = np.all((x >= bounds.min(0)) & (x <= bounds.max(0)), axis=1)
    # Preserve the source's float32 plane-distance rounding before promotion.
    top_distance = -np.sum(top * base); bottom_distance = -np.sum(bottom * base)
    homogeneous = np.column_stack([x, np.ones(len(x))])
    inside = x[bounded & (homogeneous @ np.r_[top, top_distance].astype(float) > 0)
                      & (homogeneous @ np.r_[bottom, bottom_distance].astype(float) > 0)]
    n = len(inside)
    height_dist = max(.25 - np.mean(inside[:, 2]), 0) if n else 1
    lift_reward = min(n / target, 1) - max(0, n-target-500) * .001
    reach = height = flat = 0.
    if height_dist > .1 and n > target + 300:
        reach = 1.; height = 1 - np.tanh(3 * height_dist)
        flat = 1 - np.tanh(25 * (max(base[2]+.01-bl[2], 0) + max(bl[2]-br[2], 0)))
    elif height_dist <= .1:
        reach = 1.; height = 1 - np.tanh(3 * height_dist)
        lift_reward = 1 + min(n / target, 1) - max(0, n-target-100) * .001
        flat = 1 - np.tanh(25 * (max(base[2]-.01-bl[2], 0) + max(bl[2]-br[2], 0)))
    elif np.all((matrix[:2, 3] > -.1) & (matrix[:2, 3] < .1)):
        reach = 1 - np.tanh(10 * (matrix[2, 3] + max(.04-matrix[0, 3], 0)))
    reward = 6. if success else .5*reach + lift_reward + height + flat - spill/100
    return dict(success=success, lifted_particles=lift, spilled_particles=spill,
                reward=float(reward), inside=inside)


def check_task_snapshot(data, model, reward_limit):
    failures = []
    count = len(data['counts'])
    targets = data['state/task/target_num']
    if (targets.shape != (count, 1) or not np.isfinite(targets).all()
            or np.any(targets <= 0) or np.any(targets != np.floor(targets))):
        raise ValueError('Invalid task target counts')
    if not np.array_equal(data['reported/target'], targets.astype(np.float32)):
        failures.append('Target observation differs from per-environment checkpoint')
    for i in range(count):
        metrics = task_metrics(data[f'actual/{i}/x'], data[f'actual/{i}/v'], targets[i, 0],
                               data['bucket_pose'][i], model['bucket_reward_hull'][i])
        for key in ('success', 'lifted_particles', 'spilled_particles'):
            if data['reported/'+key].shape != (count,) or data['reported/'+key][i] != metrics[key]:
                failures.append(f'env{i}: independent task predicate differs: {key}')
        if not np.array_equal(data[f'reported/inside_bucket/{i}'], metrics['inside']):
            failures.append(f'env{i}: bucket membership differs from own native hull and pose')
        if abs(float(data['reported/reward'][i]) - metrics['reward']) > reward_limit:
            failures.append(f'env{i}: dense reward differs from independent task equation')
    return failures


def check_old_recipe(new, old, old_checkpoint):
    failures = []
    for key in ('x','v','F','C','vc','mass'):
        if not np.array_equal(old[key],new['actual/0/'+key]):
            failures.append('New N1 recipe differs from frozen old source: '+key)
    for key in ('particle_mass','particle_vol','particle_type','particle_mu_lam_ys','particle_friction_cohesion'):
        field = 'mpm_material/'+key
        if not np.array_equal(old_checkpoint[field],new['state/'+field]):
            failures.append('New N1 material differs from frozen old source: '+key)
    if not np.array_equal(old['task_state'],new['state/task/target_num'][0]):
        failures.append('New N1 target differs from frozen old source')
    if not np.array_equal(old['qpos'],new['qpos'][0]):
        failures.append('New N1 robot initialization differs from frozen old source')
    return failures


def evaluate_excavate(root, protocol, baseline_root):
    root, baseline_root = Path(root), Path(baseline_root)
    base = evaluate_controllers(root, protocol)
    failures, reports, cases_data = list(base['failures']), {}, {}
    baseline = protocol['baseline']
    for name, digest in baseline['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or file_hash(baseline_root/name) != digest:
            raise ValueError('Old N1 baseline identity mismatch')
    def load(path):
        with np.load(path, allow_pickle=False) as z:
            return {k:z[k] for k in z.files}
    for case in protocol['cases']:
        if case['task'] != 'Excavate':
            continue  # Fill regression cases are checked by the common controller suite.
        name = case['name']; directory = root/name
        result = json.loads((directory/'result.json').read_text())
        if not result['complete']: continue
        if result['reward_mode'] != 'dense': failures.append(name+': wrong actual reward mode')
        data = {f[:-4]:load(directory/f) for f in result['files'] if f.endswith('.npz')}
        model = data['model-initial']; count = len(case['seeds'])
        if not np.array_equal(model['stored_reward_hull'], model['bucket_reward_hull']):
            failures.append(name+': task hull differs from actual first native collision hull')
        for key in ('wall_mass', 'wall_inertia'):
            if not np.all(model[key] == 1.): failures.append(name+': wall inertial placeholders changed')
        if not np.array_equal(model['wall_half_size'], np.broadcast_to(np.array([.12,.02,.03],np.float32),(count,4,3))):
            failures.append(name+': original wall dimensions changed')
        labels = [label for label, values in data.items() if 'counts' in values]
        for label in labels:
            failures.extend(name+'/'+label+': '+f for f in check_task_snapshot(data[label],model,protocol['reward_limit']))
        for step in result['steps']:
            actual = data[step['label']]
            if not np.array_equal(step['reward'], actual['reported/reward']):
                failures.append(name+'/'+step['label']+': returned reward differs from task method')
        cases_data[name] = data
        reports[name] = dict(task_snapshots=len(labels), initial_counts=data['initial']['counts'].tolist(),
                             initial_targets=data['initial']['state/task/target_num'].ravel().tolist())
    for comparison in protocol['initial_comparisons']:
        batch_name, index, single_name = comparison
        if batch_name not in cases_data or single_name not in cases_data: continue
        batch, single = cases_data[batch_name], cases_data[single_name]
        for key, value in single['model-initial'].items():
            if not np.array_equal(value[0],batch['model-initial'][key][index]):
                failures.append(batch_name+f'/env{index}: native model differs from N1: '+key)
        for label in ('initial','reconfigured'):
            for key, value in single[label].items():
                if key.startswith('state/mpm/') or key.startswith('state/mpm_material/'):
                    n = single[label]['counts'][0]
                    equal = np.array_equal(value[0],batch[label][key][index,:n])
                elif key in ('qpos','state/task/target_num','counts'):
                    equal = np.array_equal(value[0],batch[label][key][index])
                else: continue
                if not equal: failures.append(batch_name+f'/env{index}/{label}: seed recipe differs from N1: '+key)
        if len(set(batch['initial']['counts'].tolist())) != 2:
            failures.append(batch_name+': protocol did not exercise different live particle counts')
    old = load(baseline_root/baseline['snapshot'])
    new_name = baseline['new_case']
    if new_name in cases_data:
        new = cases_data[new_name]['initial']
        failures.extend(check_old_recipe(new,old,load(baseline_root/baseline['checkpoint'])))
    return dict(passed=not failures, failures=failures, reports=reports,
                controller_lifecycle=base, scope=protocol['scope'], full_port_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path); parser.add_argument('protocol',type=Path)
    parser.add_argument('baseline',type=Path); parser.add_argument('output',type=Path)
    args = parser.parse_args()
    result = evaluate_excavate(args.root,json.loads(args.protocol.read_text()),args.baseline)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2)); raise SystemExit(0 if result['passed'] else 1)

"""Independent checks of actual shared-world task state and partial resets."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .gpu_rendering_checks import measure_frame
from .job_archive import file_hash, inventory


def check_snapshot(data, count, capacity):
    failures = []
    counts = data['counts']
    if counts.shape != (count,) or counts.dtype.kind not in 'iu' or np.any(counts <= 0):
        raise ValueError('Invalid actual particle counts')
    for index, n in enumerate(counts):
        x = data[f'actual/{index}/x']
        if x.shape != (n, 3) or data[f'actual/{index}/mass'].shape != (n,):
            raise ValueError('Actual solver particle count differs from its record')
        if count > 1:
            if data['state/mpm_meta/count'][index, 0] != n:
                failures.append('Exposed count differs from actual solver particles')
            if not np.array_equal(data['state/mpm_meta/mask'][index], np.arange(capacity) < n):
                failures.append('Exposed mask does not identify actual live particles')
            for key, value in data.items():
                if key.startswith('state/mpm/') or key.startswith('state/mpm_material/'):
                    if value.shape[:2] != (count, capacity) or np.any(value[index, n:] != 0):
                        failures.append('Nonzero or malformed inactive particle padding')
                if key.startswith(f'actual/{index}/'):
                    field = key.rsplit('/', 1)[1]
                    exposed = ('state/mpm_material/particle_mass' if field == 'mass' else 'state/mpm/' + field)
                    if not np.array_equal(data[exposed][index, :n], value):
                        failures.append('Exposed particle data differ from actual solver data: ' + field)
    if data['qpos'].shape != (count, 7) or data['qvel'].shape != (count, 7):
        raise ValueError('Wrong native robot batch shape')
    return failures


def state_changes(before, before_index, after, after_index, groups=None):
    keys = [k for k in before if k.startswith('state/') and (groups is None or k.split('/')[1] in groups)]
    after_keys = [k for k in after if k.startswith('state/') and (groups is None or k.split('/')[1] in groups)]
    return sorted(set(keys) ^ set(after_keys)) + [k for k in keys if k in after and not np.array_equal(before[k][before_index], after[k][after_index])]


def particle_errors(a, ai, b, bi):
    result = {}
    for key in ('x', 'v', 'F', 'C', 'vc'):
        av, bv = a[f'actual/{ai}/' + key], b[f'actual/{bi}/' + key]
        if av.shape != bv.shape:
            raise ValueError('Incomparable actual particle counts')
        result[key] = float(np.max(np.abs(av.astype(float) - bv.astype(float))))
    for key in ('qpos', 'qvel', 'drive_position', 'drive_velocity'):
        result[key] = float(np.max(np.abs(a[key][ai].astype(float) - b[key][bi].astype(float))))
    return result


def expected_actions(case, initial_qpos):
    scales = np.asarray(case['action_scales'], dtype=np.float64)[:, None]
    if not case.get('controller_lifecycle'):
        return np.array([np.array([.02,-.04,.01,.03,-.01,.03,-.02],np.float32)*s for s in case['action_scales']])
    mode = case['control_mode']; q = initial_qpos.astype(np.float64)
    if mode == 'pd_joint_pos':
        value = q + scales * np.array([.002, -.003, .001, -.002, .001, .002, -.001])
    elif mode == 'pd_joint_pos_vel':
        value = np.concatenate([q + scales * .002, np.repeat(scales * .02, 7, axis=1)], axis=1)
    elif mode == 'pd_joint_delta_pos_vel':
        value = np.concatenate([np.repeat(scales * .01, 7, axis=1), np.repeat(scales * .02, 7, axis=1)], axis=1)
    elif 'ee' in mode:
        value = scales * np.array([.01, -.02, .01, .03, -.02, .01])[:3 if mode.endswith('_pos') else 6]
    else:
        value = np.repeat(scales * .01, 7, axis=1)
    return value.astype(np.float32)


def evaluate(root, protocol):
    root = Path(root); failures, reports, arrays = [], {}, {}
    execution = json.loads((root/'execution.json').read_text())
    for key in ('input_sha256', 'image_id'):
        if execution[key] != protocol[key]: failures.append('Frozen identity mismatch: ' + key)
    if execution['exit_code'] != 0: failures.append('Worker execution failed')
    if file_hash(root/'request.json') != protocol['request_sha256'] or file_hash(root/'probe.py') != protocol['probe_sha256']:
        failures.append('Frozen request/probe mismatch')
    request = json.loads((root/'request.json').read_text()); prefix = 'mani_skill/envs/softbody/'
    sources = {k[len(prefix):]: v for k, v in request['source_files'].items()
               if k.startswith(prefix) and k.endswith('.py') and '/' not in k[len(prefix):]}
    sources.update({name:request['source_files']['mani_skill/envs/'+name] for name in ('scene.py','sapien_env.py')})
    if inventory(root/'source') != sources: failures.append('Actual runtime source mismatch')
    for case in protocol['cases']:
        name=case['name']; directory=root/name; result=json.loads((directory/'result.json').read_text())
        if result['case'] != case or result['probe_sha256'] != protocol['probe_sha256']:
            failures.append(name+': case/probe identity mismatch')
        if not result['complete'] or json.loads((directory/'execution.json').read_text())['exit_code'] != 0:
            failures.append(name+': '+result.get('error','incomplete'));continue
        count=len(case['seeds'])
        if (result['num_envs'] != count or result['world_couplers'] != count or result['physics_system'] != 'PhysxGpuSystem'
                or not result['shared_native_system'] or not result['model_scene_ownership'] or not result['replaced_native_system']):
            failures.append(name+': missing genuine shared-world/native scene ownership or reconfiguration')
        if result['mpm_substeps'] != [4]*count or result['rigid_dt'] != [float(np.float32(.002))]*count:
            failures.append(name+': physical timestep changed')
        labels=['warmup-1','warmup-2','expected']+(['continued'] if count==1 else ['partial-stepped','count-stepped','flat-stepped'])
        if [s['label'] for s in result['steps']] != labels or result['native_steps'] != 25*len(labels):
            failures.append(name+': missing controls or wrong total native step count')
        for step in result['steps']:
            if step['native_steps'] != 25 or step['world_models'] != count or len(step['reward']) != count:
                failures.append(name+': a control must advance the shared native world exactly25 times')
        data={}
        for filename,digest in result['files'].items():
            if Path(filename).name != filename or file_hash(directory/filename) != digest: raise ValueError('Invalid trace archive identity')
            with np.load(directory/filename,allow_pickle=False) as z: data[filename[:-4]]={k:z[k] for k in z.files}
            if any(v.dtype.kind not in 'fibu' or not np.isfinite(v).all() for v in data[filename[:-4]].values()):raise ValueError('Invalid numeric trace')
        if not np.array_equal(np.array(result['action'],np.float32),expected_actions(case,data['initial']['qpos'])):
            failures.append(name+': action recipe changed')
        for label, values in data.items():
            if 'counts' in values:failures.extend(name+'/'+label+': '+f for f in check_snapshot(values,count,case.get('capacity')))
        exact={}
        if count>1:
            for label,before,ai,after,bi,groups in [
                ('partial-unchosen', 'expected',1,'partial-restored',1,None),
                ('fresh-unchosen','partial-stepped',0,'partial-fresh',0,None),
                ('count-unchosen','partial-fresh',1,'partial-count',1,None),
                ('partial-selected','warmup-2',0,'partial-restored',0,protocol['exact_checkpoint_groups']),
                ('count-selected','reduced-checkpoint',0,'partial-count',0,protocol['exact_checkpoint_groups']),
                ('flat-env0','warmup-2',0,'flat-restored',0,protocol['exact_checkpoint_groups']),
                ('flat-env1','warmup-2',1,'flat-restored',1,protocol['exact_checkpoint_groups'])]:
                changes=state_changes(data[before],ai,data[after],bi,groups);exact[label]=changes
                if changes:failures.append(name+'/'+label+': changed checkpoint fields '+','.join(changes))
            expected_counts = data['partial-fresh']['counts'].copy()
            expected_counts[0] = (data['warmup-2']['counts'][0] + 1) // 2
            if not np.array_equal(data['partial-count']['counts'],expected_counts):failures.append(name+': count-changing reset did not affect only its selected model')
            for before,after,index in [('expected','partial-restored',1),('partial-stepped','partial-fresh',0),('partial-fresh','partial-count',1)]:
                if data[before]['elapsed_steps'][index]!=data[after]['elapsed_steps'][index]:failures.append(name+': reset changed unselected elapsed-step counter')
            leaves=[value.reshape(count,-1) for key,value in data['warmup-2'].items() if key.startswith('state/')]
            if not np.array_equal(np.concatenate(leaves,axis=1),data['checkpoint-flat']['flat']):failures.append(name+': flat checkpoint differs from dictionary values/order')
        for key,value in data['model-initial'].items():
            if not np.array_equal(value,data['model-reconfigured'][key]):failures.append(name+': reconfiguration changed native robot model inputs '+key)
        cameras={}
        expected_renders=[(label,index) for label in (['camera-initial','camera-reconfigured'] if count==1 else ['camera-initial','camera-partial-count','camera-reconfigured']) for index in range(count)]
        if [(r['label'],r['index']) for r in result['renders']] != expected_renders:failures.append(name+': missing batched camera frames')
        for frame in result['renders']:
            label=frame['file'][:-4];measured=measure_frame(data[label],.0025,protocol['rendering'])
            failures.extend(name+'/'+label+': '+f for f in measured['failures']);cameras[label]=measured
            with Image.open(directory/(label+'.png')) as picture:
                if not np.array_equal(np.asarray(picture),data[label]['rgb'][0]):failures.append(name+': rendered PNG differs from native RGB')
        arrays[name]=data;reports[name]=dict(exact_checkpoints=exact,cameras=cameras)
    if protocol.get('compare_batch_vs_single', True) and all(case['name'] in arrays for case in protocol['cases']):
        batch=arrays['Fill-batch']; comparisons={}
        for i,reference in enumerate(('Fill-single-101','Fill-single-17')):
            single=arrays[reference]
            for key,value in single['model-initial'].items():
                if not np.array_equal(value[0],batch['model-initial'][key][i]):failures.append('Batch native robot model inputs differ from N1: '+key)
            for label in ('initial','reconfigured'):
                error=particle_errors(batch[label],i,single[label],0)
                if any(error[k]!=0 for k in ('x','v','F','C','vc','qpos')):failures.append(f'env{i}/{label}: initialization differs from independent N1 seed')
            for label,reference_label in [('expected','expected'),('flat-stepped','expected'),('partial-stepped','expected' if i==0 else 'continued')]:
                error=particle_errors(batch[label],i,single[reference_label],0);comparisons[f'env{i}/{label}']=error
                for key,limit in protocol['replay_limits'].items():
                    if error[key]>limit:failures.append(f'env{i}/{label}: {key} difference {error[key]} exceeds {limit}')
        reports['batch_vs_single']=comparisons
    return dict(passed=not failures,failures=failures,reports=reports,scope=protocol['scope'],full_port_complete=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('protocol',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();result=evaluate(args.root,json.loads(args.protocol.read_text()))
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)

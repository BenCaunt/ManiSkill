"""Independent checks of actual shared-world task state and partial resets."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .gpu_rendering_checks import measure_frame
from .job_archive import file_hash, inventory


def check_snapshot(data, count, capacity, dof=7):
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
    if data['qpos'].shape != (count, dof) or data['qvel'].shape != (count, dof):
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


def expected_actions(case, initial_qpos, initial_ee=None):
    scales = np.asarray(case['action_scales'], dtype=np.float64)[:, None]
    if not case.get('controller_lifecycle'):
        return np.array([np.array([.02,-.04,.01,.03,-.01,.03,-.02],np.float32)*s for s in case['action_scales']])
    mode = case['control_mode']; q = initial_qpos[:,:7].astype(np.float64)
    if mode == 'pd_joint_pos':
        value = q + scales * np.array([.002, -.003, .001, -.002, .001, .002, -.001])
    elif mode == 'pd_joint_pos_vel':
        value = np.concatenate([q + scales * .002, np.repeat(scales * .02, 7, axis=1)], axis=1)
    elif mode == 'pd_joint_delta_pos_vel':
        value = np.concatenate([np.repeat(scales * .01, 7, axis=1), np.repeat(scales * .02, 7, axis=1)], axis=1)
    elif mode == 'pd_ee_pose':
        pose=np.asarray(initial_ee,dtype=np.float32)
        if pose.shape!=(len(q),7) or not np.isfinite(pose).all():raise ValueError('Invalid initial absolute EE poses')
        value=np.concatenate([pose[:,:3]+scales*np.array([.001,-.002,.001]),
            Rotation.from_quat(pose[:,[4,5,6,3]]).as_rotvec()+scales*np.array([.002,-.001,.003])],axis=1)
    elif 'ee' in mode:
        value = scales * np.array([.01, -.02, .01, .03, -.02, .01])[:3 if mode.endswith('_pos') else 6]
    else:
        value = np.repeat(scales * .01, 7, axis=1)
    if case.get('task') in ('Hang','Pour','Pinch'):
        value = np.column_stack([value,np.zeros(len(value))])
    return value.astype(np.float32)


def reduction_indices(case, snapshot):
    count = int(snapshot['counts'][0]); keep = np.arange(0,count,2)
    if case.get('preserve_task_particle_indices'):
        selected = snapshot['state/task/selected_indices'][0]
        if selected.shape != (5,) or not np.isfinite(selected).all() or np.any(selected != np.floor(selected)) or np.any((selected<0)|(selected>=count)):
            raise ValueError('Invalid task particle indices before reduction')
        keep = np.union1d(keep,selected.astype(int))
    return keep


def check_reduced_checkpoint(case, before, reduced):
    keep = reduction_indices(case,before); failures = []
    for key,value in before.items():
        if not key.startswith('state/'):continue
        expected = value[:1].copy()
        if key.startswith(('state/mpm/','state/mpm_material/')) or (case.get('task')=='Pinch' and key.startswith('state/task_particles/')):
            expected[:] = 0;expected[0,:len(keep)] = value[0,keep]
        elif key == 'state/mpm_meta/count':expected[:] = len(keep)
        elif key == 'state/mpm_meta/mask':expected[:] = np.arange(case['capacity']) < len(keep)
        elif key == 'state/task/selected_indices' and case.get('preserve_task_particle_indices'):
            expected[0] = np.searchsorted(keep,value[0])
        if key not in reduced or not np.array_equal(expected,reduced[key]):
            failures.append('Count-reduction recipe changed: '+key)
    return failures


def check_native_extension(root, request):
    """Check archived binary/build evidence independently of the staging helper.

    The helper checksum marks the new capture contract. Historical v11/v12
    requests retain their original verdict scope and are not retroactively passed.
    """
    if 'native_extensions_helper_sha256' not in request:
        return []
    failures=[];spec=request['native_actor_extension']
    if file_hash(root/'native_extensions.py')!=request['native_extensions_helper_sha256']:
        failures.append('Native actor staging helper differs')
    record=json.loads((root/'native-extension.json').read_text())
    filename='sapien303_actor_bridge.cpython-310-x86_64-linux-gnu.so'
    for key in ('build','sha256','cpp_sha256'):
        if record.get(key)!=spec[key]:failures.append('Native actor staging identity differs: '+key)
    if record.get('filename')!=filename or record.get('module')!='sapien303_actor_bridge':
        failures.append('Native actor module/ABI filename differs')
    binary=root/'native-extension'/filename
    if not binary.is_file() or file_hash(binary)!=spec['sha256']:
        failures.append('Archived native actor binary differs')
    build_path=root/'extension-build.json'
    if file_hash(build_path)!=record.get('build_manifest_sha256') or file_hash(root/'native-extension/build.json')!=file_hash(build_path):
        failures.append('Native actor build record differs')
    build=json.loads(build_path.read_text())
    expected={filename:spec['sha256'],'actor_bridge.cpp':spec['cpp_sha256'],
              'libsapien.so':'57b9dbf776bd216a2a71c86c34fadeab12469cc9067f6bf2338234499d80f097'}
    if build.get('sapien')!='3.0.3' or not build.get('python','').startswith('3.10.'):
        failures.append('Native actor build ABI differs')
    for name,digest in expected.items():
        if [v for k,v in build.get('files',{}).items() if Path(k).name==name]!=[digest]:
            failures.append('Native actor build provenance differs: '+name)
    if file_hash(root/'sapien303.py')!=request['source_files']['mani_skill/utils/sapien303.py']:
        failures.append('Native actor Python adapter differs from frozen source')
    return failures


def evaluate(root, protocol):
    root = Path(root); failures, reports, arrays = [], {}, {}
    execution = json.loads((root/'execution.json').read_text())
    for key in ('input_sha256', 'image_id'):
        if execution[key] != protocol[key]: failures.append('Frozen identity mismatch: ' + key)
    if execution['exit_code'] != 0: failures.append('Worker execution failed')
    if file_hash(root/'request.json') != protocol['request_sha256'] or file_hash(root/'probe.py') != protocol['probe_sha256']:
        failures.append('Frozen request/probe mismatch')
    request = json.loads((root/'request.json').read_text()); prefix = 'mani_skill/envs/softbody/'
    if protocol.get('rng_contract') is not None and (protocol['rng_contract']!=1 or request.get('rng_contract')!=1):
        raise ValueError('RNG verification requires the declared capture contract')
    failures.extend(check_native_extension(root,request))
    sources = {k[len(prefix):]: v for k, v in request['source_files'].items()
               if k.startswith(prefix) and k.endswith('.py') and '/' not in k[len(prefix):]}
    sources.update({name:request['source_files']['mani_skill/envs/'+name] for name in ('scene.py','sapien_env.py')})
    if inventory(root/'source') != sources: failures.append('Actual runtime source mismatch')
    for case in protocol['cases']:
        name=case['name']; directory=root/name; result=json.loads((directory/'result.json').read_text())
        if result['case'] != case or result['probe_sha256'] != protocol['probe_sha256']:
            failures.append(name+': case/probe identity mismatch')
        if 'native_extensions_helper_sha256' in request:
            loaded=result.get('native_actor_extension',{})
            if loaded.get('sha256')!=request['native_actor_extension']['sha256'] or not loaded.get('abi'):
                failures.append(name+': loaded native actor adapter differs or lacks ABI evidence')
        if not result['complete'] or json.loads((directory/'execution.json').read_text())['exit_code'] != 0:
            failures.append(name+': '+result.get('error','incomplete'));continue
        count=len(case['seeds'])
        dof=9 if case['task'] in ('Hang','Pour','Pinch') else 7
        if (result['num_envs'] != count or result['world_couplers'] != count or result['physics_system'] != 'PhysxGpuSystem'
                or not result['shared_native_system'] or not result['model_scene_ownership'] or not result['replaced_native_system']):
            failures.append(name+': missing genuine shared-world/native scene ownership or reconfiguration')
        if result['mpm_substeps'] != [4]*count or result['rigid_dt'] != [float(np.float32(.002))]*count:
            failures.append(name+': physical timestep changed')
        if 'particle_radius' in case and result.get('particle_radius') != [float(np.float32(case['particle_radius']))]*count:
            failures.append(name+': native particle radius changed')
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
        if not np.array_equal(np.array(result['action'],np.float32),expected_actions(case,data['initial']['qpos'],data['initial'].get('ee_pose_at_base'))):
            failures.append(name+': action recipe changed')
        for label, values in data.items():
            if 'counts' in values:failures.extend(name+'/'+label+': '+f for f in check_snapshot(values,count,case.get('capacity'),dof))
        if protocol.get('rng_contract') == 1:
            from .gpu_rng_checks import check_sequence
            failures.extend(name+': '+f for f in check_sequence(case,data,result['steps']))
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
            expected_counts[0] = len(reduction_indices(case,data['warmup-2']))
            if not np.array_equal(data['partial-count']['counts'],expected_counts):failures.append(name+': count-changing reset did not affect only its selected model')
            failures.extend(name+': '+f for f in check_reduced_checkpoint(case,data['warmup-2'],data['reduced-checkpoint']))
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
            label=frame['file'][:-4];measured=measure_frame(data[label],case.get('particle_radius',.0025),protocol['rendering'])
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

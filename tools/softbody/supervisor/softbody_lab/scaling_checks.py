"""Independent all-row scaling checks. No simulator imports or fitted tolerances."""
import argparse
import json
from pathlib import Path
import numpy as np

from .gpu_batch_checks import check_snapshot, state_changes
from .gpu_rng_checks import check_reset, validate_snapshot
from .scaling_contract import SNAPSHOTS, selection, validate_output


def reset_changes(before, after, checkpoint, selected):
    """Audit every selected and unselected row, including actual native buffers."""
    n=len(before['counts']); result={}
    for i in range(n):
        expected=checkpoint if i in selected else before
        changed=state_changes(expected,i,after,i)
        for key in ('qpos','qvel','drive_position','drive_velocity'):
            if not np.array_equal(expected[key][i],after[key][i]):changed.append(key)
        for key in expected:
            if key.startswith((f'actual/{i}/',f'native/{i}/')):
                if key not in after or not np.array_equal(expected[key],after[key]):changed.append(key)
        for key in expected:
            if key.startswith('model/') and (key not in after or not np.array_equal(expected[key][i],after[key][i])):
                changed.append(key)
        if changed:
            errors={}
            for key in sorted(set(changed)):
                if key not in expected or key not in after:
                    errors[key]=None;continue
                a,b=expected[key],after[key]
                if not key.startswith((f'actual/{i}/',f'native/{i}/')):a,b=a[i],b[i]
                errors[key]=(float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.shape==b.shape and a.size else None)
            result[str(i)]=dict(selected=i in selected,fields=sorted(set(changed)),max_abs_errors=errors)
    return result


def check_numeric(data, n, capacity):
    failures=check_snapshot(data,n,capacity)
    validate_snapshot(data,n)
    indices=[]
    for i in range(n):
        ix=data[f'native/{i}/indices']; rigid=data[f'native/{i}/rigid']
        if ix.ndim!=1 or ix.dtype.kind not in 'iu' or len(ix)<7 or np.any(ix<0) or rigid.shape!=(len(ix),13):
            raise ValueError('Invalid per-environment native body readback')
        robot=data[f'native/{i}/robot_indices'];coupler=data[f'native/{i}/coupler_indices']
        if (robot.ndim!=1 or robot.dtype.kind not in 'iu' or len(robot)<7 or np.any(robot<0)
                or coupler.ndim!=1 or coupler.dtype.kind not in 'iu' or len(coupler)<2 or np.any(coupler<0)
                or not np.array_equal(ix,np.union1d(robot,coupler))):
            raise ValueError('Native body readback omits robot or coupling bodies')
        indices.extend(ix.tolist())
        if np.any(data[f'actual/{i}/mass']<=0):failures.append(f'env{i}: nonpositive physical mass')
    if len(indices)!=len(set(indices)):failures.append('Native rigid rows overlap between environments')
    for key in ('model/shared_system','model/scene_ownership'):
        if data[key].shape!=(n,) or not data[key].all():failures.append(key+': not all models belong to their declared scene/world')
    return failures


def evaluate(root, request, payload_sha256):
    root=Path(root); record=validate_output(root,request,payload_sha256)
    case=request['scaling_case']; n=case['num_envs']; chosen=selection(n); failures=[]
    if (record.get('actual_backend')!='physx_cuda' or record.get('mpm_device')!='cuda'
            or record.get('gpu_sim_enabled') is not True or record.get('actual_num_envs')!=n
            or record.get('world_models')!=n or record.get('shared_native_system') is not True
            or record.get('model_scene_ownership') is not True or record.get('replaced_native_system') is not True
            or record.get('selected_indices')!=chosen):
        failures.append('Actual backend, scene ownership, model count, selection or reconstruction differs')
    if record['rigid_dt']!=[float(np.float32(.002))]*n or record['mpm_substeps']!=[4]*n:
        failures.append('Original physical timesteps changed')
    capacity={'Fill-v0':4096,'Excavate-v0':32768}[case['env_id']]
    if record['capacity']!=capacity:failures.append('Default padded capacity changed')
    data={}
    for label in SNAPSHOTS:
        with np.load(root/(label+'.npz'),allow_pickle=False) as z:values={k:z[k] for k in z.files}
        if any(v.dtype.kind not in 'fibu' or not np.isfinite(v).all() for v in values.values()):
            raise ValueError('Invalid numeric scaling snapshot')
        failures.extend(label+': '+f for f in check_numeric(values,n,capacity))
        data[label]=values
    expected_action=(data['initial']['qpos'].astype(np.float64)+(.04+.01*np.arange(n))[:,None]*np.array([.002,-.003,.001,-.002,.001,.002,-.001])).astype(np.float32)
    if record['control_mode']!='pd_joint_pos' or not np.array_equal(np.asarray(record['action'],np.float32),expected_action):
        failures.append('Nonzero physical control recipe changed')
    expected_steps=['warmup-0','warmup-1']+['timed-'+str(i) for i in range(case['timed_controls'])]+['partial-step','fresh-step','flat-step']
    if [s['label'] for s in record['steps']]!=expected_steps or record['native_steps']!=25*len(expected_steps):
        failures.append('Missing controls or incorrect native step count')
    if not np.isfinite(record['setup_seconds']) or record['setup_seconds']<=0:
        raise ValueError('Invalid setup timing')
    times=[]
    for s in record['steps']:
        if (s['native_steps']!=25 or s['world_models']!=n or len(s['reward'])!=n
                or not np.isfinite(s['seconds']) or s['seconds']<=0
                or s['timed']!=s['label'].startswith('timed-')):
            failures.append('Invalid physical step or timing record: '+s['label'])
        if s['timed']:times.append(s['seconds'])
    exact={}
    for label,before,checkpoint,selected in [('partial-restored','measured','warmup',chosen),
          ('flat-restored','fresh-stepped','warmup',list(range(n)))]:
        changes=reset_changes(data[before],data[label],data[checkpoint],selected);exact[label]=changes
        failures.extend(label+f'/env{i}: changed '+','.join(v['fields']) for i,v in changes.items())
    fresh=reset_changes(data['partial-stepped'],data['partial-fresh'],data['partial-fresh'],chosen)
    exact['partial-fresh']=fresh
    failures.extend('partial-fresh'+f'/env{i}: changed '+','.join(v['fields']) for i,v in fresh.items())
    for label,before,seeds,selected in [('initial',None,case['seeds'],list(range(n))),
        ('partial-restored','measured',[700+i for i in chosen],chosen),
        ('partial-fresh','partial-stepped',[900+i for i in chosen],chosen),
        ('flat-restored','fresh-stepped',[1100+i for i in reversed(range(n))],list(reversed(range(n)))),
        ('reconfigured',None,case['seeds'],list(range(n)))]:
        failures.extend(label+': '+f for f in check_reset(data[before] if before else None,data[label],selected,seeds,n))
        if before:
            untouched=[i for i in range(n) if i not in selected]
            if not np.array_equal(data[before]['elapsed_steps'][untouched],data[label]['elapsed_steps'][untouched]):
                failures.append(label+': unselected elapsed steps changed')
        if np.any(data[label]['elapsed_steps'][selected]!=0):failures.append(label+': selected elapsed steps did not reset')
    for before,after in [('initial','warmup'),('warmup','measured'),('partial-restored','partial-stepped'),
                         ('partial-fresh','fresh-stepped'),('flat-restored','flat-stepped')]:
        for i in range(n):
            if not np.array_equal(data[before][f'actual/{i}/mass'],data[after][f'actual/{i}/mass']):
                failures.append(after+f'/env{i}: particle mass/count changed during a rollout')
        for key in data[before]:
            if key.startswith('rng/') and not np.array_equal(data[before][key],data[after][key]):
                failures.append(after+': deterministic controls changed '+key)
    for label in SNAPSHOTS[1:]:
        for key in data['initial']:
            if key.startswith('model/') and not np.array_equal(data['initial'][key],data[label][key]):
                failures.append(label+': native model inputs changed: '+key)
    initial_changes=reset_changes(data['initial'],data['reconfigured'],data['initial'],list(range(n)))
    exact['reconfigured']=initial_changes
    failures.extend('reconfigured'+f'/env{i}: changed '+','.join(v['fields']) for i,v in initial_changes.items())
    if len(record['memory'])!=len(SNAPSHOTS) or [m['label'] for m in record['memory']]!=list(SNAPSHOTS):
        raise ValueError('Missing memory observations')
    for m in record['memory']:
        if any(type(v) is not int or v<0 for k,v in m.items() if k!='label') or not 0<m['device_used_bytes']<=m['device_total_bytes']:
            raise ValueError('Invalid memory observation')
    return dict(schema_version=1,passed=not failures,failures=failures,exact_resets=exact,
        scope='Every-row state/reset isolation and short synchronized scaling; strict differences retained; not full task or reference parity',
        case=case,initial_counts=data['initial']['counts'].tolist(),
        initial_mass_kg=[float(data['initial'][f'actual/{i}/mass'].sum()) for i in range(n)],
        observed_device_used_max_bytes=max(m['device_used_bytes'] for m in record['memory']),
        process_high_water_rss_kib=max(m['process_high_water_rss_kib'] for m in record['memory']),
        setup_seconds=record['setup_seconds'],timed_seconds=times,
        median_control_seconds=float(np.median(times)),aggregate_env_controls_per_second=n/float(np.median(times)),
        full_port_complete=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('job',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    from .remote_replay import recover_collected
    execution=recover_collected(a.job)
    if execution is None or execution['phase']!='complete':
        raise ValueError('Scaling audit requires an integrity-verified completed job')
    h=json.loads((a.job/'remote-job.json').read_text())
    request=json.loads((a.job/'collected/inputs/job.json').read_text())
    result=evaluate(a.job/'collected/output',request,h['payload_sha256'])
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result));raise SystemExit(0 if result['passed'] else 1)

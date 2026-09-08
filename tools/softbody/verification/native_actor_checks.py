"""Independent native actor selection checks; imports no candidate/simulator."""
import argparse
import json
from pathlib import Path
import numpy as np
from .job_archive import file_hash


def check_arrays(data,limit):
    failures=[]
    for a,b in [('initial_public','initial_bridge'),('selected_after','selected_bridge'),
                ('invalid_before','invalid_after')]:
        if not np.array_equal(data[a],data[b]):failures.append(f'{a}/{b}: native arrays differ')
    selection=data['selection'].astype(int)
    if selection.tolist()!=[3,1]:raise ValueError('Changed reordered selection recipe')
    for name in ['initial_public','initial_bridge','initial_requested','assigned',
                 'selected_before','selected_after','selected_bridge','invalid_before','invalid_after']:
        if data[name].shape!=(4,13) or data[name].dtype!=np.float32:raise ValueError('Invalid actor record shape/type')
    if data['selected_requested'].shape!=(2,13):raise ValueError('Invalid requested state')
    other=[i for i in range(4) if i not in selection]
    if not np.array_equal(data['selected_before'][other],data['selected_after'][other]):
        failures.append('Selected apply changed an unselected native actor')
    expected=data['selected_before'][selection].copy()
    expected[:,0]+=[.0125,-.0175]
    expected[:,7:10]=[[.13,.24,.35],[-.16,-.27,-.38]]
    if not np.array_equal(expected,data['selected_requested']):failures.append('Selected request recipe changed')
    delta=np.abs(data['selected_after'][selection].astype(float)-expected.astype(float))
    if delta[:,:7].max()>limit:failures.append('Selected pose exceeds declared native assignment tolerance')
    if np.any(delta[:,7:]!=0):failures.append('Selected velocities differ from request')
    repeated=data['selected_roundtrips'];full=data['full_roundtrips']
    if repeated.shape!=(21,4,13) or full.shape!=(21,4,13):raise ValueError('Missing repeated native assignments')
    if not np.array_equal(repeated[0],data['selected_after']) or not np.array_equal(full[0],repeated[-1]):
        failures.append('Native roundtrip sequence is discontinuous')
    for i in (0,2,3):
        if not np.array_equal(repeated[:,i],np.repeat(repeated[:1,i],21,axis=0)):
            failures.append(f'Repeated selected apply changed unselected actor {i}')
    if not np.array_equal(repeated[:,:,7:],np.repeat(repeated[:1,:,7:],21,axis=0)):
        failures.append('Repeated selected apply changed native velocities')
    return dict(failures=failures,selected_position_max_abs_m=float(delta[:,:3].max()),
                selected_quaternion_max_abs=float(delta[:,3:7].max()),
                selected_roundtrip_max_per_actor=np.max(abs(repeated-repeated[:1]),axis=(0,2)).tolist(),
                full_roundtrip_max_per_actor=np.max(abs(full-full[:1]),axis=(0,2)).tolist())


def evaluate(root,protocol,input_sha):
    root=Path(root);failures=[];reports={}
    execution=json.loads((root/'execution.json').read_text())
    if execution['exit_code']!=0 or execution['input_sha256']!=input_sha:failures.append('Execution/input identity mismatch')
    if json.loads((root/'protocol.json').read_text())!=protocol:failures.append('Protocol differs from frozen input')
    if file_hash(root/'probe.py')!=protocol['probe_sha256']:failures.append('Probe hash mismatch')
    build=json.loads((root/'extension-build.json').read_text())
    binaries=[digest for name,digest in build['files'].items() if name.endswith('actor_bridge.cpython-310-x86_64-linux-gnu.so')]
    if binaries!=[protocol['extension_sha256']]:failures.append('Native extension identity mismatch')
    for name in protocol['cases']:
        path=root/name;report=json.loads((path/'report.json').read_text())
        if not report['complete'] or json.loads((path/'execution.json').read_text())['exit_code']!=0:
            failures.append(name+': native execution incomplete');continue
        if report['case']!=name or report['explicit_steps_after_reset']!=0:failures.append(name+': changed diagnostic case/step scope')
        if file_hash(path/'actors.npz')!=report['files']['actors.npz']:raise ValueError('Native trace hash mismatch')
        metadata=report['metadata']
        if metadata['body_data_size']!=64 or metadata['pair_size']!=16:failures.append(name+': native structure sizes differ')
        if [a['gpu_index'] for a in metadata['actors']]!=report['python_gpu_indices']:failures.append(name+': native actor indexing differs')
        names={r['name'] for r in report['invalid_calls']}
        if names!={'wrong_system','wrong_actor','duplicate','wrong_shape','wrong_dtype','nonfinite','nonunit'} or not all(r['rejected'] for r in report['invalid_calls']):
            failures.append(name+': missing native input rejection')
        with np.load(path/'actors.npz',allow_pickle=False) as z:
            data={k:z[k] for k in z.files}
        if any(v.dtype.kind not in 'fibu' or not np.isfinite(v).all() for v in data.values()):raise ValueError('Invalid numeric trace')
        result=check_arrays(data,protocol['selected_pose_max_abs']);reports[name]=result
        failures.extend(name+': '+f for f in result['failures'])
    return dict(passed=not failures,scope=protocol['scope'],failures=failures,cases=reports)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('protocol',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--input-sha256',required=True);args=p.parse_args()
    result=evaluate(args.root,json.loads(args.protocol.read_text()),args.input_sha256)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)

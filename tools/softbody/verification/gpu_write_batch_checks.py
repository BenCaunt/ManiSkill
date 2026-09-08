"""Independent Write batch equations, original reset inputs and goal isolation.

Uses the existing PTX interval raster contract. No simulator/candidate imports.
Task equations are from MS2 v0.5.3; LEGACY-SOURCE-LICENSE.md applies.
"""
import argparse
import io
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .gpu_controller_batch_checks import evaluate_controllers
from .job_archive import file_hash
from .write_checks import outcome


def dense_reward(x, tcp_matrix, iou):
    # The reference rounds the stick bottom to float32 before the distance.
    bottom=np.asarray(tcp_matrix[:3,3]+tcp_matrix[:3,2]*.02,dtype=np.float32)
    distance=np.linalg.norm(x-bottom,axis=1).min()
    tilt=np.arcsin(np.clip(np.linalg.norm(np.cross(tcp_matrix[:3,2],[0,0,-1])),-1,1))
    return float(iou+.1*(1-np.tanh(10.*distance))+.1*(1-tilt))


def check_task_snapshot(data,reward_limit,pose_limit):
    if not np.isfinite([reward_limit,pose_limit]).all() or min(reward_limit,pose_limit)<0:
        raise ValueError('Invalid Write verifier limits')
    count=len(data['counts']);failures=[];measurements=[]
    shapes={'state/task/goal_points':(count,19404,3),'goal_height_mm':(count,64,64),
        'current_height_mm':(count,64,64),'reported/goal':(count,64,64),
        'reported/tcp_pose':(count,7),'tcp_pose':(count,7),'tcp_matrix':(count,4,4),
        'reported/iou':(count,),'reported/success':(count,),'reported/reward':(count,)}
    for key,shape in shapes.items():
        value=np.asarray(data[key])
        if value.shape!=shape or value.dtype.kind not in 'fibu' or not np.isfinite(value).all():
            raise ValueError('Invalid Write snapshot field: '+key)
    if data['reported/goal'].dtype!=np.uint8:
        failures.append('Goal observation must retain uint8 image values')
    if not np.array_equal(data['reported/tcp_pose'],data['tcp_pose']):
        failures.append('TCP observation differs from its own native link pose')
    indices=data['tcp_gpu_indices'];native=data['native_rigid']
    if (indices.shape!=(count,) or indices.dtype.kind not in 'iu' or len(np.unique(indices))!=count
            or native.ndim!=2 or native.shape[1]<7 or np.any((indices<0)|(indices>=len(native)))):
        raise ValueError('Invalid native TCP buffer ownership')
    if not np.array_equal(data['tcp_pose'],native[indices,:7]):
        failures.append('TCP poses differ from actual native GPU buffer rows')
    for i,n in enumerate(data['counts']):
        x=data[f'actual/{i}/x']
        if x.shape!=(n,3) or not np.isfinite(x).all():raise ValueError('Wrong live Write particle count')
        pose=data['tcp_pose'][i];matrix=data['tcp_matrix'][i]
        expected=np.eye(4);expected[:3,3]=pose[:3]
        expected[:3,:3]=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
        if np.max(np.abs(matrix-expected))>pose_limit:
            failures.append(f'env{i}: TCP matrix differs from native pose')
        value=outcome(dict(task_state=data['state/task/goal_points'][i],x=x,
            goal_height_mm=data['goal_height_mm'][i],current_height_mm=data['current_height_mm'][i]))
        if data['reported/success'][i]!=value['success']:
            failures.append(f'env{i}: success differs from independent raster predicates')
        if data['reported/iou'][i]!=np.float32(value['iou']):
            failures.append(f'env{i}: IoU differs from independent pixel counts')
        display=np.clip(data['goal_height_mm'][i,:,::-1],0,255).astype(np.uint8)
        if not np.array_equal(display,data['reported/goal'][i]):
            failures.append(f'env{i}: goal observation differs from its own verified raster')
        reward=dense_reward(x,matrix,value['iou'])
        if abs(float(data['reported/reward'][i])-reward)>reward_limit:
            failures.append(f'env{i}: reward differs from own particles, stick pose and goal')
        measurements.append({**value,'reward':reward})
    return dict(failures=failures,measurements=measurements)


def check_reference_initial(data,index,old,material):
    failures=[]
    for key in ('x','v','F','C','vc','mass'):
        if not np.array_equal(data[f'actual/{index}/'+key],old[key]):
            failures.append('Initial Write particles differ from frozen MS2: '+key)
    for key in ('qpos','qvel'):
        if not np.array_equal(data[key][index],old[key]):
            failures.append('Initial Write robot differs from frozen MS2: '+key)
    if not np.array_equal(data['state/task/goal_points'][index].reshape(-1),old['task_state']):
        failures.append('Initial goal differs from frozen MS2')
    n=len(old['x'])
    for key,value in material.items():
        if not np.array_equal(data['state/mpm_material/'+key][index,:n],value):
            failures.append('Initial Write material differs from frozen MS2: '+key)
    return failures


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def load_goals(level_root,files):
    goals={}
    for name in files:
        if not name.endswith('.h5'):continue  # Pinned license/provenance files are not levels.
        if Path(name).name!=name:raise ValueError('Invalid Write goal filename')
        path=Path(level_root)/name
        if file_hash(path)!=files[name]:raise ValueError('Pinned Write goal changed: '+name)
        if path.stat().st_size>64*1024**2:raise ValueError('Oversized pinned Write goal')
        with h5py.File(io.BytesIO(path.read_bytes()),'r') as f:
            if not isinstance(f.get('goal',getlink=True),h5py.HardLink):raise ValueError('Write goal must be inline')
            dataset=f['goal']
            if (not isinstance(dataset,h5py.Dataset) or dataset.shape!=(19404,3) or dataset.dtype.kind not in 'fiu'
                    or dataset.is_virtual or dataset.external):raise ValueError('Invalid pinned Write dataset')
            goals[name]=np.asarray(dataset,np.float32)
        if not np.isfinite(goals[name]).all():raise ValueError('Invalid pinned Write goal')
    return goals


def evaluate_write(root,protocol,baseline_root,pack_root,level_root):
    root,baseline_root,pack_root,level_root=map(Path,(root,baseline_root,pack_root,level_root))
    base=evaluate_controllers(root,protocol);failures=list(base['failures']);reports={}
    if 'task' not in protocol['exact_checkpoint_groups']:
        raise ValueError('Write lifecycle must verify exact task checkpoint restoration')
    for directory,files in [(baseline_root,protocol['baseline']['files']),
                            (pack_root,protocol['pack_files']),(level_root,protocol['level_files'])]:
        for name,digest in files.items():
            if file_hash(directory/name)!=digest:raise ValueError('Pinned Write input changed: '+name)
    pack=json.loads((pack_root/'export.json').read_text())
    baselines={(b['seed'],b['level_file']):b for b in protocol['baseline']['cases']}
    if len(baselines)!=len(protocol['baseline']['cases']):raise ValueError('Duplicate Write baseline identity')
    if any(b[k] not in protocol['baseline']['files'] for b in baselines.values() for k in ('snapshot','material')):
        raise ValueError('Unpinned Write reference input')
    goals=load_goals(level_root,protocol['level_files'])
    for case in protocol['cases']:
        if case['task']!='Write':continue
        name=case['name'];result=json.loads((root/name/'result.json').read_text())
        if not result['complete']:continue
        if result['reward_mode']!='dense':failures.append(name+': wrong native reward mode')
        data={f[:-4]:load(root/name/f) for f in result['files']};n=len(case['seeds'])
        if len(case['level_files'])!=n:raise ValueError('Missing per-row Write goals')
        model=data['model-initial'];robot=pack['robot_parameters'];walls=pack['geometry'][1:]
        expected={'robot_mass':[b['mass'] for b in robot['links']],
            'robot_inertia':[b['inertia'] for b in robot['links']], 'robot_com':[b['com'] for b in robot['links']],
            'joint_parent_pose':[j['parent_pose'] for j in robot['joints']],
            'joint_child_pose':[j['child_pose'] for j in robot['joints']],
            'wall_mass':[b['mass'] for b in walls], 'wall_inertia':[b['inertia'] for b in walls],
            'wall_com':[b['com'] for b in walls], 'wall_half_size':[[.15,.02,.04]]*4}
        for field,value in expected.items():
            expected_value=np.repeat(np.asarray(value,np.float32)[None],n,axis=0)
            if not np.array_equal(model[field],expected_value):failures.append(name+': original model changed: '+field)
        labels=[k for k,v in data.items() if 'counts' in v];measurements={}
        for label in labels:
            verdict=check_task_snapshot(data[label],protocol['reward_limit'],protocol['initial_pose_limit'])
            failures.extend(name+'/'+label+': '+f for f in verdict['failures'])
            measurements[label]=verdict['measurements']
        for step in result['steps']:
            if not np.array_equal(step['reward'],data[step['label']]['reported/reward']):
                failures.append(name+'/'+step['label']+': returned reward differs from measured state')
        for label in ('initial','reconfigured'):
            provenance=result['reset_level_provenance'][label]
            if provenance!={'files':case['level_files'],'sha256':[protocol['level_files'][f] for f in case['level_files']]}:
                failures.append(name+'/'+label+': loaded goals differ from requested files')
            for i,(seed,filename) in enumerate(zip(case['seeds'],case['level_files'])):
                baseline=baselines[(seed,filename)]
                failures.extend(name+f'/{label}/env{i}: '+f for f in check_reference_initial(data[label],i,
                    load(baseline_root/baseline['snapshot']),load(baseline_root/baseline['material'])))
                if not np.array_equal(data[label]['state/task/goal_points'][i],goals[filename]):
                    failures.append(name+f'/{label}/env{i}: goal differs from pinned file')
        if n>1:
            if np.array_equal(data['initial']['state/task/goal_points'][0],data['initial']['state/task/goal_points'][1]):
                failures.append(name+': distinct row goals were not exercised')
            fresh=goals[case['fresh_level_file']]
            if not np.array_equal(data['partial-fresh']['state/task/goal_points'][1],fresh):
                failures.append(name+': partial fresh reset did not load its requested goal')
            if any(np.array_equal(fresh,goal) for goal in data['initial']['state/task/goal_points']):
                failures.append(name+': restored goals were not exercised against a different reset goal')
        reports[name]=dict(task_snapshots=len(labels),measurements=measurements)
    if not reports:failures.append('No completed Write task cases')
    return dict(passed=not failures,failures=failures,reports=reports,controller_lifecycle=base,
        scope=protocol['scope'],full_port_complete=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','protocol','baseline','pack','levels','output'):p.add_argument(name,type=Path)
    a=p.parse_args();result=evaluate_write(a.root,json.loads(a.protocol.read_text()),a.baseline,a.pack,a.levels)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)

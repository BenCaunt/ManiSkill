"""Independent Pinch batch scores, camera goals and original input checks.

No simulator/candidate imports. The existing directed-fourth-norm arithmetic
contract is retained; it is not a physical trajectory tolerance.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .gpu_controller_batch_checks import evaluate_controllers
from .job_archive import file_hash
from .pinch_checks import outcome

CAMERA_SHAPES={'goal_depths':(4,128,128),'goal_rgbs':(4,128,128,3),'goal_cam_pos':(4,3),
               'goal_cam_rot':(4,4),'goal_cam_intrinsic':(3,3),'deformed_distance':(2,)}


def project_goal(data):
    row,col=np.indices((128,128));pixels=np.stack([col+.5,row+.5,np.ones_like(row)],axis=-1).reshape(-1,3)
    rays=pixels@np.linalg.inv(data['goal_cam_intrinsic']).T
    points=[]
    for depth,position,quaternion in zip(data['goal_depths'],data['goal_cam_pos'],data['goal_cam_rot']):
        depth=depth.reshape(-1);valid=depth>0
        camera=rays[valid]*depth[valid,None]
        local=np.column_stack([camera[:,2],-camera[:,0],-camera[:,1]])
        q=np.asarray(quaternion,np.float32)[[1,2,3,0]]
        world=Rotation.from_quat(q).apply(local)+np.asarray(position,np.float32)
        points.append(np.column_stack([world,np.ones(len(world))]))
    points=np.concatenate(points)
    return np.pad(points,((0,65536-len(points)),(0,0)))


def check_task_snapshot(data,reward_limit=1e-6,projection_limit=1e-7,pose_limit=1e-6):
    if not np.isfinite([reward_limit,projection_limit,pose_limit]).all() or min(reward_limit,projection_limit,pose_limit)<0:
        raise ValueError('Invalid Pinch verification limits')
    count=len(data['counts']);failures=[];measurements=[]
    shapes={**{'state/task/'+k:(count,*shape) for k,shape in CAMERA_SHAPES.items()},
        'reported/progress':(count,),'reported/success':(count,),'reported/reward':(count,),
        'reported/chamfer':(count,2),'reported/target_rgb':(count,4,128,128,3),
        'reported/target_depth':(count,4,128,128),'reported/target_points':(count,65536,4),
        'reported/tcp_pose':(count,7),'tcp_pose':(count,7),'tcp_matrix':(count,4,4)}
    for key,shape in shapes.items():
        value=np.asarray(data[key])
        if value.shape!=shape or value.dtype.kind not in 'fibu' or not np.isfinite(value).all():
            raise ValueError('Invalid Pinch snapshot field: '+key)
    goals=data['state/task_particles/goal']
    if goals.ndim!=3 or goals.shape[0]!=count or goals.shape[2]!=3 or goals.dtype!=np.float32 or not np.isfinite(goals).all():
        raise ValueError('Invalid padded Pinch goals')
    indices=data['tcp_gpu_indices'];native=data['native_rigid']
    if (indices.shape!=(count,) or indices.dtype.kind not in 'iu' or len(np.unique(indices))!=count
            or native.ndim!=2 or native.shape[1]<7 or np.any((indices<0)|(indices>=len(native)))):
        raise ValueError('Invalid Pinch native TCP ownership')
    if not np.array_equal(data['tcp_pose'],native[indices,:7]) or not np.array_equal(data['tcp_pose'],data['reported/tcp_pose']):
        failures.append('TCP observations differ from their own native GPU buffer rows')
    for i,n in enumerate(data['counts']):
        x=data[f'actual/{i}/x']
        if x.shape!=(n,3) or n<1 or n>goals.shape[1] or np.any(goals[i,n:]!=0):
            raise ValueError('Pinch goals must match live particle counts with zero padding')
        camera={k:data['state/task/'+k][i] for k in CAMERA_SHAPES}
        if (np.any(camera['goal_depths']<0) or abs(np.linalg.det(camera['goal_cam_intrinsic']))<1e-12
                or not np.allclose(np.linalg.norm(camera['goal_cam_rot'],axis=1),1,rtol=0,atol=1e-5)):
            raise ValueError('Invalid Pinch camera goal')
        value=outcome(dict(x=x,task_state=np.r_[camera['deformed_distance'],goals[i,:n].reshape(-1)]))
        directed=data['reported/chamfer'][i]
        for actual,bound in zip(directed,value['directed_fourth_norms']):
            if not bound['lower']<=actual<=bound['upper']:failures.append(f'env{i}: directed norm exceeds independent arithmetic bounds')
        distance=sum(directed);total=sum(camera['deformed_distance'])
        if data['reported/success'][i]!=value['success']:failures.append(f'env{i}: success differs from independent shape metric')
        if data['reported/progress'][i]!=1-distance/total:failures.append(f'env{i}: progress differs from its own goal normalization')
        if not value['progress_lower']<=data['reported/progress'][i]<=value['progress_upper']:
            failures.append(f'env{i}: progress exceeds independent arithmetic bounds')
        pose=data['tcp_pose'][i];matrix=data['tcp_matrix'][i]
        expected=np.eye(4);expected[:3,3]=pose[:3];expected[:3,:3]=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
        if np.max(abs(matrix-expected))>pose_limit:failures.append(f'env{i}: TCP matrix differs from native pose')
        bottom=np.asarray(matrix[:3,3]+.02*matrix[:3,2],np.float32)
        reach=1-np.tanh(10*np.linalg.norm(x-bottom,axis=1).min())
        tilt=np.arcsin(np.clip(np.linalg.norm(np.cross(matrix[:3,2],[0,0,-1])),-1,1))
        reward=-100*distance+.1*reach+.1*(1-tilt)
        if abs(float(data['reported/reward'][i])-reward)>reward_limit:failures.append(f'env{i}: reward differs from own particles/goal/TCP')
        if data['reported/target_rgb'].dtype!=np.uint8 or not np.array_equal(data['reported/target_rgb'][i],camera['goal_rgbs']):
            failures.append(f'env{i}: RGB goal observation differs from stored camera goal')
        if not np.array_equal(data['reported/target_depth'][i],camera['goal_depths'].astype(np.float32)):
            failures.append(f'env{i}: depth goal observation differs from stored camera goal')
        projection=project_goal(camera);actual=data['reported/target_points'][i]
        if not np.array_equal(actual[:,3],projection[:,3]) or np.max(abs(actual[:,:3]-projection[:,:3]))>projection_limit:
            failures.append(f'env{i}: point-cloud goal differs from independent camera projection')
        measurements.append({**value,'reported_directed_norms':directed.tolist(),'reward':float(reward)})
    return dict(failures=failures,measurements=measurements)


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def load_reference_inputs(config,baseline_root,pack_root):
    """Resolve every consumed input through a pinned, contained file list."""
    baseline_root,pack_root=map(Path,(baseline_root,pack_root))
    baseline=config['baseline']
    def pinned(root,files,name):
        path=root/name
        if name not in files or Path(name).is_absolute() or '..' in Path(name).parts or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Unpinned or non-contained Pinch input: '+name)
        if file_hash(path)!=files[name]:raise ValueError('Pinned Pinch input changed: '+name)
        return path
    for root,files in ((baseline_root,baseline['files']),(pack_root,config['pack_files'])):
        for name in files:pinned(root,files,name)
    pack=json.loads(pinned(pack_root,config['pack_files'],'export.json').read_text())
    records=json.loads(pinned(pack_root,config['pack_files'],'levels/levels.json').read_text())['levels']
    refs={};levels={}
    for record in baseline['cases']:
        key=(record['seed'],record['level_file'],record['control_mode'])
        if key in refs:raise ValueError('Duplicate Pinch baseline identity')
        refs[key]=record
        for field in ('snapshot','material'):pinned(baseline_root,baseline['files'],record[field])
    for filename,record in records.items():
        if Path(filename).name!=filename:raise ValueError('Invalid Pinch level name')
        name='levels/'+record['file'];path=pinned(pack_root,config['pack_files'],name)
        if config['pack_files'][name]!=record['sha256']:raise ValueError('Pinch level hash disagrees with manifest')
        levels[filename]=load(path)
    return pack,records,levels,refs


def check_level_goal(data,index,level,projection_limit):
    failures=[];n=len(level['goal'])
    if data['counts'][index]!=n or not np.array_equal(data['state/task_particles/goal'][index,:n],level['goal']):
        failures.append('Goal particles differ from requested pinned level')
    for key in CAMERA_SHAPES:
        if not np.array_equal(data['state/task/'+key][index],level[key]):
            failures.append('Goal metadata differs from requested pinned level: '+key)
    if np.max(abs(data['reported/target_points'][index]-level['goal_points_observation']))>projection_limit:
        failures.append('Goal projection differs from frozen MS2 observation')
    return failures


def check_initial(data,index,old,material,level,projection_limit):
    failures=[];n=len(old['x'])
    if abs(data['reported/progress'][index])>=1e-12:
        failures.append('Initial progress exceeds the existing strict 1e-12 gate')
    for key in ('x','v','F','C','vc','mass'):
        if not np.array_equal(data[f'actual/{index}/'+key],old[key]):failures.append('Initial particles differ from frozen MS2: '+key)
    for key in ('qpos','qvel','drive_position','drive_velocity'):
        if not np.array_equal(data[key][index],old[key]):failures.append('Initial robot/drives differ from frozen MS2: '+key)
    for key,value in material.items():
        if not np.array_equal(data['state/mpm_material/'+key][index,:n],value):failures.append('Initial material differs from frozen MS2: '+key)
    expected_goal=old['task_state'][2:].reshape(n,3)
    if not np.array_equal(data['state/task_particles/goal'][index,:n],expected_goal):failures.append('Initial goal particles differ from frozen MS2')
    for key in CAMERA_SHAPES:
        expected=old['task_state'][:2] if key=='deformed_distance' else old[key]
        if not np.array_equal(data['state/task/'+key][index],expected):failures.append('Initial goal metadata differs from frozen MS2: '+key)
    if np.max(abs(data['reported/target_points'][index]-level['goal_points_observation']))>projection_limit:
        failures.append('Goal projection differs from frozen MS2 observation')
    return failures


def check_initial_root(data,index,old):
    """Compare the saved physical articulation root, not its controller cache."""
    keys=[k for k in data if k.startswith('state/articulations/')]
    if len(keys)!=1:
        raise ValueError('Pinch initial root requires exactly one articulation')
    actual=np.asarray(data[keys[0]])
    if (actual.shape!=(len(data['counts']),31) or actual.dtype!=np.float32
            or not np.isfinite(actual).all() or not 0<=index<len(actual)):
        raise ValueError('Invalid Pinch physical articulation state')
    failures=[]
    for key,shape,part in [('root_pose',(1,7),slice(0,7)),('root_velocity',(1,6),slice(7,13))]:
        expected=np.asarray(old[key])
        if expected.shape!=shape or expected.dtype!=np.float32 or not np.isfinite(expected).all():
            raise ValueError('Invalid Pinch reference '+key)
        if not np.array_equal(actual[index,part],expected[0]):
            failures.append('Initial physical '+key+' differs from frozen MS2')
    return failures


def evaluate_pinch(root,protocol,baseline_root,pack_root):
    root,baseline_root,pack_root=map(Path,(root,baseline_root,pack_root))
    base=evaluate_controllers(root,protocol);failures=list(base['failures']);reports={}
    if not {'task','task_particles'}<=set(protocol['exact_checkpoint_groups']):raise ValueError('Pinch goals must be in exact checkpoint gates')
    config=protocol['pinch']
    if config.get('initial_root_contract') not in (None,1):
        raise ValueError('Unsupported Pinch initial root contract')
    pack,records,levels,refs=load_reference_inputs(config,baseline_root,pack_root)
    for case in protocol['cases']:
        if case['task']!='Pinch':continue
        name=case['name'];result=json.loads((root/name/'result.json').read_text())
        if not result['complete']:continue
        data={f[:-4]:load(root/name/f) for f in result['files']};n=len(case['seeds']);robot=pack['robot_parameters']
        if len(case['level_files'])!=n or any(f not in levels for f in case['level_files']):
            raise ValueError('Missing per-row pinned Pinch levels')
        for field,expected in [('robot_mass',[b['mass'] for b in robot['links']]),
            ('robot_inertia',[b['inertia'] for b in robot['links']]),('robot_com',[b['com'] for b in robot['links']]),
            ('joint_parent_pose',[j['parent_pose'] for j in robot['joints']]),('joint_child_pose',[j['child_pose'] for j in robot['joints']])]:
            if not np.array_equal(data['model-initial'][field],np.repeat(np.asarray(expected,np.float32)[None],n,axis=0)):
                failures.append(name+': original Pinch model differs: '+field)
        measurements={}
        for label,values in data.items():
            if 'counts' not in values:continue
            checked=check_task_snapshot(values,config['reward_limit'],config['projection_limit'],protocol['target_pose_limit'])
            failures.extend(name+'/'+label+': '+f for f in checked['failures']);measurements[label]=checked['measurements']
        if result['reward_mode']!='dense':failures.append(name+': wrong reward mode')
        for step in result['steps']:
            if not np.array_equal(step['reward'],data[step['label']]['reported/reward']):failures.append(name+': returned reward differs from measured state')
        for label in ('initial','reconfigured'):
            expected_provenance={'files':case['level_files'],'sha256':[records[f]['source_sha256'] for f in case['level_files']]}
            if result['reset_level_provenance'][label]!=expected_provenance:
                failures.append(name+'/'+label+': loaded levels differ from requested files')
            for i,(seed,filename) in enumerate(zip(case['seeds'],case['level_files'])):
                key=(seed,filename,case['control_mode'])
                failures.extend(name+f'/{label}/env{i}: '+f for f in check_level_goal(data[label],i,levels[filename],config['projection_limit']))
                # Require the exact controller baseline; never substitute another mode.
                reference=refs.get(key)
                if reference is None:
                    failures.append(name+f'/env{i}: missing matching reference seed/goal/controller baseline');continue
                old=load(baseline_root/reference['snapshot'])
                failures.extend(name+f'/{label}/env{i}: '+f for f in check_initial(data[label],i,
                    old,load(baseline_root/reference['material']),levels[filename],config['projection_limit']))
                if config.get('initial_root_contract')==1:
                    failures.extend(name+f'/{label}/env{i}: '+f for f in check_initial_root(data[label],i,old))
        if n>1:
            initial=[levels[f]['goal'] for f in case['level_files']]
            if any(np.array_equal(a,b) for i,a in enumerate(initial) for b in initial[i+1:]):
                failures.append(name+': distinct row goals were not exercised')
            filename=case['fresh_level_file'];fresh=levels[filename]
            failures.extend(name+'/partial-fresh/env1: '+f for f in check_level_goal(data['partial-fresh'],1,fresh,config['projection_limit']))
            provenance=result['reset_level_provenance']['partial-fresh']
            if provenance['files'][1]!=filename or provenance['sha256'][1]!=records[filename]['source_sha256']:
                failures.append(name+': partial fresh reset loaded the wrong level')
            if any(np.array_equal(fresh['goal'],goal) for goal in initial):
                failures.append(name+': checkpoint restore did not exercise a different reset goal')
        reports[name]=dict(task_snapshots=len(measurements),measurements=measurements)
    if not reports:failures.append('No completed Pinch cases')
    return dict(passed=not failures,failures=failures,reports=reports,controller_lifecycle=base,
                scope=protocol['scope'],full_port_complete=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','protocol','baseline','pack','output'):p.add_argument(name,type=Path)
    a=p.parse_args();result=evaluate_pinch(a.root,json.loads(a.protocol.read_text()),a.baseline,a.pack)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)

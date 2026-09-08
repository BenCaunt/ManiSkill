"""Independent Pour fluid counts, reward, native geometry and ring checks.

Equations follow ManiSkill2 v0.5.3 pour_env.py under the source terms in
LEGACY-SOURCE-LICENSE.md. This host verifier imports no simulator/candidate.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull

from .gpu_controller_batch_checks import evaluate_controllers
from .job_archive import file_hash


def grasped(impulses, finger_matrices):
    flags=[]
    for impulse,matrix,sign in zip(impulses,finger_matrices,(1,-1)):
        direction=sign*matrix[:3,1]
        cosine=impulse@direction/max(float(np.linalg.norm(impulse)*np.linalg.norm(direction)),1e-12)
        flags.append(np.linalg.norm(impulse)>=1e-6 and np.rad2deg(np.arccos(np.clip(cosine,-1,1)))<=85)
    return all(flags)


def task_metrics(x, qvel, source, beaker, tcp, left, right, impulses, heights, model):
    inside=(np.sum((x[:,:2]-beaker[:2,3])**2,axis=1)<model['target_radius']**2)&(x[:,2]<model['target_height'])
    counts=np.array([np.count_nonzero(inside&(x[:,2]>heights[0])),np.count_nonzero(inside&(x[:,2]>heights[1])),
                     np.count_nonzero(~inside&(x[:,2]<.001)),np.count_nonzero(inside)])
    start,end,spill,contained=counts
    success=bool(start>100 and end<10 and spill<100 and source[2,2]>=.866 and qvel.max()<.05 and qvel.min()>-.05)
    contact=grasped(impulses,(left,right))
    if success:return dict(counts=counts,success=True,reward=15.,grasp=contact)
    a,b=model['source_aabb'];rotation=source[:3,:3];position=source[:3,3]
    grasp_target=rotation@(a*np.array([.5,.5,.67])+b*np.array([.5,.5,.33]))+position
    distance=np.linalg.norm(tcp[:3,3]-grasp_target)
    reach=0. if distance<.05 else -distance
    held=bool(distance<.05 and contact)
    top=rotation@np.r_[(a[:2]+b[:2])*.5,b[2]]+position
    bottom=rotation@np.r_[(a[:2]+b[:2])*.5,a[2]]+position
    tx,ty,_,_,tz=model['target_aabc']
    destination=beaker[:3,:3]@np.array([tx,ty,tz])+beaker[:3,3]
    lf=np.linalg.norm(left[:2,3]-destination[:2]);rf=np.linalg.norm(right[:2,3]-destination[:2])
    fingers=10*(rf-lf);vertical=destination[2]-tcp[2,3]
    done=not(start<100 or (start>100 and start-end<100))
    fill=contained-max(0,start-500)*2 if done else contained
    orientation=distance_reward=0.
    if held:
        tilt=np.arcsin(np.clip(np.linalg.norm(np.cross(source[:3,2],[0,0,1])),-1,1))
        if done:
            orientation=3.5-tilt;distance_reward=5.5-np.tanh(10*(bottom[2]-top[2]));fingers=1.
        elif vertical>-.06:
            orientation=1-tilt;distance_reward=1-np.tanh(10*max(0,vertical+.06))
        else:
            orientation=2-np.tanh(10*max(0,top[2]-bottom[2]))
            distance_reward=2-np.tanh(10*np.linalg.norm(top-destination))
            if rf-lf>0:
                fingers=max(10*(rf-lf),10*(np.linalg.norm(bottom[:2]-destination[:2])-np.linalg.norm(top[:2]-destination[:2])))
    reward=reach+float(held)+.001*fill-.01*spill+orientation+distance_reward+fingers
    return dict(counts=counts,success=success,reward=float(reward),grasp=contact)


def check_task_snapshot(data, model, limit):
    failures=[];count=len(data['counts']);heights=data['state/task/fill_heights']
    if heights.shape!=(count,2) or not np.isfinite(heights).all() or np.any(heights[:,0]<=0) or np.any(heights[:,0]>=heights[:,1]):
        raise ValueError('Invalid per-environment fill heights')
    if not np.array_equal(data['reported/target'],heights[:,:1].astype(np.float32)):
        failures.append('Target observation differs from saved fill height')
    for i in range(count):
        values=task_metrics(data[f'actual/{i}/x'],data['qvel'][i],
            *[data[name+'_matrix'][i] for name in ('source','beaker','tcp','leftfinger','rightfinger')],
            [data['contact/left'][i],data['contact/right'][i]],heights[i],{k:v[i] for k,v in model.items()})
        for key,field in [('counts','task_counts'),('success','success'),('grasp','grasp')]:
            if not np.array_equal(values[key],data['reported/'+field][i]):failures.append(f'env{i}: independent {key} differs')
        if abs(float(data['reported/reward'][i])-values['reward'])>limit:
            failures.append(f'env{i}: independent dense reward differs')
    return failures


def check_ring(frame,index,protocol):
    failures=[];identifier=int(frame['target_ring_id'])
    for body_id in frame['task_visual_ids']:
        if np.count_nonzero(frame['segmentation']==body_id)<protocol['minimum_container_pixels']:
            failures.append('Missing source bottle or target beaker pixels')
    heights=frame['before/checkpoint/task/fill_heights'][index]
    beaker=frame['before/checkpoint/actors/target_beaker'][index,:3]
    if not np.array_equal(heights,frame['target_ring_heights']):failures.append('Ring heights differ from checkpoint')
    expected=np.r_[beaker[:2],heights[0]]
    if np.max(np.abs(frame['target_ring_pose'][:3,3]-expected))>protocol['initial_pose_limit']:
        failures.append('Ring pose differs from selected beaker and fill height')
    mask=frame['segmentation'][0,...,0]==identifier
    pixels=int(mask.sum())
    if pixels<protocol['minimum_ring_pixels']:failures.append('Too few target-ring pixels')
    maximum=0.
    if pixels:
        camera=frame['position'][0][mask].astype(float)/1000.
        transform=frame['cam2world_gl'][0];points=camera@transform[:3,:3].T+transform[:3,3]
        radius=float(frame['target_radius'])*1.02;radial=np.linalg.norm(points[:,:2]-beaker[:2],axis=1)
        maximum=float(max(np.max(radius*np.cos(np.pi/16)-radial),np.max(radial-radius),np.max(heights[0]-points[:,2]),np.max(points[:,2]-heights[1]),0))
        if maximum>protocol['ring_surface_limit_m']:failures.append('Ring pixels differ from physical target geometry')
    return dict(pixels=pixels,max_bound_error_m=maximum,failures=failures)


def hull_error(actual,expected):
    """Bidirectional convex containment ignores native cooking vertex order."""
    def outside(points,hull):
        planes=ConvexHull(hull).equations
        return float(np.max(points@planes[:,:3].T+planes[:,3]))
    return max(0.,outside(actual,expected),outside(expected,actual))


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def evaluate_pour(root,protocol,baseline_root,pack_root,collision_path,cooked_root=None):
    root,baseline_root,pack_root=map(Path,(root,baseline_root,pack_root))
    base=evaluate_controllers(root,protocol);failures=list(base['failures']);reports={};cases={}
    baseline=protocol['baseline']
    for name,digest in baseline['files'].items():
        if file_hash(baseline_root/name)!=digest:raise ValueError('Old-source baseline hash mismatch: '+name)
    if file_hash(pack_root/'export.json')!=protocol['pack_manifest_sha256'] or file_hash(Path(collision_path))!=protocol['bottle_collision_sha256']:
        raise ValueError('Pinned Pour pack/collision mismatch')
    pack=json.loads((pack_root/'export.json').read_text());collision=load(collision_path)
    cooked=None
    if 'cooked' in protocol:
        if cooked_root is None:raise ValueError('Cooked Pour verification requires its pinned geometry evidence')
        from .cooked_geometry_checks import load_evidence,check_model
        request=json.loads((root/'request.json').read_text())
        cooked=load_evidence(root,request,protocol['cooked'],cooked_root)
    for case in protocol['cases']:
        if case['task']!='Pour':continue
        name=case['name'];result=json.loads((root/name/'result.json').read_text())
        if not result['complete']:continue
        if result['reward_mode']!='dense':failures.append(name+': wrong reward mode')
        data={f[:-4]:load(root/name/f) for f in result['files']};cases[name]=data;model=data['model-initial'];n=len(case['seeds'])
        for prefix,record in zip(('source','beaker'),pack['geometry']):
            for field in ('mass','inertia','com'):
                expected=np.broadcast_to(np.asarray(record[field],np.float32),model[prefix+'_'+field].shape)
                if not np.array_equal(expected,model[prefix+'_'+field]):failures.append(name+': original rigid model changed: '+prefix+'/'+field)
        for field in ('source_aabb','target_aabb','target_aabc','target_radius','target_height'):
            if not np.array_equal(model[field],np.broadcast_to(pack[field],model[field].shape)):
                failures.append(name+': exported reward geometry changed: '+field)
        if not np.all(model['source_shape_count']==len(collision)):failures.append(name+': bottle convex component count changed')
        maximum=0.
        for j,key in enumerate(sorted(collision,key=lambda k:int(k.split('_')[-1]))):
            for i in range(n):maximum=max(maximum,hull_error(model[f'bottle_convex_{j}'][i],collision[key]))
        if maximum>protocol['collision_limit_m']:failures.append(name+': native bottle cavity geometry differs from pinned decomposition')
        cooked_report=None
        if cooked is not None:
            loaded=result.get('native_cooked_extension',{})
            if loaded.get('sha256')!=request['native_cooked_extension']['sha256'] or not loaded.get('abi'):
                failures.append(name+': missing actual loaded cooked extension identity')
            if result.get('bottle_cooked_pack')!={'sha256':protocol['cooked']['pack_manifest_sha256'],'leaves':cooked[0]['leaves']}:
                failures.append(name+': actual cooked bottle pack identity differs')
            cooked_report=check_model(model,*cooked,n,protocol['collision_limit_m'])
            failures.extend(name+': '+f for f in cooked_report['failures'])
        labels=[label for label,v in data.items() if 'counts' in v]
        for label in labels:failures.extend(name+'/'+label+': '+f for f in check_task_snapshot(data[label],model,protocol['reward_limit']))
        for step in result['steps']:
            if not np.array_equal(step['reward'],data[step['label']]['reported/reward']):failures.append(name+'/'+step['label']+': returned reward differs')
        rings={}
        for frame in result['renders']:
            measured=check_ring(data[frame['file'][:-4]],frame['index'],protocol);rings[frame['file']]=measured
            failures.extend(name+'/'+frame['file']+': '+f for f in measured['failures'])
        reports[name]=dict(task_snapshots=len(labels),initial_counts=data['initial']['counts'].tolist(),initial_heights=data['initial']['state/task/fill_heights'].tolist(),max_bottle_hull_error_m=maximum,rings=rings)
        if cooked_report is not None:reports[name]['cooked']=cooked_report
    for batch_name,index,single_name in protocol['initial_comparisons']:
        if batch_name not in cases or single_name not in cases:continue
        batch,single=cases[batch_name],cases[single_name]
        for key,value in single['model-initial'].items():
            if key.startswith('cooked/0/'):
                equal=np.array_equal(value,batch['model-initial'][key.replace('cooked/0/',f'cooked/{index}/',1)])
            else:
                equal=np.array_equal(value[0],batch['model-initial'][key][index])
            if not equal:failures.append(batch_name+f'/env{index}: model differs from N1: '+key)
        for label in ('initial','reconfigured'):
            for field in ('source_pose','beaker_pose'):
                if np.max(np.abs(single[label][field][0]-batch[label][field][index]))>protocol['initial_pose_limit']:
                    failures.append(batch_name+f'/{label}/env{index}: initial container pose differs from N1: '+field)
            for key,value in single[label].items():
                if key.startswith(('state/mpm/','state/mpm_material/')):
                    equal=np.array_equal(value[0],batch[label][key][index,:single[label]['counts'][0]])
                elif key in ('qpos','qvel','drive_position','drive_velocity','counts','reset_ik_attempts','state/task/fill_heights'):
                    equal=np.array_equal(value[0],batch[label][key][index])
                else:continue
                if not equal:failures.append(batch_name+f'/{label}/env{index}: recipe differs from N1: '+key)
        if len(set(batch['initial']['counts'].tolist()))<2:failures.append(batch_name+': distinct particle counts were not exercised')
    if baseline['new_case'] in cases:
        new=cases[baseline['new_case']]['initial'];old=load(baseline_root/baseline['snapshot']);checkpoint=load(baseline_root/baseline['checkpoint'])
        for key in ('x','v','F','C','vc','vol','mass'):
            if not np.array_equal(new['actual/0/'+key],old[key]):failures.append('N1 old-source particles changed: '+key)
        for key in ('qpos','qvel','drive_position','drive_velocity'):
            if not np.array_equal(new[key][0],old[key]):failures.append('N1 old-source robot/drives changed: '+key)
        if not np.array_equal(new['state/task/fill_heights'][0],old['task_state']):failures.append('N1 old-source fill heights changed')
        for key,value in checkpoint.items():
            if key.startswith('mpm_material/') and not np.array_equal(new['state/'+key],value):failures.append('N1 old-source material changed: '+key)
    return dict(passed=not failures,failures=failures,reports=reports,controller_lifecycle=base,scope=protocol['scope'],full_port_complete=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','protocol','baseline','pack','collision','output'):p.add_argument(name,type=Path)
    p.add_argument('--cooked-root',type=Path)
    a=p.parse_args();v=evaluate_pour(a.root,json.loads(a.protocol.read_text()),a.baseline,a.pack,a.collision,a.cooked_root)
    a.output.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');print(json.dumps(v,indent=2));raise SystemExit(0 if v['passed'] else 1)

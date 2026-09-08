"""Independent installed-module identity and analytic contact acceptance."""
import json
from pathlib import Path, PurePosixPath
import zipfile
import hashlib

import numpy as np


def check_origins(record,wheel_files,probe_sha256):
    failures=[];root=PurePosixPath('/work/site')
    if record.get('installed_root')!=str(root) or record.get('probe_sha256')!=probe_sha256:
        failures.append('Installed root or diagnostic identity differs')
    for phase in ('before','after'):
        modules=record.get(phase,{})
        if not {'mani_skill.envs.softbody.mpm','warp','sapien303_actor_bridge','sapien303_cooked_bridge'}.issubset(modules):
            failures.append(phase+': required installed module missing')
        for name,item in modules.items():
            relative=PurePosixPath(item['relative_path']);absolute=PurePosixPath(item['path'])
            if relative.is_absolute() or '..' in relative.parts or absolute!=root/relative:
                failures.append(phase+': module escaped installation: '+name)
            if wheel_files.get(str(relative))!=item.get('sha256'):
                failures.append(phase+': imported file differs from wheel: '+name)
    return failures


def check_contact(data,variant):
    """Recompute impulses from the trace under the original authored limits."""
    failures=[];metrics={}
    shapes={'particle_x':(101,64,3),'particle_v':(101,64,3),'particle_mass':(64,),
        'rigid_pose':(101,7),'rigid_v':(101,3),'wrenches':(100,4,1,6),'total_momentum':(101,3)}
    if variant in ('slider','hinge'):shapes.update(joint_qpos=(100,1),joint_qvel=(100,1))
    for key,shape in shapes.items():
        if key not in data or data[key].shape!=shape or not np.isfinite(data[key]).all():
            raise ValueError('Invalid physical contact trace: '+key)
    mass=data['particle_mass'].astype(float);rigid_mass=float(np.float32(.05))
    if not np.allclose(mass,1000*.003**3,rtol=1e-6,atol=0):failures.append('Authored particle mass changed')
    w=data['wrenches'].astype(float).mean(axis=1)[:,0,:]
    impulse=np.cumsum(w[:,3:]*.002,axis=0)
    displacement=float(np.linalg.norm(data['rigid_pose'][-1,:3]-data['rigid_pose'][0,:3]))
    clearance=float(.3-abs(data['particle_x'][...,0]).max())
    force=float(np.linalg.norm(data['wrenches'][...,3:],axis=-1).max())
    metrics.update(displacement_m=displacement,grid_clearance_m=clearance,peak_force_n=force)
    if displacement<=1e-5 or clearance<=.02 or force<=0:failures.append('Missing isolated physical contact motion')
    if variant in ('free','rotated'):
        p=(data['particle_v'].astype(float)*mass[None,:,None]).sum(axis=1)+rigid_mass*data['rigid_v']
        error=float(np.linalg.norm(p-p[0],axis=1).max())
        rigid_error=float(np.linalg.norm(rigid_mass*data['rigid_v'][-1]-impulse[-1]))
        metrics.update(momentum_error_kg_m_s=error,rigid_impulse_error_kg_m_s=rigid_error)
        if error>2e-6 or rigid_error>2e-7:failures.append('Linear impulse/momentum gate failed')
    else:
        if variant=='slider':expected=impulse[:,0];actual=rigid_mass*data['joint_qvel'][:,0];limit=2e-7
        else:
            about_pivot=w[:,:3]+np.cross(data['rigid_pose'][:-1,:3]-[0,0,.1],w[:,3:])
            expected=np.cumsum(about_pivot[:,2]*.002)
            inertia_z=float(np.float32(.05/3*(.006**2+.04**2)))
            actual=(inertia_z+rigid_mass*.025**2)*data['joint_qvel'][:,0];limit=2e-8
        error=float(abs(actual-expected).max());metrics['joint_impulse_error']=error
        if error>limit:failures.append('Generalized impulse gate failed')
    return dict(passed=not failures,failures=failures,metrics=metrics)


def verify(root,protocol,wheel):
    root=Path(root);failures=[];contacts={};origins={}
    with zipfile.ZipFile(wheel) as z:
        files={name:hashlib.sha256(z.read(name)).hexdigest() for name in z.namelist()}
    if hashlib.sha256(Path(wheel).read_bytes()).hexdigest()!=protocol['wheel']['sha256']:
        raise ValueError('Installed input wheel differs')
    execution=json.loads((root/'execution.json').read_text())
    if execution!={k:protocol[k] for k in ('input_sha256','image_id')}|{'exit_code':0}:
        failures.append('Installed worker execution/identity failed')
    for device in ('cpu','cuda'):
        for variant in ('free','rotated','slider','hinge'):
            name='coupling-'+device+'-'+variant;directory=root/name
            record=json.loads((directory/'module-origins.json').read_text())
            origins[name]=check_origins(record,files,protocol['coupling_probe_sha256'])
            report=json.loads((directory/'contact/report.json').read_text())
            if (json.loads((directory/'execution.json').read_text())['exit_code']!=0
                    or not report['passed'] or report['device']!=device or report['steps']!=100
                    or report['rotated_plate']!=(variant=='rotated')
                    or report['articulation']!=(variant if variant in ('slider','hinge') else None)):
                failures.append(name+': diagnostic execution differs')
            with np.load(directory/'contact/states.npz',allow_pickle=False) as z:contacts[name]=check_contact(dict(z),variant)
            failures.extend(name+': '+f for f in contacts[name]['failures'])
    for case in protocol['cases']:
        name=case['name'];record=json.loads((root/name/'module-origins.json').read_text())
        origins[name]=check_origins(record,files,protocol['probe_sha256'])
    for name,errors in origins.items():failures.extend(name+': '+f for f in errors)
    return dict(passed=not failures,failures=failures,contacts=contacts,
        origins={name:dict(passed=not errors,failures=errors) for name,errors in origins.items()},
        scope='Installed wheel identity in twelve processes and eight analytic contacts. Separate unchanged Pour acceptance verdict required.')

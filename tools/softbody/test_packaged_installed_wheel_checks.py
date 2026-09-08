import copy
import numpy as np
import pytest
from verification.installed_wheel_checks import check_origins,check_contact


def origin_fixture():
    files={};modules={}
    for name in ('mani_skill.envs.softbody.mpm','warp','sapien303_actor_bridge','sapien303_cooked_bridge'):
        path=name.replace('.','/')+'.py';files[path]='a'*64
        modules[name]=dict(relative_path=path,path='/work/site/'+path,sha256='a'*64)
    return dict(installed_root='/work/site',probe_sha256='b'*64,before=modules,after=copy.deepcopy(modules)),files


def test_installed_identity():
    record,files=origin_fixture();assert not check_origins(record,files,'b'*64)


@pytest.mark.parametrize('kind',['outside','traversal','hash','missing','probe'])
def test_reject_import_substitution(kind):
    record,files=origin_fixture();item=record['after']['warp']
    if kind=='outside':item['path']='/source/warp.py'
    if kind=='traversal':item.update(relative_path='../warp.py',path='/work/warp.py')
    if kind=='hash':item['sha256']='c'*64
    if kind=='missing':del record['after']['warp']
    if kind=='probe':record['probe_sha256']='c'*64
    assert check_origins(record,files,'b'*64)


def trace():
    result={key:np.zeros(shape) for key,shape in dict(particle_x=(101,64,3),particle_v=(101,64,3),
        particle_mass=(64,),rigid_pose=(101,7),rigid_v=(101,3),wrenches=(100,4,1,6),total_momentum=(101,3)).items()}
    result['particle_mass'][:]=1000*.003**3
    return result


def test_static_trace_cannot_pass_contact():
    assert not check_contact(trace(),'free')['passed']


def test_injected_motion_fails_momentum():
    data=trace();data['rigid_pose'][-1,0]=.01;data['rigid_v'][-1,0]=1
    data['wrenches'][:,:,:,3]=1
    result=check_contact(data,'free')
    assert 'Linear impulse/momentum gate failed' in result['failures']


def test_bad_mass_and_nonfinite_trace_rejected():
    data=trace();data['particle_mass']*=2
    assert 'Authored particle mass changed' in check_contact(data,'free')['failures']
    data['particle_x'][0,0,0]=np.nan
    with pytest.raises(ValueError):check_contact(data,'free')

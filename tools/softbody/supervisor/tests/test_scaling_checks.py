import copy
import json
from pathlib import Path

import numpy as np
import pytest

from softbody_lab import remote_replay
from softbody_lab.job_archive import file_hash, inventory
from softbody_lab.scaling_contract import SNAPSHOTS, digest, selection, validate_case, validate_output
from softbody_lab.scaling_checks import check_numeric, reset_changes


def case(n=32):
    return dict(env_id='Fill-v0',num_envs=n,seeds=list(range(101,101+n)),timed_controls=5)


@pytest.mark.parametrize('change', [dict(env_id='Pour-v0'),dict(num_envs=1),dict(num_envs=True),
    dict(timed_controls=0),dict(timed_controls=21),dict(seeds=[1]*32),dict(seeds=list(range(31))),
    dict(seeds=[-1]+list(range(31))),dict(seeds=[[1]]*32),dict(extra='untrusted')])
def test_scaling_case_rejects_undeclared_or_unbounded_inputs(change):
    with pytest.raises(ValueError):validate_case({**case(),**change})


def snapshot(n=32):
    rng=[np.random.RandomState(i).get_state() for i in range(n)]
    s={'counts':np.full(n,2,dtype=int),'qpos':np.zeros((n,7)),'qvel':np.zeros((n,7)),
       'drive_position':np.ones((n,7)), 'drive_velocity':np.zeros((n,7)),
       'state/agent/qpos':np.zeros((n,7)), 'state/mpm/x':np.zeros((n,2,3)),
       'state/mpm_material/particle_mass':np.ones((n,2)), 'state/mpm_meta/count':np.full((n,1),2),
       'state/mpm_meta/mask':np.ones((n,2),bool),
       'model/shared_system':np.ones(n,bool),'model/scene_ownership':np.ones(n,bool),
       'rng/main_seed':np.arange(n),'rng/episode_seed':np.arange(n)}
    for name, states in [('main',rng),('episode',rng),('legacy_main',rng[:1])]:
        for field,k in [('words',1),('cursor',2),('has_gauss',3),('gaussian',4)]:s[f'rng/{name}/{field}']=np.array([r[k] for r in states])
    for i in range(n):
        s[f'actual/{i}/x']=np.zeros((2,3));s[f'actual/{i}/mass']=np.ones(2)
        s[f'native/{i}/indices']=np.arange(i*8,(i+1)*8);s[f'native/{i}/rigid']=np.zeros((8,13))
        s[f'native/{i}/robot_indices']=np.arange(i*8,(i+1)*8);s[f'native/{i}/coupler_indices']=np.arange(i*8,i*8+2)
    return s


@pytest.mark.parametrize('index',[0,15,16,30,31])
@pytest.mark.parametrize('field',['state/agent/qpos','qpos','native','particle'])
def test_reset_audit_detects_fault_in_every_part_of_large_batch(index,field):
    before=snapshot(); checkpoint=copy.deepcopy(before); after=copy.deepcopy(before)
    if field=='native':after[f'native/{index}/rigid'][0,0]=.001
    elif field=='particle':after[f'actual/{index}/x'][0,0]=.001
    else:after[field][index,0]=.001
    result=reset_changes(before,after,checkpoint,selection(32))
    assert str(index) in result and result[str(index)]['selected']==(index in selection(32))


def test_reset_audit_does_not_require_selected_rows_to_match_current_time():
    before=snapshot();checkpoint=copy.deepcopy(before);after=copy.deepcopy(before)
    for index in selection(32):
        checkpoint['state/agent/qpos'][index,0]=2
        after['state/agent/qpos'][index,0]=2
    assert reset_changes(before,after,checkpoint,selection(32))=={}


@pytest.mark.parametrize('fault',[None,'overlap','ownership','padding'])
def test_actual_state_audit_checks_late_rows(fault):
    s=snapshot()
    if fault=='overlap':
        for field in ('indices','robot_indices','coupler_indices'):s['native/31/'+field][0]=s['native/0/indices'][0]
    if fault=='ownership':s['model/scene_ownership'][31]=False
    if fault=='padding':s['state/mpm_meta/mask'][31,1]=False
    failures=check_numeric(s,32,2)
    assert bool(failures)==(fault is not None)


@pytest.mark.parametrize('expired',[False,True])
def test_scaling_preparation_contains_no_reference_data(monkeypatch,tmp_path,expired):
    def copy_source(_,destination):
        destination.mkdir();(destination/'setup.py').write_text('# synthetic build input\n');return 'a'*40
    monkeypatch.setattr(remote_replay,'copy_source',copy_source)
    lease=tmp_path/'lease.json';lease.write_text(json.dumps({'terminate_at_epoch':remote_replay.time.time()+(-1 if expired else 3600)}))
    harness=tmp_path/'harness/softbody_lab';harness.mkdir(parents=True);(harness/'__init__.py').write_text('')
    (harness/'reference-secret.npz').write_bytes(b'Must not be packaged')
    output=tmp_path/'output'
    if expired:
        with pytest.raises(ValueError,match='Lease'):remote_replay.prepare_scaling(None,output,lease,'sha256:'+'b'*64,case(2),harness=harness.parent)
        assert not output.exists()
    else:
        h=remote_replay.prepare_scaling(None,output,lease,'sha256:'+'b'*64,case(2),harness=harness.parent)
        r=json.loads((output/'payload/job.json').read_text())
        assert r['role']=='scaling' and r['candidate_sim_backend']=='physx_cuda'
        assert set(r['file_sha256'])=={'source','harness'}
        assert set(inventory(output/'payload/harness'))=={'softbody_lab/__init__.py'}
        assert 'scaling_contract.py' in h['controller_sha256']


@pytest.mark.parametrize('fault',[None,'case','image','payload','source','harness','request','file','missing'])
def test_scaling_output_rejects_changed_job_and_artifacts(tmp_path,fault):
    request=dict(scaling_case=case(2),image='sha256:'+'a'*64,file_sha256={'source':{'a.py':'b'*64},'harness':{'capture.py':'c'*64}})
    provenance=dict(role='scaling',case=case(2),image=request['image'],payload_sha256='d'*64,
        source_files_digest=digest(request['file_sha256']['source']),harness_files_digest=digest(request['file_sha256']['harness']),request_digest=digest(request))
    files={}
    for n in SNAPSHOTS:
        p=tmp_path/(n+'.npz');p.write_bytes(b'Synthetic integrity-only stand-in');files[p.name]=file_hash(p)
    if fault=='case':provenance['case']['seeds'][0]=999
    if fault=='image':provenance['image']='changed'
    if fault=='payload':provenance['payload_sha256']='e'*64
    for k in ('source','harness','request'):
        if fault==k:provenance[k+('_files_digest' if k!='request' else '_digest')]='f'*64
    if fault=='file':(tmp_path/'initial.npz').write_bytes(b'altered')
    if fault=='missing':files.pop('reconfigured.npz')
    (tmp_path/'result.json').write_text(json.dumps(dict(complete=True,provenance=provenance,files=files)))
    if fault:
        with pytest.raises(ValueError):validate_output(tmp_path,request,'d'*64)
    else:assert validate_output(tmp_path,request,'d'*64)['complete']


def test_native_readback_cannot_omit_robot_links():
    s=snapshot();s['native/31/robot_indices']=s['native/31/robot_indices'][:2]
    with pytest.raises(ValueError,match='omits robot'):check_numeric(s,32,2)

import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from softbody_lab.job_archive import archive_inventory, atomic_json, file_hash, inventory, safe_extract
from softbody_lab import remote_replay


@pytest.mark.parametrize('fault', [None, 'checksum', 'symlink'])
def test_offline_physx_gpu_staging_requires_exact_ordinary_library(monkeypatch, tmp_path, fault):
    from softbody_lab import native_extensions as native
    source = tmp_path/'provenance/physx-gpu'/native.PHYSX_GPU_VERSION/'files'/native.PHYSX_GPU_FILENAME
    source.parent.mkdir(parents=True)
    source.write_bytes(b'Synthetic ELF stand-in, never loaded')
    monkeypatch.setattr(native, 'PHYSX_GPU_SHA256', file_hash(source))
    if fault == 'checksum': source.write_bytes(b'Changed binary')
    if fault == 'symlink':
        target = tmp_path/'elsewhere'; source.rename(target); source.symlink_to(target)
    destination = tmp_path/'staged'
    if fault:
        with pytest.raises(ValueError): native.stage_physx_gpu(tmp_path, destination)
        assert not destination.exists()
    else:
        record = native.stage_physx_gpu(tmp_path, destination)
        assert file_hash(destination/native.PHYSX_GPU_FILENAME) == record['sha256'] == native.PHYSX_GPU_SHA256
        assert (destination/'record.json').exists()


@pytest.mark.parametrize('backend', ['auto', 'cuda', '', 1])
def test_invalid_backend_cannot_prepare_or_upload(tmp_path, backend):
    with pytest.raises(ValueError, match='Candidate backend'):
        remote_replay.prepare(None, None, tmp_path/'job', None, None, candidate_sim_backend=backend)
    assert not (tmp_path/'job').exists()


@pytest.mark.parametrize('conflict', [False, True])
def test_backend_is_pinned_separately_from_fixture(monkeypatch, tmp_path, conflict):
    fixture = tmp_path/'fixture'; fixture.mkdir()
    record = dict(fixture=dict(env_id='Excavate-v0', env_kwargs={}), fixture_sha256='e'*64)
    if conflict:
        record['fixture']['env_kwargs']['sim_backend'] = 'physx_cpu'
    for name in ('fixture.json', 'initial.npz', 'actions.npy'):
        (fixture/name).write_text('Synthetic package fixture, no physics claim')
    original = inventory(fixture)
    monkeypatch.setattr(remote_replay, 'load_fixture', lambda _: (record, None, None, None))
    def copy_source(_candidate, destination):
        destination.mkdir(); (destination/'setup.py').write_text('# synthetic input\n')
        return 'a'*40
    monkeypatch.setattr(remote_replay, 'copy_source', copy_source)
    lease = tmp_path/'lease.json'
    lease.write_text(json.dumps({'terminate_at_epoch': remote_replay.time.time()+2000}))
    output = tmp_path/'job'
    args = (None, fixture, output, lease, 'sha256:'+'b'*64)
    if conflict:
        with pytest.raises(ValueError, match='conflicts'):
            remote_replay.prepare(*args, candidate_sim_backend='physx_cuda')
        assert not output.exists()
    else:
        handle = remote_replay.prepare(*args, candidate_sim_backend='physx_cuda')
        request = json.loads((output/'payload/job.json').read_text())
        assert request['candidate_sim_backend'] == 'physx_cuda'
        assert request['file_sha256']['fixture'] == original == inventory(fixture)
        assert handle['request_sha256'] == remote_replay.digest_json(request)


@pytest.mark.parametrize('names', [['../escape'], ['/absolute'], ['a/../escape'],
                                ['a', 'a'], ['a//b'], ['a', 'a/b'], ['a\\b']])
def test_archive_rejects_unsafe_names_before_writing(tmp_path, names):
    path = tmp_path/'input.tgz'
    with tarfile.open(path, 'w:gz') as archive:
        for name in names:
            item = tarfile.TarInfo(name)
            item.size = 1
            archive.addfile(item, io.BytesIO(b'x'))
    with pytest.raises(ValueError):
        safe_extract(path, tmp_path/'out')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('kind', [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE])
def test_archive_rejects_nonregular_entries(tmp_path, kind):
    path = tmp_path/'input.tgz'
    with tarfile.open(path, 'w:gz') as archive:
        item = tarfile.TarInfo('entry'); item.type = kind; item.linkname = '/elsewhere'
        archive.addfile(item)
    with pytest.raises(ValueError):
        safe_extract(path, tmp_path/'out')


def test_archive_preserves_exact_regular_bytes_and_enforces_size(tmp_path):
    path = tmp_path/'input.tgz'
    with tarfile.open(path, 'w:gz') as archive:
        item = tarfile.TarInfo('nested/file'); item.size = 3
        archive.addfile(item, io.BytesIO(b'abc'))
    with pytest.raises(ValueError, match='budget'):
        safe_extract(path, tmp_path/'small', max_bytes=2)
    safe_extract(path, tmp_path/'out', max_bytes=3)
    assert (tmp_path/'out/nested/file').read_bytes() == b'abc'
    with pytest.raises(ValueError, match='new'):
        safe_extract(path, tmp_path/'out')


def test_resume_attaches_existing_job_without_upload_or_start(monkeypatch, tmp_path):
    handle = dict(job_id='a'*32, lease='/unused', payload_sha256='b'*64, request_sha256='c'*64)
    (tmp_path/'remote-job.json').write_text(json.dumps(handle))
    calls = []
    class FakeTransport:
        def __init__(self, *_): pass
        def command(self, *_args, **_kwargs): return {'started': True}
        def operation(self, handle, operation):
            calls.append((handle['job_id'], operation))
            return {'phase': 'running', 'manager_live': True}
        def put(self, *_): pytest.fail('Cannot reupload inputs to a started job')
    monkeypatch.setattr(remote_replay, 'Transport', FakeTransport)
    result = remote_replay.submit_or_resume(tmp_path)
    assert result['job_id'] == 'a'*32
    assert calls == [('a'*32, 'status')]


def test_transient_observation_error_never_submits_replacement(monkeypatch, tmp_path):
    (tmp_path/'remote-job.json').write_text(json.dumps({'deadline_epoch': 1000}))
    calls = []
    def observe(_):
        calls.append('observe')
        if len(calls) == 1: raise subprocess.TimeoutExpired('ssh', 1)
        return {'phase': 'complete'}
    monkeypatch.setattr(remote_replay, 'observe', observe)
    monkeypatch.setattr(remote_replay, 'submit_or_resume', lambda *_: pytest.fail('duplicate submission'))
    monkeypatch.setattr(remote_replay, 'collect', lambda _: {'phase': 'complete'})
    monkeypatch.setattr(remote_replay.time, 'time', lambda: 0)
    monkeypatch.setattr(remote_replay.time, 'sleep', lambda _: None)
    assert remote_replay.wait(tmp_path)['phase'] == 'complete'
    assert calls == ['observe', 'observe']


@pytest.mark.parametrize('stdout', ['', '{"phase":', 'upload in progress\n'])
def test_partial_manager_response_retries_same_live_job(monkeypatch, tmp_path, stdout):
    handle = dict(job_id='a'*32, lease='/unused', deadline_epoch=1000)
    (tmp_path/'remote-job.json').write_text(json.dumps(handle))
    calls = []
    monkeypatch.setattr(remote_replay, 'connection', lambda _: ([], 'worker'))
    def command(args, **kwargs):
        calls.append(args)
        output = stdout if len(calls) == 1 else '{"phase":"complete"}'
        return subprocess.CompletedProcess(args, 0, output, '')
    monkeypatch.setattr(remote_replay.subprocess, 'run', command)
    monkeypatch.setattr(remote_replay, 'submit_or_resume', lambda *_: pytest.fail('duplicate submission'))
    monkeypatch.setattr(remote_replay, 'collect', lambda _: {'phase': 'complete'})
    monkeypatch.setattr(remote_replay.time, 'time', lambda: 0)
    monkeypatch.setattr(remote_replay.time, 'sleep', lambda _: None)
    assert remote_replay.wait(tmp_path)['phase'] == 'complete'
    assert len(calls) == 2 and calls[0] == calls[1]
    assert 'status ' + handle['job_id'] in calls[0][-1]
    assert 'invalid JSON response' in (tmp_path/'observation-error.json').read_text()


def test_collect_rejects_changed_download(monkeypatch, tmp_path):
    handle = dict(job_id='a'*32, lease='/unused', request_sha256='b'*64, payload_sha256='c'*64)
    (tmp_path/'remote-job.json').write_text(json.dumps(handle))
    class FakeTransport:
        def __init__(self, *_): pass
        def operation(self, *_): return {'phase': 'complete', 'path': '/unused', 'sha256': '0'*64}
        def get(self, _remote, local): Path(local).write_bytes(b'corrupt')
    monkeypatch.setattr(remote_replay, 'Transport', FakeTransport)
    with pytest.raises(ValueError, match='checksum'):
        remote_replay.collect(tmp_path)
    assert not (tmp_path/'collected').exists()


def test_status_checks_live_container_if_manager_disappears(monkeypatch, tmp_path):
    from softbody_lab import gpu_job
    (tmp_path/'state.json').write_text(json.dumps({'phase': 'running', 'pid': 1, 'pid_birth': 'old'}))
    monkeypatch.setattr(gpu_job, 'birth', lambda _: None)
    monkeypatch.setattr(gpu_job.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, 'worker running\n', ''))
    assert gpu_job.status(tmp_path)['phase'] == 'orphaned'
    monkeypatch.setattr(gpu_job.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, '', ''))
    assert gpu_job.status(tmp_path)['phase'] == 'lost'


def test_missing_process_birth_is_never_live(monkeypatch, tmp_path):
    from softbody_lab import gpu_job
    (tmp_path/'state.json').write_text(json.dumps({'phase': 'launching', 'pid': 1, 'pid_birth': None}))
    monkeypatch.setattr(gpu_job, 'birth', lambda _: None)
    monkeypatch.setattr(gpu_job.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, '', ''))
    result = gpu_job.status(tmp_path)
    assert not result['manager_live']
    assert result['phase'] == 'lost'


def test_cancelled_uncertain_launch_cannot_execute_late_child(monkeypatch, tmp_path):
    from softbody_lab import gpu_job
    (tmp_path/'state.json').write_text(json.dumps({'phase': 'launching'}))
    assert gpu_job.status(tmp_path)['launch_uncertain']
    monkeypatch.setattr(gpu_job.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, '', ''))
    assert gpu_job.cancel_job(tmp_path)['phase'] == 'cancelled'
    # No job inputs exist; execute must return before reading or launching any.
    gpu_job.execute(tmp_path)
    assert json.loads((tmp_path/'state.json').read_text())['phase'] == 'cancelled'


def test_collect_archives_lost_observation_instead_of_stale_running_state(monkeypatch, tmp_path):
    from softbody_lab import gpu_job
    (tmp_path/'state.json').write_text(json.dumps({'phase': 'running', 'pid': 1, 'pid_birth': 'old'}))
    (tmp_path/'inputs').mkdir()
    (tmp_path/'inputs/job.json').write_text('{}')
    monkeypatch.setattr(gpu_job, 'birth', lambda _: None)
    monkeypatch.setattr(gpu_job.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, '', ''))
    result = gpu_job.collect(tmp_path)
    with tarfile.open(result['path']) as archive:
        state = json.load(archive.extractfile('state.json'))
    assert result['phase'] == state['phase'] == 'lost'
    assert not state['manager_live']


def test_failed_orphan_removal_does_not_report_cancellation(monkeypatch, tmp_path):
    from softbody_lab import gpu_job
    (tmp_path/'state.json').write_text(json.dumps({'phase': 'running', 'pid': 1, 'pid_birth': 'old'}))
    monkeypatch.setattr(gpu_job, 'birth', lambda _: None)
    def docker(args, **kwargs):
        return subprocess.CompletedProcess(args, 0 if args[1] == 'ps' else 1,
            'worker running\n' if args[1] == 'ps' else '', '')
    monkeypatch.setattr(gpu_job.subprocess, 'run', docker)
    assert gpu_job.cancel_job(tmp_path)['phase'] == 'orphaned'


def result_archive(tmp_path, *, collected=False):
    request={'env_id':'Fill-v0','fixture_sha256':'f'*64}
    execution={'phase':'failed','error':'retained compiler failure','wall_time_s':2.5}
    files={'inputs/job.json':json.dumps(request).encode(),'state.json':json.dumps(execution).encode(),'build.log':b'compiler failure\n'}
    archive=tmp_path/'results.tgz'
    with tarfile.open(archive,'w:gz') as output:
        for name,value in files.items():
            member=tarfile.TarInfo(name);member.size=len(value)
            output.addfile(member,io.BytesIO(value))
    handle=dict(job_id='a'*32,lease='/expired-lease',phase='running',payload_sha256='b'*64,
                request_sha256=remote_replay.digest_json(request),deadline_epoch=1)
    if collected:
        handle.update(phase='collected',execution=execution,result_sha256=file_hash(archive))
        safe_extract(archive,tmp_path/'collected')
    else:
        receipt={k:handle[k] for k in ('job_id','request_sha256','payload_sha256')}
        receipt['result']=dict(phase='failed',path='/already-downloaded/results.tgz',sha256=file_hash(archive))
        atomic_json(tmp_path/'collection-receipt.json',receipt)
    atomic_json(tmp_path/'remote-job.json',handle)
    return execution


@pytest.mark.parametrize('fault', [None, 'role', 'dirty', 'commit', 'image', 'payload', 'initial', 'fixture'])
def test_reference_collection_checks_provenance_and_exported_fixture(monkeypatch, tmp_path, fault):
    request = dict(role='reference', env_id='Excavate-v0', reference_commit='a'*40,
                   image='sha256:'+'b'*64, file_sha256={'reference_input':{'initial.npy':'c'*64}})
    payload = 'd'*64
    provenance = dict(role='reference', source_dirty=False, source_commit=request['reference_commit'],
        runtime={'image_id':request['image']}, source_archive_sha256=payload, reference_initial_state_sha256='c'*64)
    if fault == 'role': provenance['role'] = 'candidate'
    if fault == 'dirty': provenance['source_dirty'] = True
    if fault == 'commit': provenance['source_commit'] = '0'*40
    if fault == 'image': provenance['runtime']['image_id'] = 'sha256:'+'0'*64
    if fault == 'payload': provenance['source_archive_sha256'] = '0'*64
    if fault == 'initial': provenance['reference_initial_state_sha256'] = '0'*64
    manifest = dict(provenance=provenance, fixture_sha256='e'*64)
    monkeypatch.setattr(remote_replay, 'validate_trace', lambda _: manifest)
    monkeypatch.setattr(remote_replay, 'load_fixture', lambda _: (
        dict(fixture_sha256=('0' if fault=='fixture' else 'e')*64),None,None,None))
    archive = tmp_path/'results.tgz'
    with tarfile.open(archive, 'w:gz') as target:
        for name, value in {'inputs/job.json':request, 'state.json':{'phase':'complete'}}.items():
            data=json.dumps(value).encode();item=tarfile.TarInfo(name);item.size=len(data)
            target.addfile(item,io.BytesIO(data))
    atomic_json(tmp_path/'remote-job.json',dict(job_id='f'*32, lease='/unused', phase='running',
        payload_sha256=payload, request_sha256=remote_replay.digest_json(request), result_sha256=file_hash(archive)))
    if fault:
        with pytest.raises(ValueError): remote_replay.recover_collected(tmp_path)
        assert json.loads((tmp_path/'remote-job.json').read_text())['phase'] == 'running'
    else:
        assert remote_replay.recover_collected(tmp_path)['phase'] == 'complete'
        assert json.loads((tmp_path/'remote-job.json').read_text())['phase'] == 'collected'


@pytest.mark.parametrize('operation',['collect','wait','submit_or_resume'])
def test_downloaded_receipt_recovers_offline_without_lease(monkeypatch,tmp_path,operation):
    expected=result_archive(tmp_path)
    monkeypatch.setattr(remote_replay,'Transport',lambda *a:pytest.fail('No network needed for pinned local result'))
    actual=getattr(remote_replay,operation)(tmp_path)
    if operation=='submit_or_resume':actual=actual['execution']
    assert actual==expected
    assert inventory(tmp_path/'collected')==archive_inventory(tmp_path/'results.tgz')
    handle=json.loads((tmp_path/'remote-job.json').read_text())
    assert handle['phase']=='collected' and handle['result_sha256']==file_hash(tmp_path/'results.tgz')


def test_previously_collected_handle_recovers_without_receipt(monkeypatch,tmp_path):
    expected=result_archive(tmp_path,collected=True)
    monkeypatch.setattr(remote_replay,'Transport',lambda *a:pytest.fail('Already collected locally'))
    assert remote_replay.collect(tmp_path)==expected


def test_receipt_recovers_legacy_partial_extraction_and_preserves_it(tmp_path):
    expected=result_archive(tmp_path)
    partial=tmp_path/'collected';partial.mkdir();(partial/'build.log').write_text('partial')
    assert remote_replay.recover_collected(tmp_path)==expected
    retained=list(tmp_path.glob('collected-incomplete-*/build.log'))
    assert len(retained)==1 and retained[0].read_text()=='partial'
    assert (partial/'build.log').read_text()=='compiler failure\n'


def test_verified_extraction_mutation_is_not_silently_repaired(tmp_path):
    result_archive(tmp_path,collected=True)
    (tmp_path/'collected/build.log').write_text('changed')
    with pytest.raises(ValueError,match='verified extracted results changed'):
        remote_replay.recover_collected(tmp_path)


def test_changed_receipt_cannot_rebind_a_different_job(tmp_path):
    result_archive(tmp_path)
    path=tmp_path/'collection-receipt.json';value=json.loads(path.read_text());value['job_id']='c'*32;atomic_json(path,value)
    with pytest.raises(ValueError,match='different job inputs'):remote_replay.recover_collected(tmp_path)


def test_changed_pinned_archive_is_rejected(tmp_path):
    result_archive(tmp_path)
    (tmp_path/'results.tgz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum differs'):remote_replay.recover_collected(tmp_path)


def test_collection_receipt_survives_interrupted_download(monkeypatch,tmp_path):
    expected=result_archive(tmp_path)
    payload=(tmp_path/'results.tgz').read_bytes()
    response=json.loads((tmp_path/'collection-receipt.json').read_text())['result']
    (tmp_path/'results.tgz').unlink();(tmp_path/'collection-receipt.json').unlink()
    calls=[]
    class FakeTransport:
        def __init__(self,*a):pass
        def operation(self,*a):calls.append('collect');return response
        def get(self,remote,local):
            calls.append('download')
            if calls.count('download')==1:raise TimeoutError('Interrupted before download')
            local.write_bytes(payload)
    monkeypatch.setattr(remote_replay,'Transport',FakeTransport)
    with pytest.raises(TimeoutError):remote_replay.collect(tmp_path)
    assert (tmp_path/'collection-receipt.json').is_file()
    assert remote_replay.collect(tmp_path)==expected
    assert calls==['collect','download','collect','download']


def test_atomic_extraction_never_publishes_partial_directory(monkeypatch,tmp_path):
    result_archive(tmp_path)
    original=tarfile.TarFile.extractfile;calls=[]
    def interrupted(self,member):
        calls.append(member.name)
        if len(calls)==2:raise KeyboardInterrupt()
        return original(self,member)
    monkeypatch.setattr(tarfile.TarFile,'extractfile',interrupted)
    with pytest.raises(KeyboardInterrupt):safe_extract(tmp_path/'results.tgz',tmp_path/'collected')
    assert not (tmp_path/'collected').exists()
    assert not list(tmp_path.glob('.collected.extract-*'))


def test_receipt_with_nonterminal_archive_does_not_claim_completion(tmp_path):
    result_archive(tmp_path)
    archive=tmp_path/'results.tgz'
    with tarfile.open(archive,'r:gz') as source:
        values={m.name:source.extractfile(m).read() for m in source.getmembers()}
    values['state.json']=b'{"phase":"running"}'
    with tarfile.open(archive,'w:gz') as target:
        for name,value in values.items():
            m=tarfile.TarInfo(name);m.size=len(value);target.addfile(m,io.BytesIO(value))
    receipt=tmp_path/'collection-receipt.json';value=json.loads(receipt.read_text());value['result']['sha256']=file_hash(archive);atomic_json(receipt,value)
    with pytest.raises(ValueError,match='not terminal'):remote_replay.recover_collected(tmp_path)


def test_prebuilt_actor_cannot_silently_ignore_candidate_cpp_changes(monkeypatch,tmp_path):
    lease=tmp_path/'lease.json';atomic_json(lease,dict(terminate_at_epoch=remote_replay.time.time()+1000))
    monkeypatch.setattr(remote_replay,'load_fixture',lambda _:(dict(fixture=dict(env_id='Fill-v0')),None,None,None))
    def source(candidate,destination):
        cpp=destination/'tools/softbody/native/actor_bridge.cpp';cpp.parent.mkdir(parents=True);cpp.write_text('// changed candidate C++\n')
        return 'a'*40
    monkeypatch.setattr(remote_replay,'copy_source',source)
    spec=dict(build='actor-v1',sha256='a'*64,cpp_sha256='b'*64)
    output=tmp_path/'job'
    with pytest.raises(ValueError,match=r'does not match candidate C\+\+'):
        remote_replay.prepare(tmp_path/'candidate',tmp_path/'fixture',output,lease,'sha256:'+'a'*64,native_actor_extension=spec)
    assert not (output/'remote-job.json').exists()


@pytest.mark.parametrize('changed',['config','cpp'])
def test_prebuilt_cooked_adapter_cannot_ignore_candidate_changes(monkeypatch,tmp_path,changed):
    from softbody_lab.cooked_extensions import source_digest
    lease=tmp_path/'lease.json';atomic_json(lease,dict(terminate_at_epoch=remote_replay.time.time()+1000))
    monkeypatch.setattr(remote_replay,'load_fixture',lambda _:(dict(fixture=dict(env_id='Pour-v0')),None,None,None))
    config={'files':{'bridge.cpp':'a'*64}}
    spec=dict(build='cooked-v1',sha256='b'*64,source_config_sha256=source_digest(config))
    def source(candidate,destination):
        directory=destination/'tools/softbody/native/cooked';directory.mkdir(parents=True)
        actual={'files':{'bridge.cpp':'c'*64}} if changed=='config' else config
        (directory/'source.json').write_text(json.dumps(actual));(directory/'bridge.cpp').write_text('// changed\n')
        return 'a'*40
    monkeypatch.setattr(remote_replay,'copy_source',source);output=tmp_path/'job'
    with pytest.raises(ValueError,match='does not match candidate source'):
        remote_replay.prepare(tmp_path/'candidate',tmp_path/'fixture',output,lease,'sha256:'+'a'*64,
            native_cooked_extension=spec,cooked_pack_sha256='d'*64)
    assert not (output/'remote-job.json').exists()


def test_runtime_snapshot_omits_unrelated_checkout_files_and_keeps_native_build(tmp_path):
    candidate=tmp_path/'candidate';candidate.mkdir()
    subprocess.run(['git','init','-q',str(candidate)],check=True)
    files=['mani_skill/envs/softbody/capture.py','warp_maniskill/build_lib.py','setup.py','README.md','LICENSE',
           'tools/softbody/native/actor_bridge.cpp','tools/softbody/native/vendor/LICENSE',
           'docs/private-note.txt','docs/source/_static/thumbnail.png','notes.csv','.softbody-tmp/coding.log',
           'tools/softbody/verification/private-protocol.json']
    for name in files:
        path=candidate/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name+'\n')
    subprocess.run(['git','add','.'],cwd=candidate,check=True)
    subprocess.run(['git','-c','user.name=Test','-c','user.email=test@localhost','commit','-qm','test fixture'],cwd=candidate,check=True)
    output=tmp_path/'snapshot';remote_replay.copy_source(candidate,output)
    assert set(inventory(output))==set(files[:7])
    assert all((output/name).read_bytes()==(candidate/name).read_bytes() for name in files[:7])

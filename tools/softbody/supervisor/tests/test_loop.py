import json
import copy
import fcntl
import os
import subprocess
import textwrap
from pathlib import Path
import sys
import tomllib

import pytest

from softbody_lab import loop
from softbody_lab.coding_runner import config_args


def test_permission_config_is_valid_toml_and_has_no_shell_network(tmp_path):
    args = config_args(tmp_path/'candidate')
    entries = {}
    for i in range(0, len(args), 2):
        assert args[i] == '-c'
        entries.update(tomllib.loads(args[i+1]))
    profile = entries['permissions']['softbody']
    assert profile['filesystem'][':root'] == 'deny'
    assert profile['filesystem'][str(tmp_path/'candidate')] == 'write'
    assert not profile['network']['enabled']
    assert entries['approval_policy'] == 'never'
    assert entries['default_permissions'] == 'softbody'


def test_bounded_child_timeout_retains_log_and_terminal_status(tmp_path):
    result = loop.run_bounded([sys.executable, '-c', 'import time; print("started",flush=True); time.sleep(30)'],
                             cwd=tmp_path, output=tmp_path/'child.jsonl', timeout_s=.15)
    assert result['status'] == 'timeout'
    assert result['exit_code'] != 0
    assert 'started' in (tmp_path/'child.jsonl').read_text()
    assert json.loads((tmp_path/'child.process.json').read_text())['status'] == 'timeout'


def setup_run(monkeypatch, tmp_path):
    config = dict(candidate=str(tmp_path/'candidate'), output=str(tmp_path/'output'),
        harness=str(tmp_path/'harness'), lease=str(tmp_path/'lease'), image='sha256:'+'a'*64,
        milestone='Synthetic orchestrator fault test', max_iterations=1, iteration_timeout_s=1,
        worker_timeout_s=1, wall_budget_s=10, gpu_budget_s=10, max_stalled_iterations=1,
        cases=[dict(id='case', reference='/reference', fixture='/fixture', protocol='/protocol')])
    monkeypatch.setattr(loop, 'config_validate', lambda _, **kw: None)
    monkeypatch.setattr(loop, 'permission_probe', lambda *a, **k: {'verified': True})
    monkeypatch.setattr(loop, 'frozen_inputs', lambda _: {'immutable': True})
    monkeypatch.setattr(loop, 'environment', lambda _: {})
    monkeypatch.setattr(loop, 'coding_command', lambda *a, **k: ['fake-agent'])
    monkeypatch.setattr(loop, 'run_bounded', lambda *a, **k: {'status': 'completed', 'exit_code': 0, 'claimed_passed': True})
    monkeypatch.setattr(loop, 'candidate_fingerprint', lambda _: 'unchanged-source')
    Path(config['candidate']).mkdir()
    protocol=tmp_path/'protocol.json';protocol.write_text('{}')
    config['cases'][0]['protocol']=str(protocol)
    return config


def fake_prepare(_candidate, _fixture, output, *_args, **_kwargs):
    output.mkdir()
    loop.write_json(output/'remote-job.json', {'job_id':'a'*32, 'deadline_epoch':loop.time.time()+100})


def test_agent_success_cannot_override_independent_failure(monkeypatch, tmp_path):
    config = setup_run(monkeypatch, tmp_path)
    monkeypatch.setattr(loop.remote_replay, 'prepare', fake_prepare)
    monkeypatch.setattr(loop.remote_replay, 'submit_or_resume', lambda _: None)
    monkeypatch.setattr(loop.remote_replay, 'wait', lambda _: {'phase': 'complete', 'wall_time_s': .1})
    monkeypatch.setattr(loop, 'compare', lambda *a: {'passed': False, 'failures': [{'metric': 'known failure'}]})
    result = loop.run(config)
    assert result['status'] == 'iteration_budget_exhausted'
    assert not result['full_port_complete']
    assert not result['iterations'][0]['cases'][0]['verdict']['passed']


def test_trusted_input_change_stops_before_paid_worker(monkeypatch, tmp_path):
    config = setup_run(monkeypatch, tmp_path)
    snapshots = iter([{'immutable': 1}, {'immutable': 1}, {'immutable': 2}])
    monkeypatch.setattr(loop, 'frozen_inputs', lambda _: next(snapshots))
    monkeypatch.setattr(loop.remote_replay, 'prepare', lambda *a, **k: pytest.fail('Must not launch GPU after oracle change'))
    result = loop.run(config)
    assert result['status'] == 'interrupted_requires_resume'
    assert not result['iterations'][0]['cases']
    assert 'inputs changed' in (Path(config['output'])/'error.json').read_text()


def test_failed_coding_run_does_not_launch_worker(monkeypatch, tmp_path):
    config = setup_run(monkeypatch, tmp_path)
    monkeypatch.setattr(loop, 'run_bounded', lambda *a, **k: {'status': 'timeout', 'exit_code': -15})
    monkeypatch.setattr(loop.remote_replay, 'prepare', lambda *a, **k: pytest.fail('Must not launch GPU after failed coding'))
    assert loop.run(config)['status'] == 'coding_run_failed'


def test_worker_build_failure_is_recorded_without_requiring_trace(monkeypatch, tmp_path):
    config = setup_run(monkeypatch, tmp_path)
    monkeypatch.setattr(loop.remote_replay, 'prepare', fake_prepare)
    monkeypatch.setattr(loop.remote_replay, 'submit_or_resume', lambda _: None)
    monkeypatch.setattr(loop.remote_replay, 'wait', lambda _: {'phase': 'failed', 'error': 'compiler failed', 'wall_time_s': .1})
    monkeypatch.setattr(loop, 'compare', lambda *a: pytest.fail('No trace exists after build failure'))
    result = loop.run(config)
    assert result['status'] == 'iteration_budget_exhausted'
    verdict = result['iterations'][0]['cases'][0]['verdict']
    assert not verdict['passed']
    assert verdict['failures'][0]['error'] == 'compiler failed'


def interrupted_run(monkeypatch,tmp_path,*,during='submit',worker_time=.1):
    config=setup_run(monkeypatch,tmp_path)
    calls=dict(coding=0,prepare=0,submit=[],wait=0,compare=0)
    def coding(*a,**k):
        calls['coding']+=1
        return {'status':'completed','exit_code':0}
    def prepare(*a,**k):calls['prepare']+=1;return fake_prepare(*a,**k)
    def submit(output):
        calls['submit'].append(loop.read_json(output/'remote-job.json')['job_id'])
        if during=='submit' and len(calls['submit'])==1:raise TimeoutError('Unobserved remote launch')
    def wait(_):calls['wait']+=1;return {'phase':'complete','wall_time_s':worker_time}
    def compare(*a):
        calls['compare']+=1
        if during=='compare' and calls['compare']==1:raise RuntimeError('Interrupted host comparison')
        return {'passed':False,'failures':[{'metric':'unchanged failure'}]}
    monkeypatch.setattr(loop,'run_bounded',coding)
    monkeypatch.setattr(loop.remote_replay,'prepare',prepare)
    monkeypatch.setattr(loop.remote_replay,'submit_or_resume',submit)
    monkeypatch.setattr(loop.remote_replay,'wait',wait)
    monkeypatch.setattr(loop,'compare',compare)
    result=loop.run(config)
    assert result['status']=='interrupted_requires_resume'
    return config,calls,result


def test_resume_uncertain_launch_reuses_exact_job_and_coding(monkeypatch,tmp_path):
    config,calls,first=interrupted_run(monkeypatch,tmp_path)
    result=loop.run(config,resume=True)
    assert result['status']=='iteration_budget_exhausted'
    assert calls==dict(coding=1,prepare=1,submit=['a'*32,'a'*32],wait=1,compare=1)
    assert result['gpu_worker_wall_time_s']==.1
    assert result['gpu_reserved_wall_time_s']==0
    assert result['deadline_epoch']==first['deadline_epoch']
    assert not result['iterations'][0]['cases'][0]['verdict']['passed']


def test_resume_comparison_does_not_recharge_or_rerun_worker(monkeypatch,tmp_path):
    config,calls,first=interrupted_run(monkeypatch,tmp_path,during='compare')
    assert first['gpu_worker_wall_time_s']==.1
    result=loop.run(config,resume=True)
    assert result['gpu_worker_wall_time_s']==.1
    assert calls==dict(coding=1,prepare=1,submit=['a'*32],wait=1,compare=2)


def test_changed_resume_budget_rejected_before_remote_work(monkeypatch,tmp_path):
    config,calls,_=interrupted_run(monkeypatch,tmp_path)
    altered=dict(config,gpu_budget_s=100)
    with pytest.raises(ValueError,match='original configuration'):
        loop.run(altered,resume=True)
    assert calls['prepare']==1 and len(calls['submit'])==1


def test_candidate_drift_rejected_before_resuming_job(monkeypatch,tmp_path):
    config,calls,_=interrupted_run(monkeypatch,tmp_path)
    monkeypatch.setattr(loop,'candidate_fingerprint',lambda _: 'external-change')
    result=loop.run(config,resume=True)
    assert result['status']=='interrupted_requires_resume'
    assert len(calls['submit'])==1
    assert 'Candidate changed' in (Path(config['output'])/'error.json').read_text()


def test_changed_oracle_rejected_on_resume(monkeypatch,tmp_path):
    config,calls,_=interrupted_run(monkeypatch,tmp_path)
    monkeypatch.setattr(loop,'frozen_inputs',lambda _: {'immutable':False})
    result=loop.run(config,resume=True)
    assert result['status']=='interrupted_requires_resume'
    assert len(calls['submit'])==1


def test_missing_prepared_handle_never_replaced(monkeypatch,tmp_path):
    config,calls,_=interrupted_run(monkeypatch,tmp_path)
    (Path(config['output'])/'iteration-0000/case/remote-job.json').unlink()
    assert loop.run(config,resume=True)['status']=='interrupted_requires_resume'
    assert calls['prepare']==1 and len(calls['submit'])==1


def test_changed_job_id_is_not_resubmitted(monkeypatch,tmp_path):
    config,calls,_=interrupted_run(monkeypatch,tmp_path)
    handle=Path(config['output'])/'iteration-0000/case/remote-job.json'
    value=loop.read_json(handle);value['job_id']='b'*32;loop.write_json(handle,value)
    assert loop.run(config,resume=True)['status']=='interrupted_requires_resume'
    assert len(calls['submit'])==1


def test_expired_budget_allows_saved_local_comparison_only(monkeypatch,tmp_path):
    config,calls,first=interrupted_run(monkeypatch,tmp_path,during='compare')
    monkeypatch.setattr(loop.time,'time',lambda: first['deadline_epoch']+500)
    result=loop.run(config,resume=True)
    assert result['deadline_epoch']==first['deadline_epoch']
    assert result['gpu_worker_wall_time_s']==.1
    assert calls['coding']==calls['prepare']==calls['wait']==1 and calls['compare']==2


def test_same_output_has_exclusive_supervisor_lock(monkeypatch,tmp_path):
    config,calls,_=interrupted_run(monkeypatch,tmp_path)
    with (Path(config['output'])/'.supervisor.lock').open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='active supervisor'):loop.run(config,resume=True)
    assert len(calls['submit'])==1


def test_unobserved_coder_exit_cannot_start_another_agent(monkeypatch,tmp_path):
    config=setup_run(monkeypatch,tmp_path)
    def interrupt(*a,**k):raise KeyboardInterrupt()
    monkeypatch.setattr(loop,'run_bounded',interrupt)
    assert loop.run(config)['status']=='interrupted_requires_resume'
    monkeypatch.setattr(loop,'run_bounded',lambda *a,**k:pytest.fail('Unobserved coder must not be relaunched'))
    monkeypatch.setattr(loop.remote_replay,'prepare',lambda *a,**k:pytest.fail('Coder may still be live'))
    result=loop.run(config,resume=True)
    assert result['status']=='interrupted_requires_resume'
    assert 'verified successful exit' in (Path(config['output'])/'error.json').read_text()


def test_resume_after_observed_child_exit_before_iteration_checkpoint(monkeypatch,tmp_path):
    real_run=loop.run_bounded
    config=setup_run(monkeypatch,tmp_path)
    def observed_then_interrupt(*a,**k):
        real_run([sys.executable,'-c','print("observed terminal")'],cwd=Path(config['candidate']),
                 output=k['output'],timeout_s=5)
        raise KeyboardInterrupt()
    monkeypatch.setattr(loop,'run_bounded',observed_then_interrupt)
    assert loop.run(config)['status']=='interrupted_requires_resume'
    monkeypatch.setattr(loop,'run_bounded',lambda *a,**k:pytest.fail('Coder already completed'))
    monkeypatch.setattr(loop.remote_replay,'prepare',fake_prepare)
    monkeypatch.setattr(loop.remote_replay,'submit_or_resume',lambda _:None)
    monkeypatch.setattr(loop.remote_replay,'wait',lambda _:{'phase':'complete','wall_time_s':.1})
    monkeypatch.setattr(loop,'compare',lambda *a:{'passed':False,'failures':[]})
    result=loop.run(config,resume=True)
    assert result['status']=='iteration_budget_exhausted'
    assert result['iterations'][0]['agent']['exit_code']==0


def test_interrupted_pure_local_preparation_retained_and_rebuilt(monkeypatch,tmp_path):
    config=setup_run(monkeypatch,tmp_path);calls=[]
    def prepare(*a,**k):
        calls.append('prepare')
        if len(calls)==1:
            a[2].mkdir();(a[2]/'partial.txt').write_text('retained');raise KeyboardInterrupt()
        return fake_prepare(*a,**k)
    monkeypatch.setattr(loop.remote_replay,'prepare',prepare)
    monkeypatch.setattr(loop.remote_replay,'submit_or_resume',lambda _:calls.append('submit'))
    monkeypatch.setattr(loop.remote_replay,'wait',lambda _:{'phase':'complete','wall_time_s':.1})
    monkeypatch.setattr(loop,'compare',lambda *a:{'passed':False,'failures':[]})
    assert loop.run(config)['status']=='interrupted_requires_resume'
    assert loop.run(config,resume=True)['status']=='iteration_budget_exhausted'
    assert calls==['prepare','prepare','submit']
    retained=list((Path(config['output'])/'iteration-0000').glob('case-incomplete-*/partial.txt'))
    assert len(retained)==1 and retained[0].read_text()=='retained'


def test_terminal_resume_never_resets_iteration_budget(monkeypatch,tmp_path):
    config,calls,_=interrupted_run(monkeypatch,tmp_path)
    result=loop.run(config,resume=True);before=copy.deepcopy(calls)
    assert loop.run(config,resume=True)['status']==result['status']
    assert calls==before


def test_elapsed_offline_time_prevents_next_coding_iteration(monkeypatch,tmp_path):
    config=setup_run(monkeypatch,tmp_path)
    config['max_iterations']=2
    calls=[]
    monkeypatch.setattr(loop.remote_replay,'prepare',fake_prepare)
    monkeypatch.setattr(loop.remote_replay,'submit_or_resume',lambda _:None)
    monkeypatch.setattr(loop.remote_replay,'wait',lambda _:{'phase':'complete','wall_time_s':.1})
    def compare(*args):
        calls.append('compare')
        if len(calls)==1:raise KeyboardInterrupt()
        return {'passed':False,'failures':[]}
    monkeypatch.setattr(loop,'compare',compare)
    first=loop.run(config)
    monkeypatch.setattr(loop.time,'time',lambda:first['deadline_epoch']+1)
    monkeypatch.setattr(loop,'run_bounded',lambda *a,**k:pytest.fail('Offline time counts against original budget'))
    result=loop.run(config,resume=True)
    assert result['status']=='resource_budget_exhausted'
    assert len(result['iterations'])==1 and result['iterations'][0]['finalized']


def test_native_process_crash_releases_lock_and_preserves_job(tmp_path):
    # Real supervisor processes and SIGKILL; remote execution is a local fake.
    script=tmp_path/'driver.py'
    script.write_text(textwrap.dedent('''
        import json,os,signal,sys
        from pathlib import Path
        from softbody_lab import loop
        root=Path(sys.argv[1]);resume=sys.argv[2]=='resume'
        candidate=root/'candidate';candidate.mkdir(exist_ok=True)
        protocol=root/'protocol.json';protocol.write_text('{}')
        config=dict(candidate=str(candidate),output=str(root/'output'),harness='/harness',lease='/lease',image='sha256:'+'a'*64,
            milestone='Local crash fault injection',max_iterations=1,max_stalled_iterations=1,iteration_timeout_s=1,
            worker_timeout_s=5,wall_budget_s=30,gpu_budget_s=10,
            cases=[dict(id='case',reference='/reference',fixture='/fixture',protocol=str(protocol))])
        loop.config_validate=lambda *a,**k:None
        loop.permission_probe=lambda *a,**k:{'verified':True}
        loop.frozen_inputs=lambda *a:{'fixture':'unchanged'}
        loop.candidate_fingerprint=lambda *a:'unchanged'
        loop.coding_command=lambda *a,**k:[]
        loop.environment=lambda *a:{}
        def coder(*a,**k):
            with (root/'coding-starts.txt').open('a') as f:f.write('start\\n')
            return dict(status='completed',exit_code=0)
        loop.run_bounded=coder
        def prepare(c,f,output,*a,**k):
            output.mkdir()
            loop.write_json(output/'remote-job.json',dict(job_id='a'*32,deadline_epoch=loop.time.time()+20))
        loop.remote_replay.prepare=prepare
        def submit(output):
            identity=loop.read_json(output/'remote-job.json')['job_id']
            with (root/'submitted-ids.txt').open('a') as f:f.write(identity+'\\n')
        loop.remote_replay.submit_or_resume=submit
        def wait(output):
            if not resume:os.kill(os.getpid(),signal.SIGKILL)
            return dict(phase='complete',wall_time_s=.25)
        loop.remote_replay.wait=wait
        loop.compare=lambda *a:dict(passed=False,failures=[dict(metric='retained failure')])
        print(json.dumps(loop.run(config,resume=resume)))
    '''))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(Path(loop.__file__).resolve().parent.parent))
    first=subprocess.run([sys.executable,str(script),str(tmp_path),'start'],env=env,capture_output=True,text=True,timeout=10)
    assert first.returncode==-9,first.stderr
    checkpoint=loop.read_json(tmp_path/'output/state.json')
    assert checkpoint['iterations'][0]['pending']['job_id']=='a'*32
    second=subprocess.run([sys.executable,str(script),str(tmp_path),'resume'],env=env,capture_output=True,text=True,timeout=10)
    assert second.returncode==0,second.stderr
    result=json.loads(second.stdout)
    assert result['deadline_epoch']==checkpoint['deadline_epoch']
    assert result['status']=='iteration_budget_exhausted'
    assert result['gpu_worker_wall_time_s']==.25
    assert (tmp_path/'coding-starts.txt').read_text().splitlines()==['start']
    assert (tmp_path/'submitted-ids.txt').read_text().splitlines()==['a'*32,'a'*32]


def test_candidate_fingerprint_detects_source_mode_and_deletion(tmp_path):
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    (tmp_path/'mani_skill').mkdir()
    source=tmp_path/'mani_skill/module.py';source.write_text('value=1\n')
    subprocess.run(['git','add','mani_skill/module.py'],cwd=tmp_path,check=True)
    before=loop.candidate_fingerprint(tmp_path)
    source.write_text('value=2\n')
    changed=loop.candidate_fingerprint(tmp_path)
    source.chmod(0o755)
    executable=loop.candidate_fingerprint(tmp_path)
    source.unlink()
    deleted=loop.candidate_fingerprint(tmp_path)
    assert len({before,changed,executable,deleted})==4


@pytest.mark.parametrize('elapsed',[-1,float('nan'),float('inf'),True])
def test_invalid_worker_time_cannot_increase_remaining_budget(monkeypatch,tmp_path,elapsed):
    config=setup_run(monkeypatch,tmp_path)
    monkeypatch.setattr(loop.remote_replay,'prepare',fake_prepare)
    monkeypatch.setattr(loop.remote_replay,'submit_or_resume',lambda _:None)
    monkeypatch.setattr(loop.remote_replay,'wait',lambda _:{'phase':'complete','wall_time_s':elapsed})
    result=loop.run(config)
    assert result['status']=='interrupted_requires_resume'
    assert result['gpu_worker_wall_time_s']==0
    assert result['gpu_reserved_wall_time_s']==config['worker_timeout_s']
    assert 'Invalid worker elapsed time' in (Path(config['output'])/'error.json').read_text()

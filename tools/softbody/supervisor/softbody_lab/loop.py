"""Bounded local Codex iterations with isolated, recoverable remote GPU replays.

Reference inputs, evaluator and thresholds stay outside the coding worker's
permissions. Development passes never imply full-port or held-out acceptance.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from .artifacts import digest_json, read_json, sha256, validate_trace, write_json
from .coding_runner import command as coding_command, environment, permission_probe
from .compare import compare
from .fixtures import load_fixture
from .job_archive import inventory
from .native_extensions import validate_actor_spec
from .cooked_extensions import validate_spec as validate_cooked_spec
from . import remote_replay


def separate(candidate, trusted):
    a, b = Path(candidate).resolve(), Path(trusted).resolve()
    if a == b or a in b.parents or b in a.parents:
        raise ValueError('Candidate checkout and trusted data must have separate directory trees')


def run_bounded(command, *, cwd, output, timeout_s, input_text=None, env=None):
    started = time.monotonic()
    status, error, process = 'failed', None, None
    with Path(output).open('wb') as stream:
        try:
            process = subprocess.Popen(command, cwd=cwd, env=env,
                stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            write_json(Path(output).with_suffix('.process.json'), {'pid': process.pid,
                'started_at_epoch': time.time(), 'status': 'running'})
            process.communicate(input_text.encode() if input_text is not None else None, timeout=timeout_s)
            status = 'completed'
        except subprocess.TimeoutExpired:
            status = 'timeout'
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
    result = {'status': status, 'exit_code': process.returncode if process else None,
              'wall_time_s': time.monotonic()-started, 'log_sha256': sha256(Path(output)), 'error': error}
    write_json(Path(output).with_suffix('.process.json'), result)
    return result


def config_validate(config, *, require_live_lease=True):
    validate_actor_spec(config.get('native_actor_extension'))
    if config.get('native_cooked_extension') is not None:
        validate_cooked_spec(config['native_cooked_extension'])
        value=config.get('cooked_pack_sha256')
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Cooked replay requires a pinned pack manifest checksum')
    required = ('candidate', 'harness', 'output', 'image', 'lease', 'cases', 'milestone',
                'max_iterations', 'iteration_timeout_s', 'worker_timeout_s',
                'wall_budget_s', 'gpu_budget_s', 'max_stalled_iterations')
    if set(required)-config.keys():
        raise ValueError('Missing loop configuration: '+str(sorted(set(required)-config.keys())))
    for key in required[7:]:
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Positive finite budget required: '+key)
    for key in ('max_iterations', 'max_stalled_iterations'):
        if type(config[key]) is not int:
            raise ValueError('Iteration limits must be integers')
    candidate = Path(config['candidate']).resolve()
    for key in ('harness', 'output', 'lease'):
        separate(candidate, config[key])
    if not (candidate/'.git').exists() or not (Path(config['harness'])/'softbody_lab/runner.py').is_file():
        raise ValueError('Git candidate and frozen capture harness are required')
    if not config['cases']:
        raise ValueError('At least one development case is required')
    ids = set()
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', config['image']):
        raise ValueError('Worker image must be an immutable ID')
    for case in config['cases']:
        if (not re.fullmatch(r'[a-zA-Z0-9_-]+', case['id']) or case['id'] in ids
                or case.get('split') != 'development'):
            raise ValueError('Cases must have unique safe IDs and use only the development split')
        ids.add(case['id'])
        for key in ('reference', 'protocol', 'fixture'):
            separate(candidate, case[key])
        f, _, _, _ = load_fixture(Path(case['fixture']))
        backend = case.get('candidate_sim_backend')
        if backend not in (None, 'physx_cpu', 'physx_cuda'):
            raise ValueError('Case candidate backend must be physx_cpu or physx_cuda')
        declared = f['fixture'].get('env_kwargs', {})
        if backend is not None and 'sim_backend' in declared and declared['sim_backend'] != backend:
            raise ValueError('Case candidate backend conflicts with the frozen fixture')
        p = read_json(Path(case['protocol']))
        reference = validate_trace(Path(case['reference']))
        if (p.get('calibrated') is not True or p['fixture_sha256'] != f['fixture_sha256']
                or p['reference_manifest_sha256'] != digest_json(reference)
                or not compare(Path(case['reference']), Path(case['reference']), p)['passed']):
            raise ValueError('Each case needs a valid frozen reference and matching calibrated protocol')
    for context in config.get('readable_context', []):
        separate(candidate, context)
        for case in config['cases']:
            for name in ('reference', 'protocol', 'fixture'):
                separate(context, case[name])
        separate(context, config['harness'])
        separate(context, config['output'])
    lease = read_json(Path(config['lease']))
    if require_live_lease and (lease['state'] != 'active' or time.time()+120 >= lease['terminate_at_epoch']):
        raise ValueError('A live existing lease with time remaining is required')


def frozen_inputs(config):
    paths = [Path(config['harness']), Path(__file__).parent]
    paths += [Path(case[key]) for case in config['cases'] for key in ('reference', 'protocol', 'fixture')]
    result = {}
    for path in paths:
        if path.is_dir():
            values = inventory(path)
            result[str(path.resolve())] = {k:v for k,v in values.items() if '__pycache__' not in Path(k).parts}
        else:
            result[str(path.resolve())] = sha256(path)
    return result


def candidate_fingerprint(candidate):
    """Hash the same runtime/package/native-build files used by remote snapshots."""
    files = {}
    for name in remote_replay.source_paths(candidate):
        relative = Path(name)
        path = candidate/relative
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != candidate.parent):
            raise ValueError('Candidate source cannot follow symlinks')
        files[name] = [sha256(path), bool(path.stat().st_mode & 0o111)] if path.exists() else None
    return digest_json(files)


def run(config, *, resume=False):
    config_validate(config, require_live_lease=not resume)
    candidate = Path(config['candidate']).resolve()
    root = Path(config['output']).resolve()
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
    elif not (root/'state.json').is_file():
        raise ValueError('No resumable checkpoint; inspect the existing process and artifacts')
    # The kernel releases this lock after a supervisor crash. It does not prove
    # a detached coding child has stopped; its separately recorded exit must.
    with (root/'.supervisor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This loop already has an active supervisor') from exc
        return _run_locked(config, candidate, root, resume)


def _run_locked(config, candidate, root, resume):
    if resume:
        state = read_json(root/'state.json')
        if state.get('schema_version') != 1 or state['config_sha256'] != digest_json(config):
            raise ValueError('Resume requires the original configuration and budgets')
        frozen = read_json(root/'trusted-inputs.json')
        if digest_json(frozen) != state['trusted_inputs_sha256']:
            raise ValueError('Saved trusted-input inventory changed')
    else:
        started = time.time()
        write_json(root/'config.json', config)
        probe = permission_probe(candidate, root, root/'permission-probe.log',
            readable=config.get('readable_context', []), python_executable=config.get('python_executable'),
            runtime_modules=config.get('runtime_modules', []))
        write_json(root/'permissions.json', probe)
        frozen = frozen_inputs(config)
        write_json(root/'trusted-inputs.json', frozen)
        state = dict(schema_version=1, config_sha256=digest_json(config), trusted_inputs_sha256=digest_json(frozen),
            started_at_epoch=started, deadline_epoch=started+config['wall_budget_s'], status='running',
            candidate_sha256=candidate_fingerprint(candidate), best_passes=-1, stalled=0,
            feedback=config.get('initial_feedback', {'status': 'initial'}), iterations=[], errors=[])

    def save():
        write_json(root/'state.json', state)
        for record in state['iterations']:
            write_json(root/f'iteration-{record["iteration"]:04d}'/'report.json', record)

    def spent():
        total = 0.
        for record in state['iterations']:
            rows = record['cases'] + ([record['pending']] if 'pending' in record else [])
            for row in rows:
                if 'worker' in row:
                    total += row['worker'].get('wall_time_s', row.get('reserved_s', config['worker_timeout_s']))
        return total

    def remaining():
        return state['deadline_epoch']-time.time()

    def check_frozen():
        if frozen_inputs(config) != frozen:
            raise RuntimeError('Trusted verification inputs changed during coding or replay')

    def finish():
        pending = sum(r.get('pending', {}).get('reserved_s', 0.) for r in state['iterations'] if 'worker' not in r.get('pending', {}))
        result = dict(status=state['status'], milestone=config['milestone'], gpu_worker_wall_time_s=spent(),
            gpu_reserved_wall_time_s=pending, wall_time_s=max(0., time.time()-state['started_at_epoch']),
            started_at_epoch=state['started_at_epoch'], deadline_epoch=state['deadline_epoch'],
            iterations=state['iterations'], config_sha256=state['config_sha256'], full_port_complete=False)
        write_json(root/'report.json', result)
        return result

    if resume and state['status'] not in {'running', 'interrupted_requires_resume'}:
        return finish()
    state['status'] = 'running'
    save()
    try:
        check_frozen()
        active = state['iterations'][-1] if state['iterations'] and not state['iterations'][-1].get('finalized') else None
        if (not active or active['agent'] is not None) and candidate_fingerprint(candidate) != state['candidate_sha256']:
            raise RuntimeError('Candidate changed outside the recorded coding run; inspect before resuming')
        while True:
            if active is None:
                iteration = len(state['iterations'])
                if iteration >= config['max_iterations']:
                    state['status'] = 'iteration_budget_exhausted'
                    break
                if remaining() <= 0 or spent() >= config['gpu_budget_s']:
                    state['status'] = 'resource_budget_exhausted'
                    break
                work = root/f'iteration-{iteration:04d}'
                if work.exists() and any(work.iterdir()):
                    raise RuntimeError('Iteration artifacts exist without a checkpoint; inspect before coding')
                work.mkdir(exist_ok=True)
                active = dict(iteration=iteration, agent=None, cases=[])
                state['iterations'].append(active)
                save()  # Intent precedes process launch; an unobserved child is never rerun.
                prompt = (
                    'Implement the next focused change for the ManiSkill soft-body port.\n'
                    f'Milestone: {config["milestone"]}\n'
                    'Preserve real scale, mass, material models and two-way physical coupling. '
                    'Use drive targets and contacts; pose/state assignments are reset-only. '
                    'Never substitute reported physics state or change acceptance thresholds to pass. '
                    'Keep changes inside this checkout. Do not publish, push, create other tasks, '
                    'or inspect verification data. Only the independent evaluator can issue a verdict. '
                    'If no justified fix follows from the evidence, report the limitation honestly.\n'
                    'Read-only source/dependency context: '+json.dumps(config.get('readable_context', []))+'\n'
                    'Verified Python executable: '+str(config.get('python_executable', 'python3'))+'\n'
                    f'Coding time limit: {min(remaining(), config["iteration_timeout_s"]):.0f} seconds. '
                    'Finish the focused deliverable and exit before that deadline; do not start an open-ended investigation.\n'
                    'Independent development feedback:\n'+json.dumps(state['feedback']))
                write_json(work/'progress.json', {'phase': 'coding', 'iteration': iteration})
                active['agent'] = run_bounded(coding_command(candidate, readable=config.get('readable_context', []),
                    model=config.get('model'), reasoning=config.get('reasoning')), cwd=candidate,
                    env=environment(candidate), output=work/'codex.jsonl', input_text=prompt,
                    timeout_s=min(remaining(), config['iteration_timeout_s']))
                state['candidate_sha256'] = candidate_fingerprint(candidate)
                save()
            work = root/f'iteration-{active["iteration"]:04d}'
            if active['agent'] is None:
                process = work/'codex.process.json'
                agent = read_json(process) if process.is_file() else {}
                if (agent.get('status') != 'completed' or agent.get('exit_code') != 0
                        or not (work/'codex.jsonl').is_file() or agent.get('log_sha256') != sha256(work/'codex.jsonl')):
                    raise RuntimeError('Coding process lacks a verified successful exit; inspect it before resuming')
                active['agent'] = agent
                state['candidate_sha256'] = candidate_fingerprint(candidate)
                save()
            agent = active['agent']
            if agent['status'] != 'completed' or agent['exit_code'] != 0:
                state['status'] = 'coding_run_failed'
                break
            check_frozen()
            for case in config['cases']:
                if any(c['id'] == case['id'] for c in active['cases']):
                    continue
                output = work/case['id']
                if 'pending' not in active:
                    available = min(config['worker_timeout_s'], config['gpu_budget_s']-spent(), remaining())
                    if available <= 0:
                        state['status'] = 'resource_budget_exhausted'
                        break
                    active['pending'] = dict(id=case['id'], reserved_s=available, phase='preparing')
                    save()
                pending = active['pending']
                if pending['id'] != case['id']:
                    raise RuntimeError('Pending case does not match the frozen case order')
                if 'worker' not in pending:
                    if not (output/'remote-job.json').exists():
                        if pending['phase'] != 'preparing':
                            raise RuntimeError('Prepared job handle disappeared; do not submit a replacement')
                        if remaining() <= 0:
                            state['status'] = 'resource_budget_exhausted'
                            break
                        if output.exists():
                            # No handle was published, and submission occurs only
                            # after phase=prepared is persisted. Retain partial files.
                            backup = work/(case['id']+'-incomplete-'+str(time.time_ns()))
                            output.rename(backup)
                        remote_replay.prepare(candidate, Path(case['fixture']), output, Path(config['lease']),
                            config['image'], timeout_s=min(pending['reserved_s'], remaining()), harness=Path(config['harness']),
                            native_actor_extension=config.get('native_actor_extension'),
                            native_cooked_extension=(config.get('native_cooked_extension')
                                if config.get('native_cooked_extension') is not None
                                and load_fixture(Path(case['fixture']))[0]['fixture']['env_id']=='Pour-v0' else None),
                            cooked_pack_sha256=config.get('cooked_pack_sha256'),
                            candidate_sim_backend=case.get('candidate_sim_backend'))
                    handle = read_json(output/'remote-job.json')
                    if 'job_id' in pending and pending['job_id'] != handle['job_id']:
                        raise RuntimeError('Prepared remote job identity changed')
                    pending.update(phase='prepared', job_id=handle['job_id'])
                    save()
                    write_json(work/'progress.json', {'phase': 'gpu_replay', 'case': case['id'], 'job_id': handle['job_id']})
                    # After the original deadline, observe/collect this exact job.
                    # Do not attempt to start a job whose launch was unobserved.
                    cached = remote_replay.recover_collected(output)
                    if cached is None and time.time() < min(state['deadline_epoch'], handle['deadline_epoch']):
                        remote_replay.submit_or_resume(output)
                    worker = cached if cached is not None else remote_replay.wait(output)
                    charge = worker.get('wall_time_s', pending['reserved_s'])
                    if type(charge) not in (int, float) or not math.isfinite(charge) or charge < 0:
                        raise RuntimeError('Invalid worker elapsed time; preserve job for inspection')
                    pending.update(worker=worker, phase='collected')
                    save()  # Persist time once, before a possibly interrupted comparison.
                worker = pending['worker']
                check_frozen()
                if worker['phase'] == 'complete':
                    verdict = compare(Path(case['reference']), output/'collected/output/trace', read_json(Path(case['protocol'])))
                else:
                    verdict = {'passed': False, 'failures': [{'metric': 'worker_execution',
                        'phase': worker['phase'], 'error': worker.get('error')}]}
                write_json(output/'verdict.json', verdict)
                active['cases'].append(dict(pending, phase='evaluated', verdict=verdict))
                del active['pending']
                save()
            if state['status'] == 'resource_budget_exhausted':
                break
            passes = sum(c['verdict']['passed'] for c in active['cases'])
            state['feedback'] = {'milestone': config['milestone'], 'cases': active['cases']}
            state['stalled'] = 0 if passes > state['best_passes'] else state['stalled']+1
            state['best_passes'] = max(passes, state['best_passes'])
            active['finalized'] = True
            if passes == len(config['cases']):
                state['status'] = 'development_gate_passed_requires_acceptance'
                break
            if state['stalled'] >= config['max_stalled_iterations']:
                state['status'] = 'stalled_requires_review'
                break
            save()
            active = None
    except BaseException as exc:
        state['status'] = 'interrupted_requires_resume'
        error = {'type': type(exc).__name__, 'message': str(exc), 'time': time.time()}
        state['errors'].append(error)
        write_json(root/'error.json', error)
    save()
    return finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--resume', action='store_true', help='Resume the same checkpoint without resetting budgets or replacing jobs')
    args = parser.parse_args()
    result = run(read_json(args.config), resume=args.resume)
    print(json.dumps({k:v for k,v in result.items() if k != 'iterations'}, indent=2))
    return 0 if result['status'] == 'development_gate_passed_requires_acceptance' else 1


if __name__ == '__main__':
    raise SystemExit(main())

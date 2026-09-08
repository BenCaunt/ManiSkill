"""Trusted Linux job manager. Does not import candidate Python on the host.

Runs once per immutable job directory. Status survives SSH disconnects; repeated
start requests attach to the same job. Cloud/Codex credentials are never mounted.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time

# The deployment copies this helper beside the manager, outside all containers.
if __package__:
    from .job_archive import atomic_json, file_hash, inventory, safe_extract
    from .native_extensions import stage_actor
    from .cooked_extensions import stage as stage_cooked, pack_path as cooked_pack_path
else:
    from job_archive import atomic_json, file_hash, inventory, safe_extract
    from native_extensions import stage_actor
    from cooked_extensions import stage as stage_cooked, pack_path as cooked_pack_path

ROOT = Path('/home/ubuntu/softbody/supervisor-jobs')
ASSETS = Path('/home/ubuntu/softbody/checkouts/ManiSkill2/mani_skill2/assets')
PACKS = {
    'Hang-v0': Path('/home/ubuntu/softbody/legacy-data/hang-v1'),
    'Pour-v0': Path('/home/ubuntu/softbody/legacy-data/pour-v1'),
    'Write-v0': Path('/home/ubuntu/softbody/legacy-data/write-v1'),
    'Pinch-v0': Path('/home/ubuntu/softbody/records/pinch-asset-pack-v2/pack'),
}
LEVELS = {
    'Write-v0': Path('/home/ubuntu/softbody/records/write-asset-pack-v2/levels'),
    'Pinch-v0': PACKS['Pinch-v0']/'levels',
}
TERMINAL = {'complete', 'failed', 'cancelled', 'lost'}


def job_path(identifier):
    if not re.fullmatch(r'[0-9a-f]{32}', identifier):
        raise ValueError('Invalid job ID')
    return ROOT/identifier


def read(path):
    return json.loads(Path(path).read_text())


def birth(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (FileNotFoundError, IndexError):
        return None


def publish(job, phase, **values):
    state = read(job/'state.json') if (job/'state.json').exists() else {}
    state.update(phase=phase, updated_at_epoch=time.time(), **values)
    atomic_json(job/'state.json', state)
    return state


def status(job):
    state = read(job/'state.json') if (job/'state.json').exists() else {'phase': 'staged'}
    if state['phase'] not in TERMINAL and state.get('pid'):
        actual_birth = birth(state['pid'])
        state['manager_live'] = actual_birth is not None and actual_birth == state.get('pid_birth')
        if not state['manager_live']:
            # Do not infer that a container stopped when its observer vanished.
            result = subprocess.run(['docker', 'ps', '-a', '--filter', f'label=softbody.job={job.name}',
                                     '--format', '{{.Names}} {{.State}}'], capture_output=True, text=True, timeout=15, check=True)
            state['containers'] = result.stdout.splitlines()
            state['phase'] = 'orphaned' if any(line.endswith(' running') for line in state['containers']) else 'lost'
    elif state['phase'] == 'launching':
        # A launcher may disappear between spawning and recording the child PID.
        # Do not call this lost: the child could still start. Cancellation takes
        # the launch lock and publishes a terminal state that a late child honors.
        state['launch_uncertain'] = True
    return state


def container_base(job, request, name):
    return ['docker', 'run', '--rm', '--init', '--name', name, '--label', f'softbody.job={job.name}',
            '--gpus', 'device=0', '--runtime=nvidia', '--network=none', '--read-only',
            '--cap-drop=ALL', '--security-opt=no-new-privileges', '--pids-limit=512',
            '--memory=24g', '--cpus=8', '--user', f'{os.getuid()}:{os.getgid()}',
            '--tmpfs', '/tmp:rw,nosuid,size=6g', '--shm-size=1g',
            '-e', 'HOME=/tmp', '-e', 'PYTHONDONTWRITEBYTECODE=1',
            '-e', 'XDG_CACHE_HOME=/cache', '-e', 'MPLCONFIGDIR=/cache/matplotlib',
            '-v', f'{job}/cache:/cache', '-e', f'SOFTBODY_IMAGE_ID={request["image"]}']


def bounded_container(job, command, name, log, deadline):
    seconds = int(deadline-time.time())
    if seconds < 1:
        raise TimeoutError('Job or lease deadline reached')
    with (job/log).open('wb') as stream:
        # Host timeout is paired with cleanup. The container has its own timeout
        # too, so loss of this manager does not leave an unbounded command.
        try:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=seconds)
            if result.returncode:
                raise RuntimeError(f'{log} exited {result.returncode}')
        finally:
            subprocess.run(['docker', 'rm', '--force', name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=20)


def execute(job):
    with (job/'launch.lock').open('a') as launch_lock:
        fcntl.flock(launch_lock, fcntl.LOCK_EX)
        if read(job/'state.json')['phase'] in TERMINAL:
            return
        request = read(job/'inputs/job.json')
        deadline = request['deadline_epoch']
        started = time.time()
        if started >= deadline:
            publish(job, 'failed', error='Deadline reached before manager started',
                    finished_at_epoch=started, wall_time_s=0.)
            return
        publish(job, 'queued', pid=os.getpid(), pid_birth=birth(os.getpid()), started_at_epoch=started)

    def cancel(*_):
        raise InterruptedError('Job cancelled')
    signal.signal(signal.SIGTERM, cancel)
    lock = (ROOT/'gpu.lock').open('a')
    phase, error = 'failed', None
    try:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() >= deadline:
                    raise TimeoutError('Budget exhausted waiting for GPU')
                time.sleep(1)
        source = job/'inputs/source'
        # Use an honest, separately labelled snapshot commit; no local .git,
        # credentials, hooks, or remote configuration are transmitted.
        environment = {**os.environ, 'GIT_AUTHOR_DATE': '2000-01-01T00:00:00Z',
                       'GIT_COMMITTER_DATE': '2000-01-01T00:00:00Z', 'GIT_CONFIG_NOSYSTEM': '1',
                       'GIT_CONFIG_GLOBAL': '/dev/null'}
        for command in [['git', 'init', '-q', '--initial-branch=snapshot'],
                        ['git', '-c', 'core.hooksPath=/dev/null', 'add', '-f', '--', '.'],
                        ['git', '-c', 'core.hooksPath=/dev/null', '-c', 'user.name=Softbody snapshot',
                         '-c', 'user.email=snapshot@localhost', 'commit', '-qm', 'Frozen worker input snapshot']]:
            subprocess.run(command, cwd=source, env=environment, check=True, capture_output=True, timeout=60)
        for directory in ('cache', 'output', 'binary'):
            (job/directory).mkdir()
        extension = stage_actor(request.get('native_actor_extension'), ROOT.parent, job/'native-extension')
        if extension is not None:
            publish(job, 'preparing', native_actor_extension=extension)
        cooked=None; cooked_pack=None
        if request.get('native_cooked_extension') is not None:
            if request['env_id']!='Pour-v0':raise ValueError('Cooked bottle adapter is only declared for Pour')
            cooked=stage_cooked(request['native_cooked_extension'],ROOT.parent,job/'cooked-extension')
            cooked_pack=cooked_pack_path(request['native_cooked_extension'],ROOT.parent,request['cooked_pack_sha256'])
            publish(job,'preparing',native_cooked_extension=cooked,cooked_pack_sha256=request['cooked_pack_sha256'])
        # Build candidate native sources in a separate writable copy. Only the
        # resulting ELF library is mounted into the read-only runtime snapshot.
        shutil.copytree(source/'warp_maniskill', job/'build/warp_maniskill')
        name = f'softbody-build-{job.name}'
        publish(job, 'building', container=name)
        command = container_base(job, request, name)+[
            '-v', f'{job}/build:/build', '-e', 'PYTHONPATH=/build/warp_maniskill',
            request['image'], 'timeout', '--kill-after=10s', str(max(1, int(deadline-time.time()))),
            'python', '/build/warp_maniskill/build_lib.py', '--cuda_path', '/usr/local/cuda', '--mode', 'release']
        bounded_container(job, command, name, 'build.log', deadline)
        library = job/'build/warp_maniskill/warp/bin/warp.so'
        if library.is_symlink() or not library.is_file() or library.stat().st_size > 512*1024**2:
            raise ValueError('Missing or invalid built Warp library')
        shutil.copyfile(library, job/'binary/warp.so')
        (source/'warp_maniskill/warp/bin').mkdir(exist_ok=True)
        (source/'warp_maniskill/warp/bin/warp.so').touch()
        with (source/'.git/info/exclude').open('a') as excluded:
            excluded.write('\n/warp_maniskill/warp/bin/\n')
        env_id = request['env_id']
        name = f'softbody-replay-{job.name}'
        publish(job, 'running', container=name, binary_sha256=file_hash(job/'binary/warp.so'))
        command = container_base(job, request, name)+[
            '-v', f'{source}:/source:ro', '-v', f'{job}/inputs/harness:/harness:ro',
            '-v', f'{job}/inputs/fixture:/fixture:ro', '-v', f'{job}/output:/output',
            '-v', f'{job}/binary/warp.so:/source/warp_maniskill/warp/bin/warp.so:ro',
            '-v', f'{ASSETS}:/legacy-assets:ro', '-e', 'MANISKILL_LEGACY_ASSET_DIR=/legacy-assets',
            '-e', 'PYTHONPATH='+('/native-extension:' if extension else '')+('/cooked-extension:' if cooked else '')+'/harness:/source:/source/warp_maniskill',
            '-e', f'SOFTBODY_SOURCE_ARCHIVE_SHA256={read(job/"identity.json")["payload_sha256"]}']
        if extension is not None:
            command += ['-v', f'{job}/native-extension:/native-extension:ro']
        if env_id in PACKS:
            command += ['-v', f'{PACKS[env_id]}:/legacy-data:ro', '-e', 'MANISKILL_LEGACY_MPM_DATA=/legacy-data']
        if cooked is not None:
            command += ['-v',f'{job}/cooked-extension:/cooked-extension:ro',
                        '-v',f'{cooked_pack}:/cooked-pack:ro',
                        '-e','MANISKILL_BOTTLE_COLLISION_DIR=/cooked-pack']
        if env_id in LEVELS:
            command += ['-v', f'{LEVELS[env_id]}:/levels:ro']
        command += [request['image'], 'timeout', '--kill-after=10s', str(max(1, int(deadline-time.time()))),
                    'python', '-m', 'softbody_lab', 'capture', '--source=/source', '--role=candidate',
                    '--replay=/fixture', '--output=/output/trace']
        bounded_container(job, command, name, 'capture.log', deadline)
        phase = 'complete'
    except InterruptedError as exc:
        phase, error = 'cancelled', str(exc)
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        lock.close()
        publish(job, phase, error=error, finished_at_epoch=time.time(), wall_time_s=time.time()-started)


def start(job):
    with (job/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (job/'state.json').exists():
            return status(job)
        identity = read(job/'identity.json')
        if any(file_hash(job/name) != digest for name,digest in identity['manager_sha256'].items()):
            raise ValueError('Job controller checksum mismatch')
        if file_hash(job/'input.tgz') != identity['payload_sha256']:
            raise ValueError('Payload checksum mismatch')
        safe_extract(job/'input.tgz', job/'inputs', max_bytes=1024**3)
        request = read(job/'inputs/job.json')
        if (request['schema_version'] != 1 or not re.fullmatch(r'sha256:[0-9a-f]{64}', request['image'])
                or request['env_id'] not in {'Fill-v0', 'Excavate-v0', *PACKS}
                or not time.time() < request['deadline_epoch'] <= time.time()+86400):
            raise ValueError('Invalid or expired job request')
        for name in ('source', 'harness', 'fixture'):
            if inventory(job/'inputs'/name) != request['file_sha256'][name]:
                raise ValueError(f'Changed {name} inputs')
        fixture = read(job/'inputs/fixture/fixture.json')
        if fixture['fixture']['env_id'] != request['env_id']:
            raise ValueError('Task differs from requested asset mounts')
        # Set state before spawning. A crash here is an explicit lost job,
        # never permission to submit a second simulation under the same ID.
        publish(job, 'launching')
        with (job/'manager.log').open('ab') as log:
            process = subprocess.Popen([sys.executable, str(job/'gpu_job.py'), 'execute', job.name],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        # Child waits for this launch lock before publishing its first phase.
        return publish(job, 'launching', pid=process.pid, pid_birth=birth(process.pid))


def collect(job):
    state = status(job)
    if state['phase'] not in TERMINAL:
        raise ValueError('Cannot collect a live or uncertain job')
    if state['phase'] == 'lost':
        # The archive must carry the observed terminal result, not stale running
        # state from a manager that died before publishing its final record.
        state = publish(job, 'lost', manager_live=False, containers=state.get('containers', []),
                        error='Manager disappeared and no running job container remains',
                        finished_at_epoch=time.time())
    archive = job/'results.tgz'
    if not archive.exists():
        items = [job/n for n in ('state.json', 'identity.json', 'manager.log', 'build.log', 'capture.log') if (job/n).is_file()]
        items.append(job/'inputs/job.json')
        items += [job/'native-extension'/name for name in ('record.json', 'build.json')
                  if (job/'native-extension'/name).is_file()]
        items += [job/'cooked-extension'/name for name in ('record.json','build.json')
                  if (job/'cooked-extension'/name).is_file()]
        if (job/'output').exists():
            items += [job/'output'/name for name in inventory(job/'output')]
        if sum(p.stat().st_size for p in items) > 4*1024**3:
            raise ValueError('Result archive exceeds budget')
        with tarfile.open(job/'results.tmp', 'w:gz') as output:
            for path in items:
                if path.is_symlink():
                    raise ValueError('Result link rejected')
                output.add(path, arcname=path.relative_to(job).as_posix(), recursive=False)
        (job/'results.tmp').replace(archive)
    return dict(phase=state['phase'], path=str(archive), sha256=file_hash(archive), bytes=archive.stat().st_size)


def cancel_job(job):
    with (job/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = status(job)
        if state.get('manager_live'):
            try:
                os.kill(state['pid'], signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            for prefix in ('softbody-build-', 'softbody-replay-'):
                subprocess.run(['docker', 'rm', '--force', prefix+job.name],
                               capture_output=True, timeout=20, check=False)
            # Recheck containers: a failed removal is not proof of cancellation.
            state = status(job)
            if state['phase'] == 'orphaned':
                return state
            if state['phase'] not in TERMINAL:
                publish(job, 'cancelled', error='Cancelled job without a live manager',
                        finished_at_epoch=time.time())
        return status(job)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['start', 'execute', 'status', 'collect', 'cancel'])
    parser.add_argument('job_id')
    args = parser.parse_args()
    job = job_path(args.job_id)
    if args.operation == 'execute':
        execute(job)
        return
    if args.operation == 'cancel':
        result = cancel_job(job)
    else:
        result = {'start': start, 'status': status, 'collect': collect}[args.operation](job)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

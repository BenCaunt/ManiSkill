"""Local control of immutable, resumable replay jobs on the existing GPU lease.

Only candidate source, trusted capture code and frozen reset/actions travel to
the worker. Reference outcomes and acceptance protocols remain on the laptop.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time
import uuid
import re

from .artifacts import digest_json, validate_trace
from .fixtures import load_fixture
from .job_archive import archive_inventory, atomic_json, file_hash, inventory, safe_extract
from .remote import connection
from .native_extensions import validate_actor_spec
from .cooked_extensions import validate_spec as validate_cooked_spec, source_digest as cooked_source_digest

REMOTE_ROOT = '/home/ubuntu/softbody/supervisor-jobs'
TERMINAL = {'complete', 'failed', 'cancelled', 'lost'}


class Transport:
    def __init__(self, lease):
        self.lease = Path(lease)

    def command(self, args, *, code=None, timeout=60):
        options, host = connection(self.lease)
        result = subprocess.run(['ssh', *options, host, shlex.join(args)], input=code,
                                text=True, capture_output=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f'SSH operation failed ({result.returncode}): {result.stderr[-2000:]}')
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            # A status request can race the first upload of its manager script.
            # Empty/malformed output is an observation failure, not a stopped job.
            raise RuntimeError('SSH operation returned an invalid JSON response') from exc

    def put(self, local, remote):
        options, host = connection(self.lease)
        subprocess.run(['scp', '-q', *options, '--', str(Path(local).resolve()), f'{host}:{remote}'],
                       check=True, capture_output=True, timeout=300)

    def get(self, remote, local):
        options, host = connection(self.lease)
        subprocess.run(['scp', '-q', *options, '--', f'{host}:{remote}', str(Path(local).resolve())],
                       check=True, capture_output=True, timeout=300)

    def operation(self, handle, operation):
        return self.command(['python3', f'{REMOTE_ROOT}/{handle["job_id"]}/gpu_job.py', operation, handle['job_id']], timeout=60)


def source_paths(candidate):
    """Select package/runtime/native-build inputs; omit unrelated checkout files."""
    candidate = Path(candidate).resolve()
    result = subprocess.run(['git', '-c', 'core.fsmonitor=false', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                            cwd=candidate, capture_output=True, check=True)
    names = sorted(set(result.stdout.decode().split('\0'))-{''})
    selected = []
    metadata = {'setup.py', 'setup.cfg', 'pyproject.toml', 'MANIFEST.in', 'LICENSE', 'README.md'}
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or '.git' in relative.parts:
            raise ValueError('Candidate source path escapes checkout')
        if (name in metadata or relative.parts[0] in {'mani_skill', 'warp_maniskill'}
                or relative.parts[:3] == ('tools', 'softbody', 'native')):
            selected.append(name)
    return selected


def copy_source(candidate, destination):
    candidate = Path(candidate).resolve()
    names = source_paths(candidate)
    destination.mkdir(parents=True)
    total = 0
    for name in names:
        relative = Path(name)
        source = candidate/relative
        if source.is_symlink() or any(p.is_symlink() for p in source.parents if p != candidate.parent):
            raise ValueError('Candidate snapshot cannot follow symlinks')
        if not source.exists():
            continue  # sparse-checkout omission or a real working-tree deletion
        if not source.is_file():
            raise ValueError('Candidate source must be ordinary files')
        total += source.stat().st_size
        if total > 512*1024**2:
            raise ValueError('Candidate source snapshot exceeds budget')
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o755 if source.stat().st_mode & 0o111 else 0o644)
    required = ['mani_skill/envs/softbody/capture.py', 'warp_maniskill/build_lib.py', 'setup.py']
    if not all((destination/name).is_file() for name in required):
        raise ValueError('Incomplete native port checkout')
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=candidate, text=True,
                          capture_output=True, check=True).stdout.strip()


def prepare(candidate, fixture, output, lease, image, *, timeout_s=900, harness=None, native_actor_extension=None,
            native_cooked_extension=None,cooked_pack_sha256=None,candidate_sim_backend=None):
    if candidate_sim_backend not in (None, 'physx_cpu', 'physx_cuda'):
        raise ValueError('Candidate backend must be physx_cpu or physx_cuda')
    validate_actor_spec(native_actor_extension)
    if native_cooked_extension is not None:
        validate_cooked_spec(native_cooked_extension)
        if not isinstance(cooked_pack_sha256,str) or not re.fullmatch('[a-f0-9]{64}',cooked_pack_sha256):
            raise ValueError('Cooked Pour replay requires its pinned pack manifest checksum')
    output, fixture = Path(output).resolve(), Path(fixture).resolve()
    if output.exists():
        raise ValueError('Use resume for an existing job directory')
    if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 86400:
        raise ValueError('Invalid worker time budget')
    if not isinstance(image, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
        raise ValueError('Worker image must be an immutable ID')
    lease_data = json.loads(Path(lease).read_text())
    deadline = min(time.time()+timeout_s, lease_data['terminate_at_epoch']-120)
    if deadline <= time.time():
        raise ValueError('Lease has no remaining safe job time')
    record, _, _, _ = load_fixture(fixture)
    declared_kwargs = record['fixture'].get('env_kwargs', {})
    if candidate_sim_backend is not None and 'sim_backend' in declared_kwargs and declared_kwargs['sim_backend'] != candidate_sim_backend:
        raise ValueError('Candidate backend conflicts with the frozen fixture')
    output.mkdir(parents=True)
    payload = output/'payload'
    payload.mkdir()
    source_commit = copy_source(candidate, payload/'source')
    if native_cooked_extension is not None:
        if record['fixture']['env_id']!='Pour-v0':raise ValueError('Cooked bottle adapter is only declared for Pour')
        directory=payload/'source/tools/softbody/native/cooked'
        source_config=json.loads((directory/'source.json').read_text())
        if cooked_source_digest(source_config)!=native_cooked_extension['source_config_sha256']:
            raise ValueError('Frozen cooked binary does not match candidate source configuration')
        for name,digest in source_config['files'].items():
            if file_hash(directory/name)!=digest:
                raise ValueError('Frozen cooked binary does not match candidate source: '+name)
    if native_actor_extension is not None:
        cpp = payload/'source/tools/softbody/native/actor_bridge.cpp'
        if not cpp.is_file() or file_hash(cpp) != native_actor_extension['cpp_sha256']:
            raise ValueError('Frozen actor binary does not match candidate C++ source; rebuild and pin the adapter before replay')
    (payload/'harness/softbody_lab').mkdir(parents=True)
    # Explicit Python modules only: no reference recordings, protocols, API
    # credentials, source checkouts or unrelated repository contents.
    harness_package = Path(harness)/'softbody_lab' if harness is not None else Path(__file__).parent
    for path in harness_package.glob('*.py'):
        shutil.copyfile(path, payload/'harness/softbody_lab'/path.name)
    (payload/'fixture').mkdir()
    for name in ['fixture.json', 'initial.npz', 'actions.npy', *(['material.npz'] if 'material_file_sha256' in record else [])]:
        shutil.copyfile(fixture/name, payload/'fixture'/name)
    request = dict(schema_version=1, env_id=record['fixture']['env_id'], image=image,
        deadline_epoch=deadline, timeout_s=timeout_s, source_checkout_commit=source_commit,
        source_commit_kind='worker creates separate commit of actual submitted files',
        fixture_sha256=record['fixture_sha256'],
        file_sha256={name: inventory(payload/name) for name in ('source', 'harness', 'fixture')})
    if native_actor_extension is not None:
        request['native_actor_extension'] = native_actor_extension
    if candidate_sim_backend is not None:
        request['candidate_sim_backend'] = candidate_sim_backend
    if native_cooked_extension is not None:
        request.update(native_cooked_extension=native_cooked_extension,cooked_pack_sha256=cooked_pack_sha256)
    return package_job(output, payload, request, lease, deadline)


def package_job(output, payload, request, lease, deadline):
    """Freeze one immutable request and its trusted worker controller."""
    atomic_json(payload/'job.json', request)
    with tarfile.open(output/'input.tgz', 'w:gz') as archive:
        for name in inventory(payload):
            archive.add(payload/name, arcname=name, recursive=False)
    (output/'controller').mkdir()
    for name in ('gpu_job.py', 'job_archive.py', 'native_extensions.py','cooked_extensions.py','scaling_contract.py'):
        shutil.copyfile(Path(__file__).parent/name, output/'controller'/name)
    handle = dict(schema_version=1, job_id=uuid.uuid4().hex, lease=str(Path(lease).resolve()),
        payload_sha256=file_hash(output/'input.tgz'), request_sha256=digest_json(request),
        deadline_epoch=deadline, phase='prepared', output=str(output), controller_sha256=inventory(output/'controller'))
    atomic_json(output/'remote-job.json', handle)
    return handle


def prepare_scaling(candidate, output, lease, image, case, *, timeout_s=2400, harness=None, native_actor_extension=None):
    """Run a fixed all-row scaling diagnostic under the existing lease and queue."""
    from .scaling_contract import validate_case
    validate_case(case)
    validate_actor_spec(native_actor_extension)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Use resume for an existing job directory')
    if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 86400:
        raise ValueError('Invalid worker time budget')
    if not isinstance(image, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
        raise ValueError('Worker image must be an immutable ID')
    deadline = min(time.time()+timeout_s, json.loads(Path(lease).read_text())['terminate_at_epoch']-120)
    if deadline <= time.time():
        raise ValueError('Lease has no remaining safe job time')
    output.mkdir(parents=True)
    payload = output/'payload'; payload.mkdir()
    source_commit = copy_source(candidate, payload/'source')
    if native_actor_extension is not None:
        cpp = payload/'source/tools/softbody/native/actor_bridge.cpp'
        if not cpp.is_file() or file_hash(cpp) != native_actor_extension['cpp_sha256']:
            raise ValueError('Frozen actor binary does not match candidate C++ source')
    (payload/'harness/softbody_lab').mkdir(parents=True)
    harness_package = Path(harness)/'softbody_lab' if harness is not None else Path(__file__).parent
    for path in harness_package.glob('*.py'):
        shutil.copyfile(path, payload/'harness/softbody_lab'/path.name)
    request = dict(schema_version=1, role='scaling', env_id=case['env_id'], scaling_case=case,
        image=image, candidate_sim_backend='physx_cuda', deadline_epoch=deadline, timeout_s=timeout_s,
        source_checkout_commit=source_commit, source_commit_kind='worker creates separate commit of actual submitted files',
        file_sha256={name:inventory(payload/name) for name in ('source','harness')})
    if native_actor_extension is not None:
        request['native_actor_extension'] = native_actor_extension
    return package_job(output, payload, request, lease, deadline)


def prepare_reference_demo(inputs, output, lease, image, *, dependencies, timeout_s=900, harness=None):
    """Generate a portable fixture in the existing clean, pinned reference checkout.

    Only the native reset vector and recorded controls are uploaded, never the
    demonstration's future states. Candidates cannot select the reference code.
    """
    from .replay_demo import capture_arguments
    from .runner import REFERENCE_COMMIT
    inputs, output = Path(inputs).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Use resume for an existing job directory')
    if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 86400:
        raise ValueError('Invalid worker time budget')
    if not isinstance(image, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
        raise ValueError('Worker image must be an immutable ID')
    if not isinstance(dependencies, dict) or set(dependencies) != {'warp', 'sdf'} or any(
        not isinstance(v, str) or not re.fullmatch('[a-f0-9]{64}', v) for v in dependencies.values()
    ):
        raise ValueError('Reference requires pinned Warp and SDF checksums')
    for name in ('input.json', 'initial.npy', 'actions.npy'):
        p = inputs/name
        if p.is_symlink() or not p.is_file() or not 0 < p.stat().st_size <= 64*1024**2:
            raise ValueError('Invalid native reference input')
    arguments = capture_arguments(inputs)
    if arguments['env_id'] not in ('Fill-v0', 'Excavate-v0', 'Hang-v0', 'Pour-v0'):
        raise ValueError('Reference demo job requires an available pinned SDF')
    lease_data = json.loads(Path(lease).read_text())
    deadline = min(time.time()+timeout_s, lease_data['terminate_at_epoch']-120)
    if deadline <= time.time():
        raise ValueError('Lease has no remaining safe job time')
    output.mkdir(parents=True)
    payload = output/'payload'
    (payload/'reference_input').mkdir(parents=True)
    (payload/'harness/softbody_lab').mkdir(parents=True)
    for name in ('input.json', 'initial.npy', 'actions.npy'):
        shutil.copyfile(inputs/name, payload/'reference_input'/name)
    harness_package = Path(harness)/'softbody_lab' if harness is not None else Path(__file__).parent
    for path in harness_package.glob('*.py'):
        shutil.copyfile(path, payload/'harness/softbody_lab'/path.name)
    request = dict(schema_version=1, role='reference', env_id=arguments['env_id'], image=image,
        deadline_epoch=deadline, timeout_s=timeout_s, reference_commit=REFERENCE_COMMIT,
        reference_dependencies=dependencies, native_input_sha256=file_hash(inputs/'input.json'),
        file_sha256={name: inventory(payload/name) for name in ('reference_input', 'harness')})
    return package_job(output, payload, request, lease, deadline)


def submit_or_resume(output):
    output = Path(output)
    if recover_collected(output) is not None:
        return json.loads((output/'remote-job.json').read_text())
    handle = json.loads((output/'remote-job.json').read_text())
    transport = Transport(handle['lease'])
    remote = f'{REMOTE_ROOT}/{handle["job_id"]}'
    helper = output/'controller' if (output/'controller').exists() else Path(__file__).parent
    if 'controller_sha256' in handle and inventory(helper) != handle['controller_sha256']:
        raise ValueError('Frozen job controller changed')
    identity = {k: handle[k] for k in ('payload_sha256', 'request_sha256')}
    helper_names = ['gpu_job.py', 'job_archive.py']
    if (helper/'native_extensions.py').is_file():
        helper_names.append('native_extensions.py')
    if (helper/'cooked_extensions.py').is_file():
        helper_names.append('cooked_extensions.py')
    if (helper/'scaling_contract.py').is_file():
        helper_names.append('scaling_contract.py')
    identity['manager_sha256'] = {name: file_hash(helper/name) for name in helper_names}
    bootstrap = '''import json, pathlib, sys
root=pathlib.Path(sys.argv[1]); identity=json.loads(sys.argv[2])
root.mkdir(parents=True,exist_ok=True)
p=root/'identity.json'
if p.exists():
    existing=json.loads(p.read_text())
    if any(existing[k]!=identity[k] for k in ('payload_sha256','request_sha256')): raise ValueError('Job identity changed')
    if not (root/'state.json').exists() and existing!=identity: raise ValueError('Unstarted controller identity changed')
else:
    with p.open('x') as f: json.dump(identity,f)
print(json.dumps({'started':(root/'state.json').exists()}))
'''
    staged = transport.command(['python3', '-', remote, json.dumps(identity)], code=bootstrap)
    if not staged['started']:
        if file_hash(output/'input.tgz') != handle['payload_sha256']:
            raise ValueError('Local job payload changed')
        for name in helper_names:
            transport.put(helper/name, remote+'/'+name)
        transport.put(output/'input.tgz', remote+'/input.tgz')
        observation = transport.operation(handle, 'start')
    else:
        observation = transport.operation(handle, 'status')
    handle.update(phase=observation['phase'], observation=observation)
    atomic_json(output/'remote-job.json', handle)
    return handle


def observe(output):
    output = Path(output)
    handle = json.loads((output/'remote-job.json').read_text())
    value = Transport(handle['lease']).operation(handle, 'status')
    handle.update(phase=value['phase'], observation=value, observed_at_epoch=time.time())
    atomic_json(output/'remote-job.json', handle)
    return value


def recover_collected(output):
    """Use a previously pinned result archive without contacting an expired lease.

    Receipt publication precedes download. Thus a crash between download,
    extraction and final handle update can recover the same verified evidence.
    """
    output = Path(output)
    handle = json.loads((output/'remote-job.json').read_text())
    digest = handle.get('result_sha256')
    expected_phase = None
    receipt = output/'collection-receipt.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if any(saved.get(k) != handle[k] for k in ('job_id', 'request_sha256', 'payload_sha256')):
            raise ValueError('Collection receipt belongs to different job inputs')
        if digest is not None and digest != saved['result']['sha256']:
            raise ValueError('Pinned result archive changed')
        digest = saved['result']['sha256']
        expected_phase = saved['result']['phase']
        if expected_phase not in TERMINAL:
            raise ValueError('Collection receipt is not terminal')
    if digest is None:
        return None
    if not re.fullmatch('[a-f0-9]{64}', digest):
        raise ValueError('Invalid pinned result checksum')
    archive = output/'results.tgz'
    if not archive.exists():
        if handle.get('phase') == 'collected':
            raise ValueError('Previously collected result archive is missing')
        return None
    if archive.is_symlink() or file_hash(archive) != digest:
        raise ValueError('Downloaded job result checksum differs')
    expected = archive_inventory(archive)
    destination = output/'collected'
    if destination.is_symlink():
        raise ValueError('Collected result directory cannot be a link')
    if destination.exists() and inventory(destination) != expected:
        if handle.get('phase') == 'collected':
            raise ValueError('Previously verified extracted results changed')
        # Older collectors could leave partial extraction at this path. Keep
        # that evidence and restore only from the already pinned archive.
        destination.rename(output/('collected-incomplete-'+str(time.time_ns())))
    if not destination.exists():
        safe_extract(archive, destination)
    if inventory(destination) != expected:
        raise ValueError('Extracted results differ from pinned archive')
    request = json.loads((destination/'inputs/job.json').read_text())
    if digest_json(request) != handle['request_sha256']:
        raise ValueError('Worker evaluated different job inputs')
    actual = json.loads((destination/'state.json').read_text())
    if actual.get('phase') not in TERMINAL:
        raise ValueError('Collected execution is not terminal')
    if expected_phase is not None and actual['phase'] != expected_phase:
        raise ValueError('Collected execution differs from the receipt')
    if handle.get('phase') == 'collected' and handle.get('execution') != actual:
        raise ValueError('Previously collected execution record changed')
    if actual['phase'] == 'complete' and request.get('role') == 'scaling':
        from .scaling_contract import validate_output
        validate_output(destination/'output', request, handle['payload_sha256'])
    elif actual['phase'] == 'complete':
        manifest = validate_trace(destination/'output/trace')
        if request.get('role') == 'reference':
            provenance = manifest['provenance']
            if (provenance.get('role') != 'reference' or provenance.get('source_dirty') is not False
                    or provenance.get('source_commit') != request['reference_commit']
                    or provenance.get('runtime', {}).get('image_id') != request['image']
                    or provenance.get('source_archive_sha256') != handle['payload_sha256']
                    or provenance.get('reference_initial_state_sha256') != request['file_sha256']['reference_input']['initial.npy']):
                raise ValueError('Reference capture provenance differs from the immutable request')
            fixture, _, _, _ = load_fixture(destination/'output/fixture')
            if fixture['fixture_sha256'] != manifest['fixture_sha256']:
                raise ValueError('Exported reference fixture differs from captured initial state')
    handle.update(phase='collected', execution=actual, result_sha256=digest)
    atomic_json(output/'remote-job.json', handle)
    return actual


def collect(output):
    output = Path(output)
    cached = recover_collected(output)
    if cached is not None:
        return cached
    handle = json.loads((output/'remote-job.json').read_text())
    transport = Transport(handle['lease'])
    result = transport.operation(handle, 'collect')
    if result['phase'] not in TERMINAL or not re.fullmatch('[a-f0-9]{64}', result['sha256']):
        raise ValueError('Remote collection lacks a terminal result and checksum')
    receipt = {k:handle[k] for k in ('job_id', 'request_sha256', 'payload_sha256')}
    receipt['result'] = result
    path = output/'collection-receipt.json'
    if path.exists() and json.loads(path.read_text()) != receipt:
        raise ValueError('Remote collection changed its pinned result')
    atomic_json(path, receipt)
    archive = output/'results.tgz'
    if not archive.exists():
        transport.get(result['path'], archive.with_suffix('.part'))
        if file_hash(archive.with_suffix('.part')) != result['sha256']:
            raise ValueError('Downloaded job result checksum differs')
        archive.with_suffix('.part').replace(archive)
    return recover_collected(output)


def wait(output, *, poll_s=5):
    cached = recover_collected(output)
    if cached is not None:
        return cached
    handle = json.loads((Path(output)/'remote-job.json').read_text())
    while True:
        try:
            state = observe(output)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            # Observation failure is not evidence of stopped work. Keep the
            # exact handle; never submit a replacement GPU job here.
            atomic_json(Path(output)/'observation-error.json', {'time': time.time(), 'error': str(exc)})
            if time.time() > handle['deadline_epoch']+60:
                raise RuntimeError('Unable to observe job after deadline; resume this exact handle') from exc
            time.sleep(poll_s)
            continue
        if state['phase'] in TERMINAL:
            return collect(output)
        if time.time() >= handle['deadline_epoch']:
            Transport(handle['lease']).operation(handle, 'cancel')
        time.sleep(poll_s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['submit', 'resume', 'status', 'wait', 'collect'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--lease', type=Path, default=Path('artifacts/softbody/lambda/lease.json'))
    parser.add_argument('--image')
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--candidate-sim-backend', choices=['physx_cpu', 'physx_cuda'])
    args = parser.parse_args()
    if args.operation == 'submit':
        prepare(args.candidate, args.fixture, args.output, args.lease, args.image, timeout_s=args.timeout,
                candidate_sim_backend=args.candidate_sim_backend)
        result = submit_or_resume(args.output)
    elif args.operation == 'resume':
        result = submit_or_resume(args.output)
    else:
        result = {'status': observe, 'wait': wait, 'collect': collect}[args.operation](args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

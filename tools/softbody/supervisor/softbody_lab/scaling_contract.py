"""Fixed scaling-job inputs and output integrity; no simulator imports."""
import hashlib
import json
from pathlib import Path

SNAPSHOTS = ('initial', 'warmup', 'measured', 'partial-restored', 'partial-stepped',
             'partial-fresh', 'fresh-stepped', 'flat-restored', 'flat-stepped', 'reconfigured')


def validate_case(case):
    if not isinstance(case, dict) or set(case) != {'env_id', 'num_envs', 'seeds', 'timed_controls'}:
        raise ValueError('Invalid scaling case fields')
    n = case['num_envs']
    if (case['env_id'] not in ('Fill-v0', 'Excavate-v0') or type(n) is not int or n not in (2, 8, 32)
            or type(case['timed_controls']) is not int or not 1 <= case['timed_controls'] <= 20):
        raise ValueError('Unsupported scaling task, size or control count')
    seeds = case['seeds']
    if (not isinstance(seeds, list) or len(seeds) != n
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds) or len(set(seeds)) != n):
        raise ValueError('Scaling requires one distinct integer seed per environment')
    return case


def selection(n):
    return [1] if n == 2 else [n-1, n//2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_output(root, request, payload_sha256):
    root = Path(root)
    record = json.loads((root/'result.json').read_text())
    validate_case(request['scaling_case'])
    expected = dict(role='scaling', case=request['scaling_case'], image=request['image'],
                    payload_sha256=payload_sha256, source_files_digest=digest(request['file_sha256']['source']),
                    harness_files_digest=digest(request['file_sha256']['harness']), request_digest=digest(request))
    if record.get('provenance') != expected or record.get('complete') is not True:
        raise ValueError('Scaling result provenance or completion differs from frozen request')
    files = record.get('files', {})
    if set(files) != {name+'.npz' for name in SNAPSHOTS}:
        raise ValueError('Scaling result is missing declared snapshots')
    for name, expected_hash in files.items():
        path = root/name
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 1024**3:
            raise ValueError('Invalid scaling numeric snapshot')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
        if actual is None:
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024**2), b''): h.update(block)
            actual = h.hexdigest()
        if actual != expected_hash:
            raise ValueError('Scaling snapshot checksum changed')
    return record

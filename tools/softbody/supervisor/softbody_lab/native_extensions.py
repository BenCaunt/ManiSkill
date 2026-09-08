"""Stage a pinned, already-built SAPIEN actor adapter for read-only containers.

This module never loads the native library in the host Python process. Its
configuration is supplied by the trusted runner, outside candidate permissions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

ACTOR_FILENAME = 'sapien303_actor_bridge.cpython-310-x86_64-linux-gnu.so'
SAPIEN_LIBRARY_SHA256 = '57b9dbf776bd216a2a71c86c34fadeab12469cc9067f6bf2338234499d80f097'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_actor_spec(spec):
    if spec is None:
        return
    if not isinstance(spec, dict) or set(spec) != {'build', 'sha256', 'cpp_sha256'}:
        raise ValueError('Actor extension requires build, binary SHA256 and C++ SHA256')
    if not isinstance(spec['build'], str) or not re.fullmatch(r'[A-Za-z0-9_-]+', spec['build']):
        raise ValueError('Actor extension build must be a safe directory identifier')
    if any(not isinstance(spec[k], str) or not re.fullmatch(r'[a-f0-9]{64}', spec[k]) for k in ('sha256', 'cpp_sha256')):
        raise ValueError('Actor extension checksums must be SHA256 hex strings')


def _ordinary_file(path, root, limit):
    if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root.parent):
        raise ValueError('Native extension paths cannot follow symlinks')
    if not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError('Native extension file is missing or exceeds its size limit')


def stage_actor(spec, worker_root, destination):
    validate_actor_spec(spec)
    if spec is None:
        return None
    root = Path(worker_root).absolute()
    build = root/'records'/spec['build']
    binary, manifest_path = build/ACTOR_FILENAME, build/'build.json'
    _ordinary_file(binary, root, 64*1024**2)
    _ordinary_file(manifest_path, root, 2*1024**2)
    if digest(binary) != spec['sha256']:
        raise ValueError('Native actor binary checksum differs from the frozen request')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('sapien') != '3.0.3' or not manifest.get('python', '').startswith('3.10.'):
        raise ValueError('Native actor build has a different SAPIEN or Python ABI')
    expected = {ACTOR_FILENAME:spec['sha256'], 'actor_bridge.cpp':spec['cpp_sha256'], 'libsapien.so':SAPIEN_LIBRARY_SHA256}
    for name, want in expected.items():
        values = [v for p,v in manifest.get('files', {}).items() if Path(p).name == name]
        if values != [want]:
            raise ValueError('Native actor build provenance mismatch: '+name)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for source in (binary, manifest_path):
        shutil.copyfile(source, destination/source.name)
    if digest(destination/ACTOR_FILENAME) != spec['sha256']:
        raise ValueError('Native actor changed while staging')
    record = dict(module='sapien303_actor_bridge', filename=ACTOR_FILENAME, build=spec['build'],
        sha256=spec['sha256'], cpp_sha256=spec['cpp_sha256'], sapien_library_sha256=SAPIEN_LIBRARY_SHA256,
        build_manifest_sha256=digest(destination/'build.json'), sapien='3.0.3', python_abi='cpython-310-x86_64-linux-gnu')
    (destination/'record.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('worker_root', type=Path)
    parser.add_argument('output', type=Path)
    args=parser.parse_args()
    spec=json.loads(args.request.read_text()).get('native_actor_extension')
    print(json.dumps(stage_actor(spec, args.worker_root, args.output)))


if __name__ == '__main__':
    main()

"""Independently verify a prepared wall pack without importing a simulator."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.softbody.wall_preparation.verify import evaluate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(prepared, reference, expected_pack_sha256):
    prepared, reference = prepared.resolve(), reference.resolve()
    limits = json.loads((Path(__file__).parent / 'wall_preparation/limits.json').read_text())
    protocol = json.loads((prepared / 'inputs/protocol.json').read_text())
    protocol.pop('manifest_sha256')
    if protocol != limits:
        raise ValueError('Preparation altered the fixed geometry acceptance gates')
    if sha(reference / 'export.json') != '54d4c40bdfe3184788d5e1f1841c8806611cff1249f0aad8b5c9a150ab982710':
        raise ValueError('Reference model differs')
    result = evaluate(prepared / 'inputs', prepared / 'native', prepared / 'structure', reference / 'body-0.npz')
    if sha(prepared / 'pack/pack.json') != expected_pack_sha256:
        raise ValueError('Expected a separately pinned pack manifest')
    pack = json.loads((prepared / 'pack/pack.json').read_text())
    original = json.loads((reference / 'export.json').read_text())['geometry'][0]
    if pack['physical'] != {k: original[k] for k in ['mass', 'inertia', 'com']}:
        raise ValueError('Prepared pack changed original mass, inertia or COM')
    if pack['source_geometry_sha256'] != sha(reference / 'body-0.npz') or pack['units'] != 'metres':
        raise ValueError('Prepared geometry source or units changed')
    manifest = json.loads((prepared / 'inputs/manifest.json').read_text())
    if pack['leaves'] != [p['name'] for p in manifest['pieces']]:
        raise ValueError('Packed pieces differ from verified coverage')
    root = prepared / 'pack'
    for name, digest in pack['files'].items():
        relative = PurePosixPath(name)
        path = root / name
        if (relative.is_absolute() or '..' in relative.parts or '\\' in name
                or not path.resolve().is_relative_to(root)
                or path.is_symlink() or sha(path) != digest):
            raise ValueError('Invalid packed file: ' + name)
    for leaf in pack['leaves']:
        name = 'blobs/' + leaf + '.bin'
        if name not in pack['files'] or sha(root / name) != sha(prepared / 'native/gpu-data' / (leaf + '.bin')):
            raise ValueError('Packed blob differs from independently verified cook: ' + leaf)
    for copied, source in [('UPSTREAM-README.md', 'UPSTREAM-README.md'),
                           ('UPSTREAM-WARP-LICENSE.md', 'UPSTREAM-WARP-LICENSE.md'),
                           ('ORIGINAL-PROVENANCE.json', 'PROVENANCE.json'),
                           ('original-export.json', 'export.json')]:
        if copied not in pack['files'] or sha(root / copied) != sha(reference / source):
            raise ValueError('Original metadata or notice not retained: ' + copied)
    result.update(pack_sha256=expected_pack_sha256, pack_files_verified=len(pack['files']),
                  original_physical_parameters_unchanged=True, original_notices_retained=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['prepared', 'reference', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--pack-sha256', required=True)
    args = parser.parse_args()
    result = verify(args.prepared, args.reference, args.pack_sha256)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['gpu_geometry_passed', 'piece_count', 'pack_sha256', 'pack_files_verified']}))
    raise SystemExit(0 if all(c['passed'] for c in result['cases'].values()) else 1)

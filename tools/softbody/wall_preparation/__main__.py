"""Prepare verified hollow bottle walls from an external pinned Pour export.

Run with --reference-pack, --build and a new --output directory. No network
access, reference pickle loading, asset redistribution or GPU provisioning.
"""
import argparse
import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from . import cells, cook, structure, verify

EXPORT_SHA256 = '54d4c40bdfe3184788d5e1f1841c8806611cff1249f0aad8b5c9a150ab982710'
SOURCE_COMMIT = '493be36121a9dd06071a57172274babe617b789f'
NOTICES = ['UPSTREAM-README.md', 'UPSTREAM-WARP-LICENSE.md']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def reference(pack):
    pack = pack.resolve()
    names = ['export.json', 'body-0.npz', 'PROVENANCE.json', *NOTICES]
    for name in names:
        path = pack / name
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 64 * 1024**2:
            raise ValueError('Invalid original reference input: ' + name)
    if sha(pack / 'export.json') != EXPORT_SHA256 or sha(pack / 'body-0.npz') != structure.SOURCE_SHA:
        raise ValueError('Expected the pinned Pour model and original geometry')
    exported = json.loads((pack / 'export.json').read_text())
    provenance = json.loads((pack / 'PROVENANCE.json').read_text())
    if exported['source_commit'] != SOURCE_COMMIT or provenance['source_commit'] != SOURCE_COMMIT:
        raise ValueError('Reference source revision differs')
    for name in ['export.json', 'body-0.npz', *NOTICES]:
        if sha(pack / name) != provenance['files'][name]:
            raise ValueError('Original provenance checksum mismatch: ' + name)
    for key in ['creator', 'attribution', 'source_urls', 'retrieved_at', 'license',
                'license_evidence_url', 'units', 'coordinate_frame', 'transforms', 'physical_basis']:
        if not provenance.get(key):
            raise ValueError('Missing original provenance: ' + key)
    bottle = exported['geometry'][0]
    if bottle['name'] != 'bottle' or bottle['file'] != 'body-0.npz':
        raise ValueError('Unexpected reference body')
    return exported, provenance


def package(reference_pack, output, verdict):
    # Revalidate external inputs immediately before copying them. Publishing a
    # pack is conditional on the independent geometry result, never child exit.
    exported, original = reference(reference_pack)
    if (set(verdict['cases']) != {'cpu', 'gpu-data'}
            or not all(v['passed'] for v in verdict['cases'].values())
            or verdict['piece_count'] != 384):
        raise ValueError('Independent wall verification did not pass')
    manifest = json.loads((output / 'inputs/manifest.json').read_text())
    records = [json.loads(s) for s in (output / 'native/gpu-data/progress.jsonl').read_text().splitlines()]
    events = {e['name']: e for e in records if e['event'] == 'complete'}
    leaves = [p['name'] for p in manifest['pieces']]
    pack = output / 'pack'
    staging = output / 'pack.pending'
    if pack.exists() or staging.exists():
        raise ValueError('Refusing to overwrite an existing pack')
    staging.mkdir()
    (staging / 'blobs').mkdir()
    for name in NOTICES:
        shutil.copyfile(reference_pack / name, staging / name)
    shutil.copyfile(reference_pack / 'PROVENANCE.json', staging / 'ORIGINAL-PROVENANCE.json')
    shutil.copyfile(reference_pack / 'export.json', staging / 'original-export.json')
    shutil.copyfile(output / 'inputs/manifest.json', staging / 'partition.json')
    for name in leaves:
        source = output / 'native/gpu-data' / (name + '.bin')
        if sha(source) != events[name]['blob_sha256'] or not events[name]['gpu_compatible']:
            raise ValueError('Unverified native blob: ' + name)
        shutil.copyfile(source, staging / 'blobs' / source.name)
    modifications = ('Original scaled visual mesh -> averaged source rings with unchanged triangle connectivity '
        '-> convex meridian cells in each original angular sector -> 384 explicitly cooked PhysX meshes. '
        'No CoACD stage. Hollow walls intentionally replace the original closed rigid hull; '
        'original simulation mass, inertia and COM retained. No physical-parity claim.')
    provenance = dict(original, modifications=modifications, upstream_files=original['files'],
        verification=dict(verdict_sha256=sha(output / 'verdict.json'),
            source_surface_bound_m=verdict['source_surface_correspondence_bound_m'],
            maximum_geometry_error_bound_m=verdict['cases']['gpu-data']['maximum_source_solid_union_bound_m'],
            minimum_cavity_radius_m=verdict['cases']['gpu-data']['minimum_continuous_cavity_clearance_m'],
            cavity_height_m=[.012, .17]))
    provenance.pop('files')
    write(staging / 'PROVENANCE.json', provenance)
    config = dict(leaves=leaves, physical={k: exported['geometry'][0][k] for k in ['mass', 'inertia', 'com']},
        source_geometry_sha256=structure.SOURCE_SHA, units='metres', modifications=modifications,
        files={str(p.relative_to(staging)): sha(p) for p in sorted(staging.rglob('*')) if p.is_file()})
    write(staging / 'pack.json', config)
    staging.rename(pack)
    return sha(pack / 'pack.json')


def prepare(reference_pack, build, output):
    reference_pack, build, output = reference_pack.resolve(), build.resolve(), output.resolve()
    reference(reference_pack)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    record = dict(started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        python=sys.version, dependencies={n: version(n) for n in ['numpy', 'scipy', 'shapely', 'sapien']},
        reference_files={n: sha(reference_pack / n) for n in ['export.json', 'body-0.npz', 'PROVENANCE.json', *NOTICES]},
        recipes={p.name: sha(p) for p in sorted(source.iterdir()) if p.suffix in ['.py', '.json']},
        complete=False)
    write(output / 'execution.json', record)
    try:
        report, arrays = structure.inspect(reference_pack / 'body-0.npz')
        (output / 'structure').mkdir()
        # Keep absolute input location in the run record, outside portable geometry.
        report['source'] = 'body-0.npz'
        np.savez_compressed(output / 'structure/structure.npz', **arrays)
        report['array_sha256'] = sha(output / 'structure/structure.npz')
        report['recipe_sha256'] = sha(source / 'structure.py')
        write(output / 'structure/report.json', report)
        cells.prepare(output / 'structure', output / 'inputs')
        declared = json.loads((source / 'limits.json').read_text())
        actual = json.loads((output / 'inputs/protocol.json').read_text())
        actual.pop('manifest_sha256')
        if actual != declared:
            raise ValueError('Generated protocol changed the declared geometry gates')
        execution = cook.run(output / 'inputs', output / 'native', build)
        failed = [c['case'] for c in execution['cases']
                  if c['exit_code'] != 0 or c['timed_out'] or c['errors'] or c['completed'] != 384]
        if failed:
            raise RuntimeError('Native cooking failed; see native/' + failed[0] + '/run.log')
        shutil.copyfile(build / 'build.json', output / 'build.json')
        verdict = verify.evaluate(output / 'inputs', output / 'native', output / 'structure', reference_pack / 'body-0.npz')
        write(output / 'verdict.json', verdict)
        digest = package(reference_pack, output, verdict)
        record.update(complete=True, pack_sha256=digest, scope='Geometry preparation; task and GPU dynamics acceptance remain separate')
    except Exception as exc:
        record.update(error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        record['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(output / 'execution.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['reference-pack', 'build', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.reference_pack, args.build, args.output), indent=2))

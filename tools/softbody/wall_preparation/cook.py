"""Cook bounded descriptors in isolated children, retaining failures and readbacks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(inputs, output, build, case):
    # Import from the explicit build directory, never a ambient Python module.
    sys.path.insert(0, str(build.resolve()))
    import numpy as np
    import sapien
    import sapien303_polyhedron_bridge as native
    report = json.loads((build / 'build.json').read_text())
    if Path(native.__file__).resolve() != (build / report['extension']).resolve():
        raise ValueError('Native cooking module origin differs')
    manifest = json.loads((inputs / 'manifest.json').read_text())
    if sha(inputs / 'cells.npz') != manifest['arrays_sha256']:
        raise ValueError('Cooking input changed')
    system = sapien.physx.PhysxCpuSystem()
    with np.load(inputs / 'cells.npz', allow_pickle=False) as arrays, (output / 'progress.jsonl').open('w', buffering=1) as progress:
        for piece in manifest['pieces']:
            name = piece['name']
            progress.write(json.dumps(dict(name=name, event='start')) + '\n')
            try:
                result = native.cook_polyhedron(system, arrays[name + '/vertices'],
                    arrays[name + '/poly_planes'], arrays[name + '/poly_indices'],
                    arrays[name + '/poly_offsets'], case == 'gpu-data')
                blob = output / (name + '.bin')
                blob.write_bytes(result.pop('cooked'))
                actual = {key: result.pop(key) for key in ['vertices', 'planes', 'polygon_indices', 'polygon_offsets']}
                record = output / (name + '.npz')
                np.savez_compressed(record, **actual)
                progress.write(json.dumps(dict(name=name, event='complete', **result,
                    blob_sha256=sha(blob), readback_sha256=sha(record))) + '\n')
            except Exception as exc:
                progress.write(json.dumps(dict(name=name, event='error', error=str(exc))) + '\n')
                raise


def run(inputs, output, build):
    report = json.loads((build / 'build.json').read_text())
    if (report['sapien'] != '3.0.3' or Path(report['extension']).name != report['extension']
            or sha(build / report['extension']) != report['extension_sha256']):
        raise ValueError('Cooking extension checksum or ABI differs')
    config = json.loads((Path(__file__).resolve().parents[1] / 'native/polyhedron/source.json').read_text())
    if report['source'] != config or report['native_library_sha256'] not in config['native_libraries'].values():
        raise ValueError('Cooking build is not from the declared sources')
    output.mkdir(exist_ok=False)
    execution = dict(scope='Offline explicit convex cooking; not GPU dynamics evidence',
                     input_manifest_sha256=sha(inputs / 'manifest.json'),
                     build_manifest_sha256=sha(build / 'build.json'), cases=[])
    for case in ['cpu', 'gpu-data']:
        folder = output / case
        folder.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', case,
                   '--input', str(inputs.resolve()), '--output', str(folder.resolve()),
                   '--build', str(build.resolve())]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        with (folder / 'run.log').open('w') as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=30)
                status = dict(exit_code=result.returncode, timed_out=False)
            except subprocess.TimeoutExpired:
                status = dict(exit_code=None, timed_out=True)
        progress = folder / 'progress.jsonl'
        events = [json.loads(s) for s in progress.read_text().splitlines()] if progress.exists() else []
        execution['cases'].append(dict(case=case, **status,
            completed=sum(e['event'] == 'complete' for e in events),
            errors=sum(e['event'] == 'error' for e in events), last_event=events[-1] if events else None))
        (output / 'report.json').write_text(json.dumps(execution, indent=2) + '\n')
    return execution


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', choices=['cpu', 'gpu-data'], required=True)
    for name in ['input', 'output', 'build']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    worker(a.input, a.output, a.build, a.worker)

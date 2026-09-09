"""Fetch pinned official demonstrations and export only one episode's reset/actions.

Full trajectories stay in the external reference cache. Native initial state is
for the reference engine only; export a portable fixture before candidate use.
"""
import argparse
import datetime
import hashlib
import json
import re
import urllib.request
from pathlib import Path

import numpy as np

REPOSITORY = 'haosulab/ManiSkill2'
REVISION = '0c367447d26e4e2de13fbf5e5d2ab09a258187da'
TASKS = ('Fill-v0', 'Hang-v0', 'Pour-v0', 'Excavate-v0', 'Pinch-v0', 'Write-v0')
MAX_DOWNLOAD = 1024**3
MAX_ARRAY = 64 * 1024**2


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path, entry):
    if path.stat().st_size != entry['size']:
        raise ValueError(f'{path.name}: wrong byte count')
    digest = sha256(path)
    if 'lfs' in entry:
        actual, expected = digest, entry['lfs']['oid']
    else:
        actual = hashlib.sha1(b'blob '+str(entry['size']).encode()+b'\0'+path.read_bytes()).hexdigest()
        expected = entry['oid']
    if actual != expected:
        raise ValueError(f'{path.name}: official object checksum mismatch')
    return digest


def download(env_id, directory):
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f'demos/v0/soft_body/{env_id}/'
    url = f'https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}/{prefix}'
    with urllib.request.urlopen(url, timeout=30) as response:
        entries = json.loads(response.read(1024**2+1))
    records = []
    for filename in ('trajectory.json', 'trajectory.h5'):
        entry, = [e for e in entries if e['path'] == prefix+filename]
        if entry['type'] != 'file' or not 0 < entry['size'] <= MAX_DOWNLOAD:
            raise ValueError('Unexpected dataset entry type/size')
        expected = entry.get('lfs', {}).get('oid', entry['oid'])
        if not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', expected):
            raise ValueError('Invalid official object checksum')
        source = f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{entry["path"]}?download=true'
        path = directory / filename
        if not path.exists():
            partial = path.with_suffix(path.suffix+'.partial')
            count = 0
            with urllib.request.urlopen(source, timeout=60) as response, partial.open('xb') as output:
                while chunk := response.read(1024**2):
                    count += len(chunk)
                    if count > entry['size']:
                        raise ValueError('Download exceeds declared size')
                    output.write(chunk)
            verify(partial, entry)
            partial.rename(path)
        digest = verify(path, entry)
        records.append(dict(file=filename, sha256=digest, bytes=entry['size'], source_url=source,
                            official_object=entry))
        print(f'Verified {env_id}/{filename}: {entry["size"]} bytes', flush=True)
    provenance = dict(repository=REPOSITORY, commit=REVISION,
        retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), files=records,
        scope='External reference data under original source terms; candidate receives portable reset inputs/actions only')
    (directory/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    return provenance


def numeric_dataset(group, name, ndim):
    import h5py
    dataset = group[name]
    if not isinstance(dataset, h5py.Dataset) or dataset.ndim != ndim or dataset.size*dataset.dtype.itemsize > MAX_ARRAY:
        raise ValueError(f'{name}: invalid numeric array shape/size')
    if dataset.dtype.kind != 'f' or dataset.dtype.itemsize not in (4, 8) or not dataset.size:
        raise ValueError(f'{name}: expected nonempty floating-point array')
    data = dataset[()]
    if not np.isfinite(data).all():
        raise ValueError(f'{name}: nonfinite values')
    return data


def export_episode(env_id, episode_id, source, output, provenance):
    import h5py
    if output.exists():
        raise FileExistsError(output)
    metadata = json.loads((source/'trajectory.json').read_text())
    episode, = [e for e in metadata['episodes'] if e['episode_id'] == episode_id]
    if metadata['env_info']['env_id'] != env_id:
        raise ValueError('Wrong environment in demonstration metadata')
    with h5py.File(source/'trajectory.h5', 'r') as dataset:
        trajectory = dataset[f'traj_{episode_id}']
        actions = numeric_dataset(trajectory, 'actions', 2)
        initial = numeric_dataset(trajectory, 'env_init_state', 1)
    if len(actions) != episode['elapsed_steps']:
        raise ValueError('Action count differs from episode metadata')
    output.mkdir(parents=True)
    np.save(output/'actions.npy', actions, allow_pickle=False)
    np.save(output/'initial.npy', initial, allow_pickle=False)
    result = dict(env_id=env_id, seed=episode['episode_seed'], steps=len(actions),
        control_mode=episode['control_mode'], reset_kwargs=episode['reset_kwargs'],
        env_kwargs=metadata['env_info']['env_kwargs'],
        registered_horizon=metadata['env_info'].get('max_episode_steps'),
        source_episode_id=episode_id, source_commit_info=metadata.get('commit_info'),
        source_dataset_provenance=provenance,
        initialization='Original numeric native env_init_state; reference-only. Candidate requires portable fixture.',
        behavior='Recorded controls; state assignment only once during reset.',
        files={name: sha256(output/name) for name in ('actions.npy', 'initial.npy')})
    (output/'input.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-id', choices=TASKS, required=True)
    parser.add_argument('--episode-id', type=int, default=0)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.episode_id < 0 or args.output.exists():
        parser.error('Episode must be nonnegative and output must not exist')
    provenance = download(args.env_id, args.source_dir)
    result = export_episode(args.env_id, args.episode_id, args.source_dir, args.output, provenance)
    print(json.dumps({key: result[key] for key in ('env_id', 'seed', 'steps', 'control_mode', 'reset_kwargs')}))


if __name__ == '__main__':
    main()

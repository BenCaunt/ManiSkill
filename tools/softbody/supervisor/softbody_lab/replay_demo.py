"""Replay checksummed native demonstration inputs in the reference engine only."""
import argparse
import json
from pathlib import Path

from .demo_inputs import TASKS, sha256
from .runner import capture


def capture_arguments(directory):
    metadata = json.loads((directory/'input.json').read_text())
    if metadata['env_id'] not in TASKS:
        raise ValueError('Unsupported demonstration task')
    for filename in ('initial.npy', 'actions.npy'):
        if sha256(directory/filename) != metadata['files'][filename]:
            raise ValueError(f'Demonstration checksum mismatch: {filename}')
    if metadata['steps'] > metadata['registered_horizon']:
        raise ValueError('Demonstration exceeds registered horizon; requires a separate explicit protocol')
    env_kwargs = metadata['env_kwargs'].copy()
    if env_kwargs.pop('obs_mode') != 'none' or env_kwargs.pop('control_mode') != metadata['control_mode']:
        raise ValueError('Inconsistent demonstration environment configuration')
    reset = metadata['reset_kwargs'].copy()
    if reset.pop('seed') != metadata['seed']:
        raise ValueError('Inconsistent demonstration seed')
    # The recording revision stored task reset options as flat keywords.
    # Pinned v0.5.3 moved these into reset(options=...).
    if set(reset) - {'target_num', 'level_file'}:
        raise ValueError('Unknown task reset option in demonstration metadata')
    return dict(env_id=metadata['env_id'], seed=metadata['seed'], steps=metadata['steps'],
        role='reference', control_mode=metadata['control_mode'], env_kwargs=env_kwargs,
        reset_kwargs={'options': reset} if reset else {}, actions_path=directory/'actions.npy',
        reference_initial_state_path=directory/'initial.npy')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture-output', type=Path,
                        help='Export the verified portable reset/actions after reference capture')
    args = parser.parse_args()
    result = capture(source=args.source, output=args.output, **capture_arguments(args.input))
    record = {'recording': str(result)}
    if args.fixture_output is not None:
        from .fixtures import export_fixture
        record['fixture'] = str(export_fixture(result, args.fixture_output))
    print(json.dumps(record))


if __name__ == '__main__':
    main()

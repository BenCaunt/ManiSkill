"""Compare frozen origin/repeat diagnostics without changing the batching verdict."""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from .gpu_batch_checks import evaluate, particle_errors
from .job_archive import file_hash


def read_arrays(root, name, label):
    with np.load(Path(root) / name / (label + '.npz'), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def compare_rollouts(root, name, other_root, other_name, other_index, limits):
    errors = {}
    for label in ('initial', 'warmup-1', 'warmup-2', 'expected'):
        errors[label] = particle_errors(read_arrays(root, name, label), 0,
                                       read_arrays(other_root, other_name, label), other_index)
    initial = read_arrays(root, name, 'camera-initial-env0')
    other = read_arrays(other_root, other_name, f'camera-initial-env{other_index}')
    matrix_error = np.abs(initial['rigid_visual_poses'].astype(float) - other['rigid_visual_poses'].astype(float))
    model = read_arrays(root, name, 'model-initial')
    other_model = read_arrays(other_root, other_name, 'model-initial')
    return dict(errors=errors,
                native_model_inputs_equal=set(model) == set(other_model) and all(
                    np.array_equal(value[0], other_model[key][other_index]) for key, value in model.items()),
                initial_rigid_translation_max_abs_m=float(matrix_error[:, :3, 3].max()),
                initial_rigid_rotation_max_abs=float(matrix_error[:, :3, :3].max()),
                within_original_final_limits={key: errors['expected'][key] <= limit for key, limit in limits.items()})


def evaluate_origins(root, protocol, baseline_root, baseline_protocol_path, baseline_archive):
    baseline_protocol_path = Path(baseline_protocol_path)
    if file_hash(baseline_protocol_path) != protocol['baseline']['protocol_sha256']:
        raise ValueError('Frozen baseline protocol identity differs')
    baseline_protocol = json.loads(baseline_protocol_path.read_text())
    if file_hash(baseline_archive) != protocol['baseline']['archive_sha256']:
        raise ValueError('Frozen baseline archive identity differs')
    current = evaluate(root, protocol)
    baseline = evaluate(baseline_root, baseline_protocol)
    failures = list(current['failures'])
    # This diagnostic can use the retained trajectory failure as its baseline;
    # it must not quietly use failed identity, lifecycle or camera checks.
    allowed = r'env[01]/(?:expected|flat-stepped|partial-stepped): x difference [0-9.e+-]+ exceeds 1e-05'
    failures.extend('Baseline integrity: ' + item for item in baseline['failures'] if not re.fullmatch(allowed, item))
    request = json.loads((Path(root) / 'request.json').read_text())
    baseline_request = json.loads((Path(baseline_root) / 'request.json').read_text())
    if request['source_files'] != baseline_request['source_files']:
        failures.append('Diagnostic runtime differs from frozen baseline runtime')
    for case in protocol['cases']:
        result = json.loads((Path(root) / case['name'] / 'result.json').read_text())
        if result.get('scene_offsets') != case.get('scene_offsets', [[0., 0., 0.]]):
            failures.append(case['name'] + ': actual native origin differs from declared origin')
    comparisons = {}
    if not failures:
        for pair in protocol['origin_pairs']:
            comparisons[pair['case']] = compare_rollouts(root, pair['case'], baseline_root,
                                                         pair['baseline_case'], pair['baseline_index'], protocol['replay_limits'])
        a, b = protocol['repeat_pair']
        comparisons['identical-input-repeats'] = compare_rollouts(root, a, root, b, 0, protocol['replay_limits'])
    return dict(integrity_passed=not failures, failures=failures, comparisons=comparisons,
                retained_baseline_passed=baseline['passed'], retained_baseline_failures=baseline['failures'],
                execution_and_rendering=current, scope=protocol['scope'], full_port_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path); parser.add_argument('protocol', type=Path)
    parser.add_argument('baseline_root', type=Path); parser.add_argument('baseline_protocol', type=Path)
    parser.add_argument('baseline_archive', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = evaluate_origins(args.root, json.loads(args.protocol.read_text()), args.baseline_root, args.baseline_protocol, args.baseline_archive)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['integrity_passed'] else 1)

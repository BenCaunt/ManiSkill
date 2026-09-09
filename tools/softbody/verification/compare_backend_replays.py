"""Audit full Excavate backend replays without importing either simulator.

Run with tools/softbody/supervisor on PYTHONPATH. This reports trajectory
differences and original task predicates; it does not fit acceptance limits.
"""
import argparse
import json
from pathlib import Path

from softbody_lab.artifacts import (
    digest_arrays, initial_numeric_state, load_frame, sha256, validate_trace,
)
from softbody_lab.compare import derived_initial_errors, frame_errors
from softbody_lab.task_checks import outcome


def describe(reference, candidate, expected_backend):
    manifests = [validate_trace(p) for p in (reference, candidate)]
    baseline, trial = manifests
    if baseline['fixture']['env_id'] != 'Excavate-v0':
        raise ValueError('This diagnostic audits Excavate predicates only')
    if baseline['fixture_sha256'] != trial['fixture_sha256']:
        raise ValueError('Different frozen physical fixtures')
    if baseline['actions'] != trial['actions'] or len(baseline['samples']) != len(trial['samples']):
        raise ValueError('Different controls or horizons')
    execution = trial['provenance']['candidate_execution']
    if expected_backend not in ('physx_cpu', 'physx_cuda') or any(
        execution[k] != expected_backend for k in ('requested_sim_backend', 'actual_sim_backend')
    ) or execution['gpu_sim_enabled'] != (expected_backend == 'physx_cuda'):
        raise ValueError('Different candidate execution backend')
    if execution['mpm_device'] != 'cuda':
        raise ValueError('Both diagnostic variants require CUDA MPM')
    rows, maxima, first_nonzero = [], {}, {}
    label_mismatches = {'reference': [], 'candidate': []}
    success_steps = {'reference': [], 'candidate': []}
    for a, b in zip(baseline['samples'], trial['samples']):
        if a['step'] != b['step'] or a['time_s'] != b['time_s']:
            raise ValueError('Different sample times')
        left, right = [load_frame(p/s['path']) for p, s in ((reference, a), (candidate, b))]
        if a['step'] == 0:
            numeric = [digest_arrays(initial_numeric_state(s, baseline['fixture'])) for s in (left, right)]
            if numeric[0] != numeric[1]:
                raise ValueError('Different independent numeric initial state')
            initial_errors = derived_initial_errors(left, right, baseline['fixture'])
        errors = frame_errors(left, right)
        for key, value in errors.items():
            maxima[key] = max(maxima.get(key, 0.), value)
            if value and key not in first_nonzero:
                first_nonzero[key] = a['step']
        tasks = {}
        for role, s, sample in (('reference', left, a), ('candidate', right, b)):
            tasks[role] = outcome(s, 'Excavate-v0')
            if tasks[role]['success'] != sample['metrics']['success']:
                label_mismatches[role].append(a['step'])
            if tasks[role]['success']:
                success_steps[role].append(a['step'])
        rows.append(dict(step=a['step'], time_s=a['time_s'], errors=errors, **tasks))
    return dict(
        schema_version=1,
        scope='Full-episode diagnostic; original task thresholds; no calibrated physics parity verdict',
        manifests_sha256={k: sha256(p/'manifest.json') for k, p in (('reference', reference), ('candidate', candidate))},
        fixture_sha256=baseline['fixture_sha256'], initial_numeric_sha256=numeric[0],
        backend=execution, samples=len(rows), simulation_seconds=rows[-1]['time_s'],
        provenance={'reference': baseline['provenance'], 'candidate': trial['provenance']},
        integrity_checks=dict(complete_finite_traces=True, identical_fixture_and_actions=True,
                              identical_independent_initial_state=True, actual_backend_matches=True),
        label_mismatches=label_mismatches,
        first_success_step={k: min(v) if v else None for k, v in success_steps.items()},
        successful_samples={k: len(v) for k, v in success_steps.items()},
        initial_derived_errors=initial_errors, max_errors=maxima, first_nonzero_error_step=first_nonzero,
        final=rows[-1], frames=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--backend', choices=['physx_cpu', 'physx_cuda'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = describe(args.reference, args.candidate, args.backend)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('backend', 'samples', 'label_mismatches', 'first_success_step', 'final')}))


if __name__ == '__main__':
    main()

"""Reaudit a complete fixed scaling matrix from immutable collected job archives.

The numerical gate is the frozen capture-time evaluator. This aggregation code
was added after the first result; it does not fit tolerances or change verdicts.
Run with the trusted frozen harness on PYTHONPATH, outside the candidate checkout.
"""
import argparse
import json
from pathlib import Path

from softbody_lab.job_archive import atomic_json, file_hash, inventory
from softbody_lab.remote_replay import recover_collected
from softbody_lab.scaling_checks import evaluate


def check_request(request, protocol, case):
    expected = dict(role='scaling', scaling_case=case, env_id=case['env_id'],
        candidate_sim_backend='physx_cuda', image=protocol['image'],
        timeout_s=protocol['job_timeout_s'], native_actor_extension=protocol['native_actor_extension'],
        source_checkout_commit=protocol['candidate_commit'])
    if any(request.get(k) != v for k, v in expected.items()):
        raise ValueError('Request differs from declared scaling case or runtime')
    if request['file_sha256'] != dict(source=protocol['runtime_source_files'],
            harness={'softbody_lab/'+k: v for k, v in protocol['harness_files'].items()}):
        raise ValueError('Submitted source or trusted harness differs from frozen inventory')


def summarize_verdict(verdict):
    selected, untouched = [], []
    reset_failures = []
    for label, rows in verdict['exact_resets'].items():
        for index, change in rows.items():
            row = dict(snapshot=label, env_index=int(index), **change)
            (selected if change['selected'] else untouched).append(row)
            reset_failures.append(label+f'/env{index}: changed '+','.join(change['fields']))
    other = [failure for failure in verdict['failures'] if failure not in reset_failures]
    return dict(strict_passed=verdict['passed'], failures=len(verdict['failures']),
        selected_changed_rows=len(selected), unselected_changed_rows=len(untouched),
        other_failures=other, selected_changes=selected, unselected_changes=untouched,
        max_selected_abs_error=max([e for c in selected for e in c['max_abs_errors'].values() if e is not None] or [0]),
        max_unselected_abs_error=max([e for c in untouched for e in c['max_abs_errors'].values() if e is not None] or [0]))


def audit(study, expected_protocol_sha256):
    study = Path(study)
    if file_hash(study/'protocol.json') != expected_protocol_sha256:
        raise ValueError('Frozen protocol changed')
    protocol = json.loads((study/'protocol.json').read_text())
    if inventory(Path(protocol['harness'])/'softbody_lab') != protocol['harness_files']:
        raise ValueError('Trusted frozen harness changed')
    import softbody_lab.scaling_checks as numeric
    import softbody_lab.remote_replay as jobs
    if (file_hash(Path(numeric.__file__)) != protocol['harness_files']['scaling_checks.py']
            or file_hash(Path(jobs.__file__)) != protocol['harness_files']['remote_replay.py']):
        raise ValueError('Run aggregation using the frozen trusted harness')
    rows = []
    for case in protocol['cases']:
        label = case['env_id'].split('-')[0]+'-N'+str(case['num_envs'])
        job = study/label
        execution = recover_collected(job)
        if execution is None or execution['phase'] != 'complete':
            raise ValueError('Every declared job must be collected and complete: '+label)
        handle = json.loads((job/'remote-job.json').read_text())
        request = json.loads((job/'collected/inputs/job.json').read_text())
        check_request(request, protocol, case)
        verdict = evaluate(job/'collected/output', request, handle['payload_sha256'])
        saved = json.loads((job/'verdict.json').read_text())
        if verdict != saved:
            raise ValueError('Previously audited numeric verdict changed: '+label)
        row = dict(label=label, case=case, job_id=handle['job_id'], execution=execution['phase'],
            request_sha256=handle['request_sha256'], input_sha256=handle['payload_sha256'],
            result_sha256=handle['result_sha256'], verdict_sha256=file_hash(job/'verdict.json'),
            result_record_sha256=file_hash(job/'collected/output/result.json'),
            warp_binary_sha256=execution['binary_sha256'],
            native_adapter=execution['native_actor_extension'], physx_gpu_library=execution['physx_gpu_library'],
            **summarize_verdict(verdict))
        for key in ('initial_counts', 'initial_mass_kg', 'median_control_seconds',
                'aggregate_env_controls_per_second', 'timed_seconds', 'setup_seconds',
                'observed_device_used_max_bytes', 'process_high_water_rss_kib'):
            row[key] = verdict[key]
        rows.append(row)
        print(json.dumps(dict(case=label, strict_passed=row['strict_passed'],
            selected_changed_rows=row['selected_changed_rows'],
            unselected_changed_rows=row['unselected_changed_rows'])), flush=True)
    result = dict(schema_version=1, protocol_sha256=expected_protocol_sha256,
        all_declared_jobs_collected=True, rows=rows,
        scope=protocol['scope'], calibrated_physics_parity=False, full_port_complete=False,
        aggregation_scope='Post-capture aggregation; unchanged frozen exact numerical gates',
        timing_scope='Median of the declared synchronized env.step controls after two warmups; observations included, rendering and snapshot I/O excluded',
        memory_scope='Maximum observed whole-device usage at ten boundaries, not continuous peak; RSS is process high-water')
    atomic_json(study/'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    audit(args.study, args.protocol_sha256)

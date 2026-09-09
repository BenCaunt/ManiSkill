"""Independent checks for paired particle-render update measurements."""
import argparse
import json
from pathlib import Path

import numpy as np

from .job_archive import file_hash, inventory


def evaluate(root, protocol):
    root = Path(root)
    failures, measurements = [], {}
    execution = json.loads((root / 'execution.json').read_text())
    request = json.loads((root / 'request.json').read_text())
    for key in ('image_id', 'input_sha256'):
        if execution[key] != protocol[key]:
            raise ValueError('Changed run identity: ' + key)
    for name, digest in [('request.json', protocol['request_sha256']),
                         ('probe.py', protocol['probe_sha256']), *protocol['helpers'].items()]:
        if file_hash(root / name) != digest:
            raise ValueError('Changed captured source: ' + name)
    if request['cases'] != protocol['cases'] or request['render_performance_helpers'] != protocol['helpers']:
        raise ValueError('Changed case/helper specification')
    baseline = protocol['baseline']['files']['mani_skill/envs/softbody/particle_visuals.py']
    if file_hash(root / 'baseline_particle_visuals.py') != baseline:
        raise ValueError('Baseline update is not the preceding runtime source')
    prefix = 'mani_skill/envs/softbody/'
    expected = {p[len(prefix):]: h for p, h in request['source_files'].items()
                if p.startswith(prefix) and p.endswith('.py') and '/' not in p[len(prefix):]}
    expected.update({p: request['source_files']['mani_skill/envs/' + p] for p in ('scene.py', 'sapien_env.py')})
    if inventory(root / 'source') != expected:
        raise ValueError('Captured runtime source differs from request')
    if execution['exit_code'] != 0:
        failures.append('Worker suite did not complete successfully')
    for case in protocol['cases']:
        name = case['name']
        directory = root / name
        main = json.loads((directory / 'result.json').read_text())
        status = json.loads((directory / 'execution.json').read_text())
        if main['case'] != case or main['probe_sha256'] != protocol['probe_sha256']:
            raise ValueError('Changed task probe identity: ' + name)
        if not main['complete'] or status['exit_code'] != 0:
            failures.append(name + ': actual task/lifecycle probe did not complete')
        path = directory / 'render-performance.json'
        if not path.exists():
            failures.append(name + ': no performance records')
            continue
        report = json.loads(path.read_text())
        if (report['helper_sha256'] != protocol['helpers']['render_update_probe.py']
                or report['baseline_sha256'] != baseline):
            raise ValueError('Changed measured update helper: ' + name)
        if len(report['records']) < len(case['seeds']):
            failures.append(name + ': missing initialized model measurements')
        rows = []
        for index, record in enumerate(report['records']):
            label = name + '/' + str(index)
            if record['index'] != index or record['file'] != f'render-performance-{index:03d}.npz':
                raise ValueError('Invalid measurement identity')
            if file_hash(directory / record['file']) != record['sha256']:
                raise ValueError('Changed numeric measurement')
            with np.load(directory / record['file'], allow_pickle=False) as z:
                data = {k: z[k] for k in z.files}
            if any(v.dtype.kind not in 'biuf' or not np.isfinite(v).all() for v in data.values()):
                raise ValueError('Invalid measurement arrays')
            before = {k[7:]: v for k, v in data.items() if k.startswith('before/')}
            after = {k[6:]: v for k, v in data.items() if k.startswith('after/')}
            if not before or before.keys() != after.keys() or 'x' not in before:
                raise ValueError('Incomplete physical state measurement')
            if any(not np.array_equal(v, after[k]) for k, v in before.items()):
                failures.append(label + ': visual updates changed physical particle state')
            positions = data['expected_positions']
            if positions.shape != (record['count'], 3) or record['pool_size'] < record['count']:
                raise ValueError('Invalid active particle count')
            if not np.array_equal(positions, before['x']):
                failures.append(label + ': expected poses are not the physical particle positions')
            for key in ('actual_render_positions', 'fallback_entity_positions'):
                if not np.array_equal(data[key], positions):
                    failures.append(label + ': ' + key + ' differ from actual particles')
            timings = {}
            if record['repetitions'] != protocol['timing_repetitions']:
                raise ValueError('Changed timing repetitions')
            for method in ('baseline', 'candidate'):
                durations = record[method + '_ns']
                if len(durations) != protocol['timing_trials'] or any(type(v) is not int or v <= 0 for v in durations):
                    raise ValueError('Invalid completed timing trials')
                timings[method + '_median_ms'] = float(np.median(durations)) / record['repetitions'] / 1e6
            speedup = timings['baseline_median_ms'] / timings['candidate_median_ms']
            direct = record['particle_device'] == 'cuda' and not record['update_entities']
            if direct:
                for key in ('borrowed_pointer_matches', 'candidate_without_readback',
                            'baseline_readback_detected', 'candidate_without_entity_writes',
                            'baseline_entity_write_detected'):
                    if record.get(key) is not True:
                        failures.append(label + ': missing mechanism check ' + key)
                if not np.array_equal(data['ordered_stream_positions'], positions):
                    failures.append(label + ': incorrect nondefault stream ordering')
                if np.array_equal(data['unordered_stream_positions'], positions):
                    failures.append(label + ': unsynchronized negative control did not expose the ordering bug')
                if speedup < protocol['minimum_median_speedup']:
                    failures.append(label + ': measured visual-update speedup below declared minimum')
            rows.append(dict(index=index, particles=record['count'], pool_size=record['pool_size'],
                             direct_cuda=direct, speedup=speedup, **timings))
        measurements[name] = rows
    return dict(passed=not failures, failures=failures, measurements=measurements,
                scope=protocol['scope'], full_port_complete=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'protocol', 'output'):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    result = evaluate(a.root, json.loads(a.protocol.read_text()))
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)

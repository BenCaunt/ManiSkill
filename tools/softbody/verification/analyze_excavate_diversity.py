"""Independently audit every completed member of the predeclared episode study."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

from softbody_lab.artifacts import digest_json
from softbody_lab.job_archive import atomic_json, file_hash

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('study', nargs='?', type=Path, default=Path(__file__).resolve().parent,
                    help='Private study directory containing frozen inputs and collected jobs')
ROOT = parser.parse_args().study.resolve()
INPUTS = json.loads((ROOT/'verification-inputs.json').read_text())
for name, expected in INPUTS['verifier_files'].items():
    if file_hash(ROOT/'verification'/name) != expected:
        raise ValueError('Independent verifier changed')
sys.path.insert(0, str(ROOT/'verification'))
from compare_backend_replays import describe


def main():
    for filename, key in (('protocol.json', 'protocol_sha256'), ('candidate-inputs.json', 'candidate_inputs_sha256')):
        if file_hash(ROOT/filename) != INPUTS[key]:
            raise ValueError('Predeclared study inputs changed')
    config = json.loads((ROOT/'candidate-inputs.json').read_text())
    plan = json.loads((ROOT/'protocol.json').read_text())
    rows, pending = [], []
    for episode in plan['episode_ids']:
        parent = ROOT/f'episode-{episode}'
        reference = parent/'reference/collected/output/trace'
        if not (reference/'manifest.json').exists():
            pending.append(f'episode-{episode}/reference')
            continue
        ref = json.loads((reference/'manifest.json').read_text())
        inputs = json.loads((parent/'native-input/input.json').read_text())
        if inputs['source_episode_id'] != episode or inputs['seed'] != ref['fixture']['seed']:
            raise ValueError('Episode identity changed')
        with np.load(reference/ref['samples'][0]['path'], allow_pickle=False) as state:
            particle_count = len(state['x']); mass = float(state['mass'].sum())
        for backend in config['backends']:
            output = parent/backend
            handle_path = output/'remote-job.json'
            if not handle_path.exists():
                pending.append(f'episode-{episode}/{backend}'); continue
            handle = json.loads(handle_path.read_text())
            if handle['phase'] != 'collected':
                pending.append(f'episode-{episode}/{backend}'); continue
            row = dict(episode_id=episode, seed=ref['fixture']['seed'], backend=backend,
                       particles=particle_count, mass_kg=mass, steps=ref['requested_steps'],
                       execution=handle['execution']['phase'], job_id=handle['job_id'],
                       input_sha256=handle['payload_sha256'], result_sha256=handle['result_sha256'])
            if row['execution'] != 'complete':
                row['error'] = handle['execution'].get('error'); rows.append(row); continue
            request = json.loads((output/'collected/inputs/job.json').read_text())
            trace = output/'collected/output/trace'
            manifest = json.loads((trace/'manifest.json').read_text())
            if (request['file_sha256']['source'] != config['runtime_source_files']
                    or request['image'] != config['image'] or request['candidate_sim_backend'] != backend
                    or manifest['provenance']['source_archive_sha256'] != handle['payload_sha256']
                    or manifest['provenance']['runtime']['image_id'] != config['image']
                    or manifest['provenance']['capture_version'] != digest_json(config['harness_files'])):
                raise ValueError('Candidate provenance differs from frozen study inputs')
            diagnostic = parent/(backend+'-diagnostic.json')
            d = describe(reference, trace, backend)
            if diagnostic.exists() and json.loads(diagnostic.read_text()) != d:
                raise ValueError('Previously audited evidence changed')
            atomic_json(diagnostic, d)
            row.update(final_reference=d['final']['reference'], final_candidate=d['final']['candidate'],
                first_success_step=d['first_success_step'], label_mismatches=d['label_mismatches'],
                max_errors=d['max_errors'], initial_derived_errors=d['initial_derived_errors'],
                diagnostic_sha256=file_hash(diagnostic))
            rows.append(row)
    result = dict(schema_version=1, scope=plan['scope'], protocol_sha256=INPUTS['protocol_sha256'],
        candidate_inputs_sha256=INPUTS['candidate_inputs_sha256'], calibrated_physics_parity=False,
        all_declared_jobs_collected=not pending and len(rows)==6, pending=pending, rows=rows)
    atomic_json(ROOT/'summary.json', result)
    print(json.dumps(dict(completed=len(rows), pending=pending, outcomes=[
        {k:r[k] for k in ('episode_id','backend','execution')} | {'final':r.get('final_candidate')} for r in rows])), flush=True)


if __name__ == '__main__':
    main()

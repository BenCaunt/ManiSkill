"""Paired visual-update timing and CUDA ordering checks inside real task runs.

This observer keeps the existing task probe's controls, resets and camera checks.
The baseline update method is copied verbatim from the preceding fork commit.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import weakref
from unittest.mock import patch

import numpy as np
import torch


def install(args):
    import sys
    sys.path.insert(0, str(args.source))
    from mani_skill.envs.softbody.particle_visuals import ParticleVisualPool
    from mani_skill.envs.softbody.mpm import wp

    root = Path(__file__).parent
    request = json.loads(args.request.read_text())
    for name, digest in request['render_performance_helpers'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Changed render performance helper: ' + name)
    spec = importlib.util.spec_from_file_location('baseline_particle_visuals', root / 'baseline_particle_visuals.py')
    baseline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline_module)
    old_update = baseline_module.ParticleVisualPool.update
    new_update = ParticleVisualPool.update
    seen = weakref.WeakKeyDictionary()
    records = []

    def legacy(pool, coupler, poses, update_entities):
        # Exact preceding pool method plus the preceding caller's CUDA copy.
        positions = old_update(pool, coupler)
        if poses is not None:
            poses[:len(positions), :3] = torch.as_tensor(positions, device=poses.device)
        return positions

    def observed(pool, coupler, render_poses=None, *, update_entities=True):
        result = new_update(pool, coupler, render_poses, update_entities=update_entities)
        if render_poses is None or coupler.model in seen:
            return result
        seen[coupler.model] = True
        particles = coupler.states[0].struct.particle_q
        count = len(pool.active)
        record = dict(index=len(records), count=count, pool_size=len(pool.entities),
                      particle_device=particles.device, update_entities=update_entities,
                      baseline_ns=[], candidate_ns=[], repetitions=request['timing_repetitions'])
        before = coupler.particle_state()
        data = {'before/' + k: v.copy() for k, v in before.items()}
        data['expected_positions'] = particles.numpy()[:count].copy()
        for _ in range(request['timing_warmup']):
            legacy(pool, coupler, render_poses, update_entities)
            new_update(pool, coupler, render_poses, update_entities=update_entities)
        for trial in range(request['timing_trials']):
            order = ('baseline', 'candidate') if trial % 2 == 0 else ('candidate', 'baseline')
            for name in order:
                torch.cuda.synchronize(render_poses.device)
                started = time.perf_counter_ns()
                for _ in range(request['timing_repetitions']):
                    if name == 'baseline':
                        legacy(pool, coupler, render_poses, update_entities)
                    else:
                        new_update(pool, coupler, render_poses, update_entities=update_entities)
                torch.cuda.synchronize(render_poses.device)
                record[name + '_ns'].append(time.perf_counter_ns() - started)

        direct = particles.device == 'cuda' and not update_entities
        if direct:
            original_numpy = type(particles).numpy
            def denied_numpy(array):
                if array is particles:
                    raise RuntimeError('Particle CPU readback forbidden by diagnostic')
                return original_numpy(array)
            with patch.object(type(particles), 'numpy', denied_numpy):
                view = new_update(pool, coupler, render_poses, update_entities=False)
                record['borrowed_pointer_matches'] = view.data_ptr() == particles.ptr
                record['candidate_without_readback'] = True
                try:
                    legacy(pool, coupler, render_poses, False)
                except RuntimeError as exc:
                    record['baseline_readback_detected'] = 'CPU readback forbidden' in str(exc)

            class NoPoseWrites:
                def __setattr__(self, name, value):
                    raise RuntimeError('Native visual pose writes forbidden by diagnostic')
            active = pool.active
            try:
                pool.active = [NoPoseWrites() for _ in active]
                new_update(pool, coupler, render_poses, update_entities=False)
                record['candidate_without_entity_writes'] = True
                try:
                    legacy(pool, coupler, render_poses, False)
                except RuntimeError as exc:
                    record['baseline_entity_write_detected'] = 'pose writes forbidden' in str(exc)
            finally:
                pool.active = active

            # Deliberately delay a destination writer on a nondefault Torch
            # stream. The copied Warp positions must follow that writer, then
            # become visible to a consumer on that same Torch stream.
            consumer = torch.cuda.Stream(device=render_poses.device)
            solver = torch.cuda.ExternalStream(wp.context.runtime.cuda_stream, device=render_poses.device)
            consumer.wait_stream(torch.cuda.current_stream(render_poses.device))
            with torch.cuda.stream(consumer):
                torch.cuda._sleep(10_000_000)
                render_poses[:count, :3].fill_(-9.)
                new_update(pool, coupler, render_poses, update_entities=False)
                ordered = render_poses[:count, :3].clone()
            consumer.synchronize()
            data['ordered_stream_positions'] = ordered.cpu().numpy()

            # Negative control: omit both stream dependencies. Synchronize only
            # the solver copy before the delayed Torch writer finishes.
            with torch.cuda.stream(consumer):
                torch.cuda._sleep(10_000_000)
                render_poses[:count, :3].fill_(-9.)
                with torch.cuda.stream(solver):
                    render_poses[:count, :3].copy_(wp.to_torch(particles)[:count])
                solver.synchronize()
                unordered = render_poses[:count, :3].clone()
            consumer.synchronize()
            data['unordered_stream_positions'] = unordered.cpu().numpy()

        # Exercise the native-entity fallback using the same real model and
        # visual objects, including every independent batch pool.
        new_update(pool, coupler, render_poses, update_entities=True)
        data['fallback_entity_positions'] = np.array([e.pose.p for e in pool.active])
        new_update(pool, coupler, render_poses, update_entities=update_entities)
        torch.cuda.synchronize(render_poses.device)
        data['actual_render_positions'] = render_poses[:count, :3].cpu().numpy().copy()
        data.update({'after/' + k: v.copy() for k, v in coupler.particle_state().items()})
        filename = 'render-performance-' + str(len(records)).zfill(3) + '.npz'
        np.savez_compressed(args.output / filename, **data)
        record.update(file=filename, sha256=hashlib.sha256((args.output / filename).read_bytes()).hexdigest())
        records.append(record)
        (args.output / 'render-performance.json').write_text(json.dumps(dict(
            helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            baseline_sha256=hashlib.sha256((root / 'baseline_particle_visuals.py').read_bytes()).hexdigest(),
            records=records, scope='Paired visual-update timing only; real task stepping/cameras are checked separately.'), indent=2) + '\n')
        return result

    ParticleVisualPool.update = observed

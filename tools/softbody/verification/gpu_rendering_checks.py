"""Host verification that particle camera surfaces follow actual physics state."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .job_archive import file_hash, inventory


def measure_frame(data, radius, protocol):
    failures = []
    before = {k[7:]: v for k, v in data.items() if k.startswith('before/')}
    after = {k[6:]: v for k, v in data.items() if k.startswith('after/')}
    if not before or set(before) != set(after):
        raise ValueError('Missing before/after physics state')
    if any(not np.array_equal(v, after[k]) for k, v in before.items()):
        failures.append('Rendering changed physical/checkpoint state')
    if protocol.get('require_native_buffer_checks') and not np.array_equal(data['before_native_rigid'], data['after_native_rigid']):
        failures.append('Rendering changed the native rigid CUDA buffer')
    if protocol.get('require_observation_pixels'):
        for key in ('rgb', 'depth', 'segmentation'):
            if not np.array_equal(data['observation_' + key], data[key]):
                failures.append('Returned observation differs from the captured camera: ' + key)
    points, ids = data['particle_x'], data['particle_ids']
    if (points.ndim != 2 or points.shape[1] != 3 or ids.shape != (len(points),)
            or not np.array_equal(points, after['mpm/x'][0])):
        raise ValueError('Particle visual identity/physical state mismatch')
    rgb, position, segmentation, depth = (data[k] for k in ('rgb', 'position', 'segmentation', 'depth'))
    if (rgb.ndim != 4 or rgb.shape[0] != 1 or rgb.shape[-1] != 3
            or position.shape != rgb.shape or segmentation.shape != (*rgb.shape[:3], 1)
            or depth.shape != segmentation.shape or max(rgb.shape[1:3]) > 2048):
        raise ValueError('Malformed camera tensors')
    model, extrinsic = data['cam2world_gl'], data['extrinsic_cv']
    if model.shape != (1, 4, 4) or extrinsic.shape != (1, 3, 4):
        raise ValueError('Malformed camera transforms')
    if not np.allclose(extrinsic[0] @ model[0], np.diag([1., -1., -1., 1.])[:3], atol=1e-5, rtol=0):
        failures.append('Camera coordinate transforms disagree')
    if len(np.unique(ids)) != len(ids) or np.any(ids <= 0):
        raise ValueError('Particle segmentation identifiers are ambiguous')
    # The minimal render target saturates scene IDs at signed-int16 maximum;
    # it does not wrap them into negative labels. Saturated pixels cannot be
    # assigned to individual particles reliably, even if an ID32767 exists.
    maximum_id = np.iinfo(segmentation.dtype).max
    if np.any(ids > maximum_id):
        failures.append('Particle scene IDs exceed the segmentation texture range')
        return dict(particle_pixels=None, maximum_particle_id=int(ids.max()),
                    maximum_texture_id=int(maximum_id), failures=failures)
    encoded = ids.astype(segmentation.dtype)
    rigid_report = {}
    if protocol.get('require_rigid_visual_bounds'):
        rigid_ids, bounds, transforms = (data[k] for k in ('rigid_visual_ids', 'rigid_visual_bounds', 'rigid_visual_poses'))
        if (rigid_ids.ndim != 1 or len(rigid_ids) < 7 or bounds.shape != (len(rigid_ids), 2, 3)
                or transforms.shape != (len(rigid_ids), 4, 4) or len(set(rigid_ids.tolist())) != len(rigid_ids)):
            raise ValueError('Invalid robot visual bounds/identity record')
        maximum, count = 0., 0
        camera_xyz = position[0].astype(float) / 1000.
        world_xyz = camera_xyz @ model[0, :3, :3].T + model[0, :3, 3]
        for identifier, bound, transform in zip(rigid_ids, bounds, transforms):
            mask = segmentation[0, ..., 0] == identifier
            count += int(mask.sum())
            if mask.any():
                local = (world_xyz[mask] - transform[:3, 3]) @ transform[:3, :3]
                maximum = max(maximum, float(np.maximum(bound[0] - local, local - bound[1]).max()))
        if count < protocol['minimum_rigid_pixels']:
            failures.append('Too few robot pixels for the visual bound check')
        if maximum > protocol['rigid_bound_error_limit_m']:
            failures.append(f'Robot pixels fall outside physical-pose visual bounds: {maximum} m')
        rigid_report = dict(rigid_pixels=count, max_rigid_bound_violation_m=maximum)
    if protocol.get('require_visual_pool_checks'):
        pool = data['pool_ids']
        if pool.ndim != 1 or len(np.unique(pool)) != len(pool) or not np.array_equal(pool[:len(ids)], ids):
            raise ValueError('Invalid active/pool visual identity mapping')
        inactive_pixels = int(np.isin(segmentation, pool[len(ids):]).sum())
        if inactive_pixels:
            failures.append(f'Inactive pooled particles remain visible: {inactive_pixels} pixels')
    order = np.argsort(encoded); sorted_ids = encoded[order]
    pixels = segmentation[0, ..., 0].ravel()
    rows = np.searchsorted(sorted_ids, pixels)
    valid = rows < len(sorted_ids)
    valid[valid] &= sorted_ids[rows[valid]] == pixels[valid]
    count = int(valid.sum())
    if count < protocol['minimum_particle_pixels']:
        failures.append('Too few actual particle pixels')
    if not count:
        return dict(particle_pixels=0, failures=failures)
    if np.any(depth[0, ..., 0].ravel()[valid] <= 0):
        failures.append('Invalid particle pixel depth')
    camera_points = position[0].reshape(-1, 3)[valid].astype(float) / 1000.
    world_points = camera_points @ model[0, :3, :3].T + model[0, :3, 3]
    centers = points[order[rows[valid]]]
    errors = np.abs(np.linalg.norm(world_points - centers, axis=1) - radius)
    maximum = float(errors.max())
    if maximum > protocol['surface_distance_error_limit_m']:
        failures.append(f'Particle pixels do not match physical spheres: surface error {maximum} m')
    return dict(**rigid_report, particle_pixels=count, visible_particles=int(len(np.unique(rows[valid]))),
        max_surface_error_m=maximum, median_surface_error_m=float(np.median(errors)), failures=failures)


def evaluate(root, protocol):
    root = Path(root); failures, reports = [], {}
    execution = json.loads((root / 'execution.json').read_text())
    if execution['exit_code'] != 0:
        failures.append('Worker failed; retained case errors follow')
    for key in ('input_sha256', 'image_id'):
        if execution[key] != protocol[key]:
            failures.append('Frozen identity mismatch: ' + key)
    if file_hash(root / 'request.json') != protocol['request_sha256'] or file_hash(root / 'probe.py') != protocol['probe_sha256']:
        failures.append('Frozen request/probe mismatch')
    request = json.loads((root / 'request.json').read_text()); prefix = 'mani_skill/envs/softbody/'
    expected_sources = {k[len(prefix):]: v for k, v in request['source_files'].items()
        if k.startswith(prefix) and k.endswith('.py') and '/' not in k[len(prefix):]}
    if protocol.get('require_scene_source'):
        expected_sources['scene.py'] = request['source_files']['mani_skill/envs/scene.py']
    if inventory(root / 'source') != expected_sources:
        failures.append('Actual task source mismatch')
    for case in protocol['cases']:
        name = case['name']; directory = root / name
        if not (directory / 'result.json').is_file() or not (directory / 'execution.json').is_file():
            failures.append(name + ': missing record'); continue
        result = json.loads((directory / 'result.json').read_text())
        status = json.loads((directory / 'execution.json').read_text())['exit_code']
        if result['case'] != case or result['probe_sha256'] != protocol['probe_sha256']:
            failures.append(name + ': case/probe identity mismatch')
        complete = result['complete'] is True and status == 0
        if not complete:
            failures.append(name + ': ' + result.get('error', 'incomplete execution'))
            if not result['frames']:
                continue
        if result['physics_system'] != ('PhysxGpuSystem' if case['backend'] == 'gpu' else 'PhysxCpuSystem'):
            failures.append(name + ': wrong physics backend')
        radius = .005 if case['task'] == 'Write' else .0025
        if np.float32(result['particle_radius']) != np.float32(radius):
            failures.append(name + ': physical particle radius changed')
        expected_frames = protocol['frames'] if complete else protocol['frames'][:len(result['frames'])]
        if [frame['label'] for frame in result['frames']] != expected_frames:
            raise ValueError('Missing rendering/reset frame')
        frames = {}
        for frame in result['frames']:
            label = frame['label']; path = directory / (label + '.npz')
            if frame['file'] != path.name or file_hash(path) != frame['sha256']:
                failures.append(name + ': frame archive identity mismatch')
            if frame['position_units'] != 'mm':
                failures.append(name + ': wrong camera position units')
            with np.load(path, allow_pickle=False) as archive:
                data = {k: archive[k] for k in archive.files}
            if any(v.dtype.kind not in 'fibu' or not np.isfinite(v).all() for v in data.values()):
                raise ValueError('Invalid numeric rendering data')
            measured = measure_frame(data, radius, protocol)
            if protocol.get('particle_counts') and len(data['particle_x']) != protocol['particle_counts'][label]:
                measured['failures'].append('Wrong live particle count for reset case')
            failures.extend(name + '/' + label + ': ' + f for f in measured['failures'])
            with Image.open(directory / (label + '.png')) as picture:
                if not np.array_equal(np.asarray(picture), data['rgb'][0]):
                    failures.append(name + '/' + label + ': PNG does not match actual RGB tensor')
            frames[label] = measured
        reports[name] = frames
    return dict(passed=not failures, failures=failures, reports=reports, scope=protocol['scope'], full_port_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path); parser.add_argument('protocol', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = evaluate(args.root, json.loads(args.protocol.read_text()))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result['passed'] else 1)

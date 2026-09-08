"""Independent pixel/state checks for explicitly altered Pour inspection views.

The ordinary camera verdict remains separate. This diagnostic tests occlusion
without changing physical state, particle radius, materials or task thresholds.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .job_archive import file_hash
from .gpu_rendering_checks import measure_frame


def compare_views(default, hidden, restored, limits):
    failures = []
    for label, frame in [('hidden', hidden), ('restored', restored)]:
        for key in default:
            if key.startswith(('before/', 'after/')) or key in (
                'before_native_rigid', 'after_native_rigid', 'particle_x',
                'particle_ids', 'pool_ids', 'cam2world_gl', 'extrinsic_cv', 'intrinsic_cv',
                'rigid_visual_ids', 'rigid_visual_bounds', 'rigid_visual_poses'):
                if key not in frame or not np.array_equal(default[key], frame[key]):
                    failures.append(label + ': physical state or camera changed: ' + key)
    for key in ('rgb', 'depth', 'position', 'segmentation', 'rigid_visual_visibility'):
        if not np.array_equal(default[key], restored[key]):
            failures.append('Restoring visibility changed the default view: ' + key)
    ids = default['rigid_visual_ids']
    visibility = hidden['rigid_visual_visibility']
    original = default['rigid_visual_visibility']
    if visibility.shape != ids.shape or original.shape != ids.shape:
        raise ValueError('Malformed visibility record')
    altered = original != visibility
    hidden_ids = ids[altered]
    if not altered.any() or np.any(visibility[altered] != 0):
        failures.append('Inspection must only hide recorded rigid visuals')
    if np.isin(hidden['segmentation'], hidden_ids).any():
        failures.append('Hidden rigid visuals remain in the camera image')
    config = dict(limits, require_rigid_visual_bounds=False)
    measured = measure_frame(hidden, .0025, config)
    failures.extend(measured['failures'])
    particle_pixels = np.isin(hidden['segmentation'][0, ..., 0], hidden['particle_ids'])
    original_labels = default['segmentation'][0, ..., 0][particle_pixels]
    front = default['depth'][0, ..., 0][particle_pixels].astype(float)
    behind = hidden['depth'][0, ..., 0][particle_pixels].astype(float)
    # Depth textures are integer millimetres. The declared 3 mm bound is the
    # existing surface-position gate, not a fitted physical tolerance.
    occluded = np.isin(original_labels, hidden_ids) & (front > 0) & (
        front <= behind + 1000 * limits['surface_distance_error_limit_m'])
    already_visible = np.isin(original_labels, default['particle_ids'])
    explained = occluded | already_visible
    ratio = float(explained.mean()) if len(explained) else 0.
    if ratio < limits['minimum_explained_pixel_fraction']:
        failures.append('New particle pixels are not explained by the hidden occluders')
    return dict(**{k:v for k,v in measured.items() if k != 'failures'},
        hidden_ids=hidden_ids.tolist(), occluded_particle_pixels=int(occluded.sum()),
        already_visible_particle_pixels=int(already_visible.sum()),
        explained_pixel_fraction=ratio, failures=failures)


def evaluate(root, protocol):
    root = Path(root)
    request = json.loads((root/'request.json').read_text())
    execution = json.loads((root/'execution.json').read_text())
    if (file_hash(root/'request.json') != protocol['request_sha256']
            or file_hash(root/'probe.py') != protocol['probe_sha256']
            or execution['input_sha256'] != protocol['input_sha256']
            or execution['image_id'] != protocol['image_id']
            or request.get('pour_visibility_diagnostic') != 1
            or request['cases'] != protocol['cases']):
        raise ValueError('Visibility diagnostic identity mismatch')
    failures, reports = [], {}
    if execution['exit_code'] != 0:
        failures.append('Worker failed')
    variants = ['inspection-bottle-hidden', 'inspection-bottle-robot-hidden', 'inspection-visibility-restored']
    for case in protocol['cases']:
        directory = root/case['name']
        report = json.loads((directory/'result.json').read_text())
        if report['case'] != case or not report['complete']:
            raise ValueError('Missing completed visibility case')
        expected = [dict(label=label,index=i,file=f'{label}-env{i}.npz')
                    for label in variants for i in range(len(case['seeds']))]
        if report['diagnostic_renders'] != expected:
            raise ValueError('Missing visibility inspection frame')
        def frame(label, index):
            path = directory/f'{label}-env{index}.npz'
            if file_hash(path) != report['files'][path.name]:
                raise ValueError('Visibility frame checksum mismatch')
            with np.load(path,allow_pickle=False) as z:
                arrays = {key:z[key] for key in z.files}
            if any(a.dtype.kind not in 'fibu' or not np.isfinite(a).all() for a in arrays.values()):
                raise ValueError('Invalid visibility numeric record')
            with Image.open(path.with_suffix('.png')) as image:
                if not np.array_equal(np.asarray(image),arrays['rgb'][0]):
                    raise ValueError('Visibility image differs from the recorded camera')
            return arrays
        for index in range(len(case['seeds'])):
            default = frame('camera-initial', index)
            restored = frame(variants[-1], index)
            for variant in variants[:-1]:
                name = f'{case["name"]}/env{index}/{variant}'
                measured = compare_views(default,frame(variant,index),restored,protocol['visibility'])
                reports[name] = measured
                failures.extend(name+': '+f for f in measured['failures'])
    return dict(passed=not failures,failures=failures,reports=reports,
        scope='Diagnostic visibility intervention only; ordinary camera and physical parity gates remain separate.')

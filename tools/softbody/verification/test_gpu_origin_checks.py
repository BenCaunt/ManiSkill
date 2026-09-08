"""Diagnostic arithmetic faults; these are not evidence of GPU repeatability."""
import numpy as np

from tools.softbody.verification import gpu_origin_checks as checks


def test_origin_comparison_keeps_particle_gate_and_reports_native_rounding(monkeypatch):
    def read(root, name, label):
        changed = name == 'b'
        if label.startswith('camera'):
            pose = np.eye(4)[None]
            pose[0, 0, 3] = 2e-7 if changed else 0.
            return {'rigid_visual_poses': pose}
        if label == 'model-initial':
            return {'mass': np.array([[1.]])}
        values = {f'actual/0/{key}': np.zeros((2,)) for key in ('x', 'v', 'F', 'C', 'vc')}
        values.update({key: np.zeros((1, 7)) for key in ('qpos', 'qvel', 'drive_position', 'drive_velocity')})
        if changed and label == 'expected':
            values['actual/0/x'][0] = 2e-5
        return values
    monkeypatch.setattr(checks, 'read_arrays', read)
    result = checks.compare_rollouts('.', 'a', '.', 'b', 0, {'x': 1e-5, 'qpos': 1e-5})
    assert result['errors']['initial']['x'] == 0.
    assert result['errors']['expected']['x'] == 2e-5
    assert result['within_original_final_limits'] == {'x': False, 'qpos': True}
    assert result['native_model_inputs_equal']
    assert result['initial_rigid_translation_max_abs_m'] == 2e-7
    assert result['initial_rigid_rotation_max_abs'] == 0.

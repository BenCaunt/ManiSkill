from types import SimpleNamespace

import numpy as np
import pytest

from softbody_lab.artifacts import InvalidArtifact, digest_arrays, initial_numeric_state
from softbody_lab.telemetry import WRENCH_FIELD, snapshot_with_telemetry, validate_telemetry


def buffer(value):
    return SimpleNamespace(ext_body_f=SimpleNamespace(numpy=lambda: np.full((3, 6), value, dtype=np.float32)))


def test_reference_reads_applied_buffers_after_rotation_without_mutation():
    # Forward integration wrote reaction values 1,3,5,7 to the old input
    # buffers. The new final state has unrelated value99 and rotates to0.
    old = [buffer(n) for n in (1, 3, 5, 7, 99)]
    env = SimpleNamespace(mpm_states=[old[-1], *old[:-1]])
    adapter = SimpleNamespace(env_id='Pinch-v0', env=env)
    state = {'unrelated': np.array([17.])}
    result = snapshot_with_telemetry(state, adapter, 'reference', [WRENCH_FIELD])
    assert np.array_equal(result[WRENCH_FIELD], np.full((3, 6), 4, dtype=np.float32))
    assert WRENCH_FIELD not in state
    assert env.mpm_states[0] is old[-1]
    assert env.mpm_states[0].ext_body_f.numpy()[0, 0] == 99


def test_candidate_reads_actual_completed_step_and_fresh_reset_buffers():
    actual = np.arange(18, dtype=np.float32).reshape(3, 6)
    env = SimpleNamespace(last_coupling_step=SimpleNamespace(mean_wrench_torque_force=actual),
                          mpm_coupler=SimpleNamespace(states=[buffer(99), buffer(2), buffer(4)]))
    adapter = SimpleNamespace(env_id='Pinch-v0', env=env)
    result = snapshot_with_telemetry({}, adapter, 'candidate', [WRENCH_FIELD])
    assert np.array_equal(result[WRENCH_FIELD], actual)
    result[WRENCH_FIELD][:] = -10
    assert actual[0, 0] == 0  # recording cannot mutate the live force buffer
    env.last_coupling_step = None
    assert np.all(snapshot_with_telemetry({}, adapter, 'candidate', [WRENCH_FIELD])[WRENCH_FIELD] == 3)


@pytest.mark.parametrize('value', [None, np.zeros((2, 6), dtype=np.float32),
    np.zeros((3, 6), dtype=np.float64), np.full((3, 6), np.nan, dtype=np.float32)])
def test_missing_or_invalid_force_telemetry_rejected(value):
    with pytest.raises(InvalidArtifact, match='wrench telemetry'):
        validate_telemetry({WRENCH_FIELD: value}, {'env_id': 'Pinch-v0', 'telemetry_fields': [WRENCH_FIELD]})


def test_telemetry_is_opt_in_and_unknown_fields_rejected():
    state = {'x': np.array([[1., 2., 3.]]), 'sim_state': np.array([9.])}
    adapter = SimpleNamespace(env_id='Fill-v0')
    assert snapshot_with_telemetry(state, adapter, 'reference', []) is state
    assert digest_arrays(initial_numeric_state(state, {})) == digest_arrays({'x': state['x']})
    for fields, env_id in [([WRENCH_FIELD], 'Fill-v0'), (['forged'], 'Pinch-v0'),
                           (None, 'Pinch-v0'), ([WRENCH_FIELD]*2, 'Pinch-v0')]:
        with pytest.raises(InvalidArtifact, match='Unsupported'):
            validate_telemetry({}, {'env_id': env_id, 'telemetry_fields': fields})
    with pytest.raises(InvalidArtifact, match='Undeclared'):
        validate_telemetry({WRENCH_FIELD: np.zeros((3, 6), dtype=np.float32)}, {})

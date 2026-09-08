"""Read actual simulator diagnostics without changing physics or reset inputs.

Wrenches are world-frame [torque (N m), force (N)] about each body's COM.
Each sample is the MPM mean applied at the LAST rigid step of a control step,
not an impulse or an average over the entire control interval. Short contacts
between recorded instants can be missed. Before stepping, read fresh buffers.
"""
import numpy as np

from .artifacts import InvalidArtifact

WRENCH_FIELD = 'last_mpm_body_wrench'


def validate_fields(fields, env_id):
    if fields == []:
        return
    if fields != [WRENCH_FIELD] or env_id != 'Pinch-v0':
        raise InvalidArtifact('Unsupported MPM telemetry fields or environment')


def validate_telemetry(state, fixture):
    fields = fixture.get('telemetry_fields', [])
    validate_fields(fields, fixture.get('env_id'))
    if fields:
        value = state.get(WRENCH_FIELD)
        if (not isinstance(value, np.ndarray) or value.shape != (3, 6)
                or value.dtype != np.dtype('float32') or not np.isfinite(value).all()):
            raise InvalidArtifact('Missing or invalid last MPM body wrench telemetry')
    elif WRENCH_FIELD in state:
        raise InvalidArtifact('Undeclared MPM wrench telemetry')


def snapshot_with_telemetry(state, adapter, role, fields):
    validate_fields(fields, adapter.env_id)
    if not fields:
        return state
    env = adapter.env
    if role == 'reference':
        # Original base_env.step_action rotates [old_last, *old[:-1]] AFTER
        # averaging old[:-1] and applying the wrench. Read those same buffers.
        buffers = env.mpm_states[1:]
        value = np.mean([s.ext_body_f.numpy() for s in buffers], axis=0)
    elif role == 'candidate':
        if env.last_coupling_step is not None:
            value = env.last_coupling_step.mean_wrench_torque_force
        else:
            value = np.mean([s.ext_body_f.numpy() for s in env.mpm_coupler.states[1:]], axis=0)
    else:
        raise InvalidArtifact('Unknown telemetry capture role')
    result = {**state, WRENCH_FIELD: np.array(value, copy=True)}
    validate_telemetry(result, {'env_id': adapter.env_id, 'telemetry_fields': fields})
    return result

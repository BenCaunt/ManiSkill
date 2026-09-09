"""Synthetic fault injection tests for the evaluator, not physics acceptance evidence."""

from pathlib import Path

import numpy as np
import pytest

from softbody_lab.artifacts import (
    InvalidArtifact, TraceWriter, digest_json, digest_arrays, initial_numeric_state,
    read_json, sha256, validate_trace, write_json,
)
from softbody_lab.compare import METRICS, INITIAL_METRICS, calibrate, compare, protocol_metrics


def test_legacy_reset_preserves_selected_level_in_fixture_metadata():
    from types import SimpleNamespace
    from softbody_lab.runner import LegacyAdapter
    selected = []
    def reset(*, seed, options):
        selected.append((seed, options.pop('level_file')))
        options['nested']['values'].clear()
    adapter = object.__new__(LegacyAdapter)
    adapter.env = SimpleNamespace(reset=reset)
    adapter.snapshot = lambda: {'captured': True}
    kwargs = {'options': {'level_file': 'elbow.h5', 'nested': {'values': [1]}}}
    for _ in range(2):
        assert adapter.reset(seed=17, reset_kwargs=kwargs) == {'captured': True}
    assert selected == [(17, 'elbow.h5'), (17, 'elbow.h5')]
    assert kwargs == {'options': {'level_file': 'elbow.h5', 'nested': {'values': [1]}}}


def test_native_demo_initialization_is_reference_only(tmp_path):
    from softbody_lab.runner import capture, load_native_demo_initialization
    path = tmp_path/'initial.npy'
    np.save(path, np.arange(10, dtype=np.float32))
    assert np.array_equal(load_native_demo_initialization(path), np.arange(10))
    for role, replay in [('candidate', None), ('reference', tmp_path/'fixture')]:
        with pytest.raises(ValueError, match='reference-only'):
            capture(source=tmp_path, output=tmp_path/'out', role=role, env_id='Hang-v0', seed=1, steps=1,
                    replay=replay, reference_initial_state_path=path)
    for bad in (np.zeros((1, 3)), np.array([np.nan]), np.array(['text'])):
        np.save(path, bad)
        with pytest.raises(InvalidArtifact, match='numeric demonstration'):
            load_native_demo_initialization(path)


def test_reference_reset_compensates_measured_pose_setter_roundoff():
    from types import SimpleNamespace
    from softbody_lab.runner import restore_legacy_actor_pose
    def pose(p, q):
        return SimpleNamespace(p=np.asarray(p, dtype=np.float32),q=np.asarray(q,dtype=np.float32))
    class BiasedActor:
        def set_pose(self, value):
            self.pose=pose(value.p+np.array([0,0,-2**-26],dtype=np.float32),value.q)
    target=np.array([.02,.03,-2**-26,1,0,0,0],dtype=np.float32)
    actor=BiasedActor()
    restore_legacy_actor_pose(actor,target,pose)
    assert np.array_equal(np.r_[actor.pose.p,actor.pose.q],target)
    class StuckActor:
        def set_pose(self, value):
            self.pose=pose([0,0,0],value.q)
    with pytest.raises(InvalidArtifact,match='translation did not restore'):
        restore_legacy_actor_pose(StuckActor(),target,pose)


def state(offset=0.0):
    return {"x": np.array([[offset, 0., 0.5], [offset + .1, 0., .5]]),
            "v": np.zeros((2, 3)), "F": np.tile(np.eye(3), (2, 1, 1)),
            "C": np.zeros((2, 3, 3)), "vc": np.zeros(2), "mass": np.array([.1, .2]),
            "qpos": np.zeros(2), "qvel": np.zeros(2), "sim_state": np.zeros(10),
            "drive_position": np.zeros(2), "drive_velocity": np.zeros(2),
            "rigid_pose": np.array([[0., 0., 0., 1., 0., 0., 0.]]),
            "rigid_velocity": np.zeros((1, 6))}


@pytest.mark.parametrize('backend', ['physx_cpu', 'physx_cuda'])
@pytest.mark.parametrize('fault', [None, 'wrong_actual_backend', 'conflicting_fixture'])
def test_backend_replay_keeps_fixture_and_actions_and_records_actual_backend(monkeypatch, tmp_path, backend, fault):
    from types import SimpleNamespace
    from softbody_lab import runner
    snapshot = state()
    material = {'synthetic_material': np.array([1., 2.])}
    description = dict(control_mode='pd_joint_pos', control_dt=.05,
                       material_sha256=digest_arrays(material))
    fixture = dict(env_id='Excavate-v0', seed=7, env_kwargs={'reward_mode': 'dense'},
                   reset_kwargs={'options': {}}, **description)
    other_backend = 'physx_cpu' if backend == 'physx_cuda' else 'physx_cuda'
    if fault == 'conflicting_fixture':
        fixture['env_kwargs']['sim_backend'] = other_backend
    fixture['initial_numeric_sha256'] = digest_arrays(initial_numeric_state(snapshot, fixture))
    baseline = dict(fixture=fixture, fixture_sha256=digest_json(fixture), steps=1)
    actions = np.array([[.2, -.3]], dtype=np.float32)
    calls = []
    closed = []
    class Adapter:
        def __init__(self, env_id, *, control_mode, env_kwargs):
            calls.append(env_kwargs)
            self.env_id = env_id
            actual = other_backend if fault == 'wrong_actual_backend' else backend
            self.env = SimpleNamespace(backend=SimpleNamespace(sim_backend=actual, sim_device='synthetic'),
                                       gpu_sim_enabled=backend == 'physx_cuda', mpm_device='cuda')
        def reset(self, **kwargs):
            assert kwargs['replay'][1] is fixture
            return snapshot
        def snapshot(self): return snapshot
        def description(self): return description.copy(), material
        def metrics(self): return {'success': False}
        def step(self, action):
            np.testing.assert_array_equal(action, actions[0])
            return {'success': False}
        def close(self): closed.append(True)
    module = SimpleNamespace(__file__=tmp_path/'mani_skill/envs/softbody/capture.py', CaptureAdapter=Adapter)
    monkeypatch.setattr(runner.importlib, 'import_module', lambda _: module)
    monkeypatch.setattr(runner, 'doctor', lambda: {'cuda_reference_ready': True})
    monkeypatch.setattr(runner, 'provenance', lambda *a: {'role': 'candidate', 'kind': 'synthetic test'})
    monkeypatch.setattr(runner, 'load_fixture', lambda _: (baseline, snapshot, actions, material))
    output = tmp_path/'trace'
    if fault is not None:
        error, message = (RuntimeError, 'did not use') if fault == 'wrong_actual_backend' else (ValueError, 'conflicts')
        with pytest.raises(error, match=message):
            runner.capture(source=tmp_path, output=output, role='candidate', env_id='Fill-v0', seed=99, steps=5,
                           replay=tmp_path/'unused', candidate_sim_backend=backend)
        assert not output.exists()
        assert closed == ([True] if fault == 'wrong_actual_backend' else [])
        return
    runner.capture(source=tmp_path, output=output, role='candidate', env_id='Fill-v0', seed=99, steps=5,
                   replay=tmp_path/'unused', candidate_sim_backend=backend)
    recorded = read_json(output/'manifest.json')
    assert recorded['status'] == 'complete'
    assert recorded['fixture'] == fixture
    assert recorded['fixture_sha256'] == baseline['fixture_sha256']
    assert recorded['actions'] == actions.tolist()
    assert calls == [{'reward_mode': 'dense', 'sim_backend': backend}]
    assert fixture['env_kwargs'] == {'reward_mode': 'dense'}
    assert recorded['provenance']['candidate_execution']['actual_sim_backend'] == backend
    assert closed == [True]


@pytest.mark.parametrize('role,backend', [('reference', 'physx_cuda'), ('candidate', 'auto'), ('candidate', 'cuda')])
def test_invalid_candidate_backend_rejected_before_hardware_access(monkeypatch, tmp_path, role, backend):
    from softbody_lab import runner
    monkeypatch.setattr(runner, 'doctor', lambda: pytest.fail('Must reject before GPU access'))
    with pytest.raises(ValueError, match='candidate-only'):
        runner.capture(source=tmp_path, output=tmp_path/'out', role=role, env_id='Excavate-v0',
                       seed=1, steps=1, candidate_sim_backend=backend)


def trace(root: Path, offset=0.0, role="reference", steps=2):
    writer = TraceWriter(root, fixture={"env_id": "synthetic-evaluator-test", "seed": 1},
                         provenance={"role": role, "run_id": root.name,
                                     "source_commit": "a" * 40, "source_dirty": False,
                                     "runtime": {"kind": "synthetic"}, "capture_version": 1},
                         steps=steps)
    writer.frame(time_s=0., state=state(), metrics={"success": False})
    for i in range(steps):
        writer.frame(time_s=(i + 1) * .02, state=state(offset),
                     metrics={"success": False}, action=[0., 0.])
    writer.finish()
    return root


def protocol(root):
    return {"schema_version": 1, "calibrated": True,
            "reference_manifest_sha256": digest_json(read_json(root / "manifest.json")),
            "fixture_sha256": read_json(root / "manifest.json")["fixture_sha256"],
            "limits": {key: 1e-4 for key in protocol_metrics(read_json(root / 'manifest.json')['fixture'])}}


def portable_trace(root, native_size=10, root_offset=0., env_id='Fill-v0'):
    s = state()
    names, rigid_count, derived, task = {
        'Fill-v0': (['ground', 'target_beaker'], 2, [0], np.zeros(2)),
        'Excavate-v0': (['ground', 'wall_0', 'wall_1', 'wall_2', 'wall_3'], 5, [0], np.zeros(1)),
        'Hang-v0': (['ground', 'rod'], 14, list(range(13)), np.array([0., 0., 1., 1., 1.])),
        'Pour-v0': (['ground', 'bottle', 'target_beaker'], 2, [], np.array([.01, .014])),
    }[env_id]
    count = len(names)
    s['rigid_pose'] = np.repeat(s['rigid_pose'], rigid_count, axis=0)
    s['rigid_velocity'] = np.zeros((rigid_count, 6))
    s.update(root_pose=np.array([[root_offset, 0., 0., 1., 0., 0., 0.]]),
             root_velocity=np.zeros((1, 6)), scene_actor_pose=s['rigid_pose'][:count].copy(),
             scene_actor_velocity=np.zeros((count, 6)), task_state=task, sim_state=np.zeros(native_size))
    if env_id == 'Hang-v0':
        s['rigid_pose'][0] = s['root_pose'][0]
    if env_id == 'Pour-v0':
        s['vol'] = np.array([1e-8, 2e-8])
        s['scene_actor_pose'] = np.r_[s['root_pose'], s['rigid_pose']]
    f = {'env_id': env_id, 'seed': 1, 'initial_state_contract': dict(version=2, derived_rigid_indices=derived, root_kind='fixed',
         scene_actor_names=names,
         scene_actor_types=['static'] + ['kinematic']*(count-1))}
    if env_id == 'Pour-v0':
        f['additional_state_fields'] = ['vol']
        f['initial_state_contract']['scene_actor_types'] = ['static', 'dynamic', 'kinematic']
    f['initial_numeric_sha256'] = digest_arrays(initial_numeric_state(s, f))
    writer = TraceWriter(root, fixture=f, provenance={'role': 'reference'}, steps=1)
    writer.frame(time_s=0., state=s, metrics={'success': False})
    writer.frame(time_s=.05, state=s, metrics=dict(success=False, reward=.1, terminated=False, truncated=False), action=[0., 0.])
    writer.finish()
    return root


def test_pour_contract_preserves_dynamic_bottle_and_fluid_volume(tmp_path):
    from copy import deepcopy
    from softbody_lab.artifacts import load_frame, validate_portable_state
    path = portable_trace(tmp_path/'pour', env_id='Pour-v0')
    fixture = validate_trace(path)['fixture']
    frame = load_frame(path/'frames/000000.npz')
    original = digest_arrays(initial_numeric_state(frame, fixture))
    changed = {k:v.copy() for k,v in frame.items()}
    changed['vol'][0] *= 1.1
    assert digest_arrays(initial_numeric_state(changed, fixture)) != original
    with pytest.raises(InvalidArtifact, match='evolving fluid volume'):
        validate_portable_state({k:v for k,v in frame.items() if k!='vol'}, fixture)
    wrong = deepcopy(fixture)
    wrong['initial_state_contract']['scene_actor_types'][1] = 'kinematic'
    with pytest.raises(InvalidArtifact, match='physical actor set'):
        validate_portable_state(frame, wrong)
    changed = {k:v.copy() for k,v in frame.items()}
    changed['scene_actor_velocity'][1,0] += .01
    with pytest.raises(InvalidArtifact, match='duplicate scene actor state'):
        validate_portable_state(changed, fixture)
    with pytest.raises(InvalidArtifact, match='fill-height'):
        validate_portable_state({**frame, 'task_state':np.array([.02,.01])}, fixture)


def test_hang_contract_preserves_rod_root_and_evaluation_indices(tmp_path):
    from softbody_lab.artifacts import load_frame, validate_portable_state
    path = portable_trace(tmp_path/'hang', env_id='Hang-v0')
    manifest = validate_trace(path)
    frame = load_frame(path/'frames/000000.npz')
    for key, row in [('scene_actor_pose', 1), ('root_pose', 0), ('root_velocity', 0)]:
        changed = {k:v.copy() for k,v in frame.items()}
        changed[key][row, 0] += .001
        with pytest.raises(InvalidArtifact, match='Inconsistent duplicate'):
            validate_portable_state(changed, manifest['fixture'])
    for index in (-1., .5, 2.):
        changed = {k:v.copy() for k,v in frame.items()}
        changed['task_state'][0] = index
        with pytest.raises(InvalidArtifact, match='particle indices'):
            validate_portable_state(changed, manifest['fixture'])


def test_hang_checks_last_derived_link_without_relaxing_rod(tmp_path):
    a, b = (portable_trace(tmp_path/k, env_id='Hang-v0') for k in 'ab')
    p = protocol(a); p['initial_derived_limits'] = dict.fromkeys(INITIAL_METRICS, 1e-8)
    edit_frame(b, 0, lambda s: s['rigid_pose'].__setitem__((12, 0), 2e-8))
    result = compare(a, b, p)
    assert any(f['metric'] == 'initial_position_m' for f in result['failures'])
    c = portable_trace(tmp_path/'c', env_id='Hang-v0')
    def move_rod(s):
        s['rigid_pose'][13, 0] += 1e-10
        s['scene_actor_pose'][1, 0] += 1e-10
    edit_frame(c, 0, move_rod)
    assert not compare(a, c, p)['passed']


@pytest.mark.parametrize('env_id', ['Fill-v0', 'Hang-v0', 'Pour-v0'])
def test_outcome_audit_rejects_forged_success_labels(tmp_path, env_id):
    from softbody_lab.task_checks import audit
    path = portable_trace(tmp_path/'trace', env_id=env_id)
    assert audit(path, fill_height=.08)['recorded_labels_match']
    edit_manifest(path, lambda m: m['samples'][1]['metrics'].update(success=True))
    result = audit(path, fill_height=.08)
    assert not result['recorded_labels_match']
    assert result['mismatched_steps'] == [1]


def test_excavate_contract_checks_every_wall_and_target(tmp_path):
    from softbody_lab.artifacts import load_frame, validate_portable_state
    path = portable_trace(tmp_path/'excavate', env_id='Excavate-v0')
    manifest = validate_trace(path)
    frame = load_frame(path/'frames/000000.npz')
    for i in range(1, 5):
        changed = {k:v.copy() for k,v in frame.items()}
        changed['scene_actor_pose'][i, 0] += .001
        with pytest.raises(InvalidArtifact, match='Inconsistent duplicate'):
            validate_portable_state(changed, manifest['fixture'])
    changed = {**frame, 'task_state': np.zeros(2)}
    with pytest.raises(InvalidArtifact, match='task_state'):
        validate_portable_state(changed, manifest['fixture'])


def test_portable_contract_keeps_native_buffers_but_compares_explicit_state(tmp_path):
    a, b = portable_trace(tmp_path/'a'), portable_trace(tmp_path/'b', native_size=17)
    p = protocol(a); p['initial_derived_limits'] = dict.fromkeys(INITIAL_METRICS, 1e-8)
    assert compare(a, b, p)['passed']
    c = portable_trace(tmp_path/'c', root_offset=2e-8)
    result = compare(a, c, p)
    assert not result['passed']
    assert any(f['metric'] == 'initial_position_m' for f in result['failures'])


@pytest.mark.parametrize('field', ['x', 'qpos', 'root_velocity', 'scene_actor_pose', 'scene_actor_velocity', 'task_state'])
def test_portable_reset_inputs_remain_exact(tmp_path, field):
    a, b = portable_trace(tmp_path/'a'), portable_trace(tmp_path/'b')
    p = protocol(a); p['initial_derived_limits'] = dict.fromkeys(INITIAL_METRICS, 1e-4)
    edit_frame(b, 0, lambda s: s[field].flat.__setitem__(0, s[field].flat[0]+1e-10))
    assert not compare(a, b, p)['passed']


@pytest.mark.parametrize('field', ['root_pose', 'root_velocity', 'scene_actor_pose', 'scene_actor_velocity', 'task_state', 'sim_state'])
def test_portable_fields_cannot_be_omitted(tmp_path, field):
    a, b = portable_trace(tmp_path/'a'), portable_trace(tmp_path/'b')
    p = protocol(a); p['initial_derived_limits'] = dict.fromkeys(INITIAL_METRICS, 1e-4)
    edit_frame(b, 1, lambda s: s.pop(field))
    assert not compare(a, b, p)['passed']


@pytest.mark.parametrize('field,value', [('reward', .2), ('reward', None), ('terminated', True), ('truncated', None)])
def test_portable_reward_and_episode_end_are_checked(tmp_path, field, value):
    a, b = portable_trace(tmp_path/'a'), portable_trace(tmp_path/'b')
    p = protocol(a); p['initial_derived_limits'] = dict.fromkeys(INITIAL_METRICS, 1e-4)
    edit_manifest(b, lambda m: m['samples'][1]['metrics'].update({field: value}))
    assert not compare(a, b, p)['passed']


def edit_manifest(root, edit):
    m = read_json(root / "manifest.json")
    edit(m)
    write_json(root / "manifest.json", m)


def edit_frame(root, step, edit):
    m = read_json(root / "manifest.json")
    p = root / m["samples"][step]["path"]
    with np.load(p) as z:
        values = dict(z)
    edit(values)
    np.savez_compressed(p, **values)
    m["samples"][step]["sha256"] = sha256(p)
    write_json(root / "manifest.json", m)


def test_identical_trace_passes_and_displaced_trace_fails(tmp_path):
    a, b = trace(tmp_path / "a"), trace(tmp_path / "b")
    assert compare(a, b, protocol(a))["passed"]
    c = trace(tmp_path / "c", offset=.01, role="candidate")
    result = compare(a, c, protocol(a))
    assert not result["passed"]
    assert result["max_errors"]["x_max_abs"] == pytest.approx(.01)
    assert result["max_errors"]["com_distance_m"] == pytest.approx(.01)


@pytest.mark.parametrize("fault", ["incomplete", "missing", "time", "actions", "fixture", "hash", "success"])
def test_manifest_faults_fail_closed(tmp_path, fault):
    a, b = trace(tmp_path / "a"), trace(tmp_path / "b")
    def edit(m):
        if fault == "incomplete": m["status"] = "running"
        if fault == "missing": m["samples"].pop()
        if fault == "time": m["samples"][1]["time_s"] = 0.
        if fault == "actions": m["actions"][0][0] = 1.
        if fault == "fixture": m["fixture"]["seed"] = 9
        if fault == "hash": m["samples"][1]["sha256"] = "0" * 64
        if fault == "success": m["samples"][1]["metrics"]["success"] = True
    edit_manifest(b, edit)
    assert not compare(a, b, protocol(a))["passed"]


@pytest.mark.parametrize("fault", ["nan", "mass", "count", "initial", "missing_control", "dtype"])
def test_numeric_faults_fail_even_with_valid_checksums(tmp_path, fault):
    a, b = trace(tmp_path / "a"), trace(tmp_path / "b")
    def edit(s):
        if fault == "nan": s["v"][0, 0] = np.nan
        if fault == "mass": s["mass"][0] += .1
        if fault == "count": s["x"] = s["x"][:1]
        if fault == "initial": s["x"][0, 0] += 1e-9
        if fault == "missing_control": del s["drive_position"]
        if fault == "dtype": s["qpos"] = np.array(["bad", "data"])
    edit_frame(b, 0 if fault == "initial" else 1, edit)
    assert not compare(a, b, protocol(a))["passed"]


def test_path_escape_and_symlink_rejected(tmp_path):
    a, b = trace(tmp_path / "a"), trace(tmp_path / "b")
    p = b / "frames/000001.npz"
    p.unlink()
    p.symlink_to(a / "frames/000001.npz")
    assert not compare(a, b, protocol(a))["passed"]
    edit_manifest(b, lambda m: m["samples"][1].update(path="../a/frames/000001.npz"))
    assert not compare(a, b, protocol(a))["passed"]


def test_unmeasured_or_incomplete_protocol_cannot_pass(tmp_path):
    a = trace(tmp_path / "a")
    p = protocol(a)
    p["calibrated"] = False
    assert not compare(a, a, p)["passed"]
    p["calibrated"] = True
    del p["limits"]["v_max_abs"]
    assert not compare(a, a, p)["passed"]


def test_calibration_measures_noise_and_rejects_excessive_variation(tmp_path):
    a = trace(tmp_path / "a")
    b, c = trace(tmp_path / "b", offset=1e-6), trace(tmp_path / "c", offset=2e-6)
    floors = dict.fromkeys(METRICS, 1e-7)
    ceilings = dict.fromkeys(METRICS, 1e-4)
    p = calibrate(a, [b, c], floors=floors, maximums=ceilings)
    assert p["limits"]["x_max_abs"] == pytest.approx(6e-6)
    assert compare(a, b, p)["passed"]
    with pytest.raises(ValueError, match="distinct"):
        calibrate(a, [b, b], floors=floors, maximums=ceilings)
    d = trace(tmp_path / "d", offset=.1)
    with pytest.raises(ValueError, match="Unstable"):
        calibrate(a, [b, d], floors=floors, maximums=ceilings)


def test_calibration_rejects_candidate_or_different_runtime(tmp_path):
    a, b, c = [trace(tmp_path / name) for name in "abc"]
    floors, ceilings = dict.fromkeys(METRICS, 0.), dict.fromkeys(METRICS, 1e-4)
    edit_manifest(c, lambda m: m["provenance"].update(role="candidate"))
    with pytest.raises(ValueError, match="Reference repeats"):
        calibrate(a, [b, c], floors=floors, maximums=ceilings)
    edit_manifest(c, lambda m: m["provenance"].update(role="reference", runtime={"kind": "other"}))
    with pytest.raises(ValueError, match="runtime"):
        calibrate(a, [b, c], floors=floors, maximums=ceilings)


def test_writer_preserves_failure_and_refuses_overwrite(tmp_path):
    p = trace(tmp_path / "a")
    with pytest.raises(FileExistsError):
        trace(p)
    edit_manifest(p, lambda m: m.update(status="failed", error="CUDA crash"))
    with pytest.raises(InvalidArtifact, match="incomplete"):
        validate_trace(p)


def test_json_digest_ignores_dict_order_but_rejects_nan():
    assert digest_json({"a": 1, "b": 2}) == digest_json({"b": 2, "a": 1})
    with pytest.raises(InvalidArtifact):
        digest_json({"v": float("nan")})


def derived_trace(root, noise=0.):
    s = state()
    s["rigid_pose"] = np.repeat(s["rigid_pose"], 2, axis=0)
    s["rigid_velocity"] = np.repeat(s["rigid_velocity"], 2, axis=0)
    s["rigid_pose"][1, 0] += noise
    fixture = {"env_id": "synthetic-derived-test",
               "initial_state_contract": {"version": 1, "derived_rigid_indices": [1]}}
    fixture["initial_numeric_sha256"] = digest_arrays(initial_numeric_state(s, fixture))
    writer = TraceWriter(root, fixture=fixture,
                         provenance={"role": "reference", "run_id": root.name,
                                     "source_commit": "a" * 40, "source_dirty": False,
                                     "runtime": {"kind": "synthetic"}, "capture_version": 2}, steps=1)
    writer.frame(time_s=0., state=s, metrics={"success": False})
    writer.frame(time_s=.02, state=s, metrics={"success": False}, action=[0., 0.])
    writer.finish()
    return root


def test_derived_initialization_requires_separate_measured_gate(tmp_path):
    a, b, c = (derived_trace(tmp_path / k, n) for k, n in zip('abc', [0., 1e-8, -2e-8]))
    kwargs = dict(floors=dict.fromkeys(METRICS, 0.), maximums=dict.fromkeys(METRICS, 1e-4))
    with pytest.raises(ValueError, match="explicit floors"):
        calibrate(a, [b, c], **kwargs)
    p = calibrate(a, [b, c], **kwargs, initial_floors=dict.fromkeys(INITIAL_METRICS, 0.),
                  initial_maximums=dict.fromkeys(INITIAL_METRICS, 1e-7))
    assert p["initial_derived_limits"]["position_m"] == pytest.approx(6e-8)
    assert compare(a, b, p)["passed"]
    # The very loose rollout ceiling cannot bypass the tight initial gate.
    p["limits"] = dict.fromkeys(METRICS, 1.)
    d = derived_trace(tmp_path / 'd', 1e-6)
    result = compare(a, d, p)
    assert not result["passed"]
    assert any(f["metric"] == 'initial_position_m' for f in result["failures"])
    from softbody_lab.fixtures import export_fixture, load_fixture
    exported = export_fixture(b, tmp_path / 'fixture')
    assert load_fixture(exported)[1]["rigid_pose"][1, 0] == 1e-8


@pytest.mark.parametrize("field", ["x", "qpos", "sim_state", "rigid_pose", "rigid_velocity"])
def test_derived_contract_does_not_relax_independent_reset_inputs(tmp_path, field):
    a, b = derived_trace(tmp_path/'a'), derived_trace(tmp_path/'b')
    p = protocol(a)
    p['initial_derived_limits'] = dict.fromkeys(INITIAL_METRICS, 1e-7)
    def edit(s):
        s[field].flat[0] += 1e-10
    edit_frame(b, 0, edit)
    assert not compare(a, b, p)['passed']


@pytest.mark.parametrize("value", [None, 0, "false"])
def test_missing_or_malformed_success_is_not_a_false_result(tmp_path, value):
    a, b = trace(tmp_path/'a'), trace(tmp_path/'b')
    edit_manifest(b, lambda m: m['samples'][0]['metrics'].update(success=value))
    assert not compare(a, b, protocol(a))['passed']


def test_protocol_cannot_be_reused_for_a_different_horizon(tmp_path):
    short = trace(tmp_path/'short', steps=1)
    long_a, long_b = trace(tmp_path/'long-a', steps=2), trace(tmp_path/'long-b', steps=2)
    result = compare(long_a, long_b, protocol(short))
    assert not result['passed']
    assert 'horizon' in result['failures'][0]['error']


def test_fluid_volume_is_compared_and_cannot_be_omitted(tmp_path):
    a, b = trace(tmp_path/'a'), trace(tmp_path/'b')
    for root in (a, b):
        def fixture(m):
            m['fixture']['additional_state_fields'] = ['vol']
            m['fixture_sha256'] = digest_json(m['fixture'])
        edit_manifest(root, fixture)
        for i in range(3):
            edit_frame(root, i, lambda s: s.update(vol=np.array([.0001, .0002])))
    p = protocol(a)
    p['limits']['vol_max_abs'] = 1e-10
    assert compare(a, b, p)['passed']
    edit_frame(b, 1, lambda s: s.update(vol=s['vol'] * 1.01))
    result = compare(a, b, p)
    assert not result['passed']
    assert any(f['metric'] == 'vol_max_abs' for f in result['failures'])
    edit_frame(b, 1, lambda s: s.pop('vol'))
    assert not compare(a, b, p)['passed']


def test_exported_fixture_preserves_and_verifies_actual_material_arrays(tmp_path):
    from softbody_lab.fixtures import export_fixture, load_fixture
    root = trace(tmp_path/'reference')
    material = {'particle_mass': np.array([.1,.2]), 'particle_type': np.array([0,1])}
    np.savez_compressed(root/'material.npz', **material)
    def edit(m):
        m['fixture']['material_sha256'] = digest_arrays(material)
        m['fixture_sha256'] = digest_json(m['fixture'])
        m['material_file_sha256'] = sha256(root/'material.npz')
    edit_manifest(root, edit)
    output = export_fixture(root, tmp_path/'fixture')
    restored = load_fixture(output)[3]
    assert all(np.array_equal(restored[k], v) for k, v in material.items())
    np.savez_compressed(output/'material.npz', particle_mass=np.array([.2,.2]))
    with pytest.raises(InvalidArtifact, match='material file checksum'):
        load_fixture(output)

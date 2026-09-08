"""Versioned numeric recordings. Candidate outputs are data, never executable pickle."""

from __future__ import annotations

import hashlib
import json
import math
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from . import SCHEMA_VERSION

STATE_SHAPES = {"x": (3,), "v": (3,), "F": (3, 3), "C": (3, 3), "vc": ()}
PORTABLE_FIELDS = ("root_pose", "root_velocity", "scene_actor_pose", "scene_actor_velocity", "task_state")
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_FRAME_BYTES = 256 * 1024 * 1024


class InvalidArtifact(ValueError):
    pass


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise InvalidArtifact(f"Not finite JSON data: {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    return json.dumps(json_value(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def digest_arrays(arrays: dict[str, np.ndarray]) -> str:
    """Hash numeric values, shapes and dtypes independently of NPZ zip timestamps."""
    result = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise InvalidArtifact(f"Invalid numeric array: {key}")
        result.update(canonical_json([key, value.dtype.str, list(value.shape)]))
        result.update(value.tobytes())
    return result.hexdigest()


def initial_numeric_state(state: dict, fixture: dict) -> dict:
    """Separate reset inputs from link poses derived by forward kinematics.

    Version 1 fixtures keep their historical exact digest. The opt-in contract
    additionally hashes native sim_state (independent actor and articulation
    state) and excludes only the declared derived link rows from that digest.
    Derived rows remain in the recording and have a separate measured gate.
    """
    from .telemetry import validate_telemetry
    validate_telemetry(state, fixture)
    contract = fixture.get("initial_state_contract")
    if contract is None:
        return {k: v for k, v in state.items() if k != "sim_state"}
    indices = contract.get("derived_rigid_indices")
    version = contract.get("version")
    if version not in (1, 2) or not isinstance(indices, list):
        raise InvalidArtifact("Unknown initial state contract")
    if any(type(i) is not int for i in indices) or indices != sorted(set(indices)):
        raise InvalidArtifact("Derived rigid indices must be unique sorted integers")
    poses, velocities = state["rigid_pose"], state["rigid_velocity"]
    if poses.ndim != 2 or poses.shape[1] != 7 or velocities.shape != (len(poses), 6):
        raise InvalidArtifact("Invalid coupled rigid state shape")
    if any(i < 0 or i >= len(poses) for i in indices):
        raise InvalidArtifact("Derived rigid index is out of bounds")
    independent = np.ones(len(poses), dtype=bool)
    independent[indices] = False
    result = {**state, "rigid_pose": poses[independent], "rigid_velocity": velocities[independent]}
    if version == 2:
        # The native buffer duplicates explicit particle/joint/actor fields and
        # includes renderer-specific actors and derived root poses. Preserve it
        # as diagnostic data, never forge a cross-runtime native buffer.
        validate_portable_state(state, fixture)
        result.pop('sim_state')
        result.pop('root_pose')  # fixed-base native FK readback; separately gated
    return result


def validate_portable_state(state, fixture):
    contract = fixture['initial_state_contract']
    tasks = {'Pinch-v0': (['ground'], 2+len(state['x'])*3, 3, [0,1,2]),
             'Fill-v0': (['ground', 'target_beaker'], 2, 2, [0]),
             'Excavate-v0': (['ground', 'wall_0', 'wall_1', 'wall_2', 'wall_3'], 1, 5, [0]),
             'Hang-v0': (['ground', 'rod'], 5, 14, list(range(13))),
             'Write-v0': (['ground', 'wall_0', 'wall_1', 'wall_2', 'wall_3'], 19404*3, 5, [0]),
             'Pour-v0': (['ground', 'bottle', 'target_beaker'], 2, 2, [])}
    if fixture.get('env_id') not in tasks or contract.get('root_kind') != 'fixed':
        raise InvalidArtifact('Unsupported portable fixed-base task')
    names, task_size, rigid_count, derived_indices = tasks[fixture['env_id']]
    count = len(names)
    types = ['static', 'dynamic', 'kinematic'] if fixture['env_id'] == 'Pour-v0' else ['static'] + ['kinematic']*(count-1)
    if contract.get('scene_actor_names') != names or contract.get('scene_actor_types') != types:
        raise InvalidArtifact('Portable task must record its entire physical actor set')
    if contract.get('derived_rigid_indices') != derived_indices:
        raise InvalidArtifact('Unexpected derived robot link indices')
    expected = {'rigid_pose': (rigid_count, 7), 'rigid_velocity': (rigid_count, 6), 'root_pose': (1, 7), 'root_velocity': (1, 6),
                'scene_actor_pose': (count, 7), 'scene_actor_velocity': (count, 6), 'task_state': (task_size,)}
    for key, shape in expected.items():
        if key not in state or state[key].shape != shape:
            raise InvalidArtifact(f'Missing or invalid portable field: {key}')
    independent = [i for i in range(rigid_count) if i not in derived_indices]
    if not np.array_equal(state['rigid_pose'][independent], state['scene_actor_pose'][1:]) or not np.array_equal(state['rigid_velocity'][independent], state['scene_actor_velocity'][1:]):
        raise InvalidArtifact('Inconsistent duplicate scene actor state')
    if fixture['env_id'] == 'Hang-v0':
        if not np.array_equal(state['root_pose'][0], state['rigid_pose'][0]) or not np.array_equal(state['root_velocity'][0], state['rigid_velocity'][0]):
            raise InvalidArtifact('Inconsistent duplicate fixed-root state')
        indices = state['task_state']
        if np.any(indices != np.floor(indices)) or np.any(indices < 0) or np.any(indices >= len(state['x'])):
            raise InvalidArtifact('Invalid rope evaluation particle indices')
    if fixture['env_id'] == 'Pour-v0':
        if 'vol' not in state or fixture.get('additional_state_fields') != ['vol']:
            raise InvalidArtifact('Pour requires actual evolving fluid volume')
        heights = state['task_state']
        if not 0 < heights[0] < heights[1]:
            raise InvalidArtifact('Invalid Pour fill-height state')
    if fixture['env_id']=='Pinch-v0':
        if np.any(state['task_state'][:2]<0) or state['task_state'][:2].sum()<=0:
            raise InvalidArtifact('Pinch deformation distance must have positive total')
        for field,shape in [('goal_depths',(4,128,128)),('goal_rgbs',(4,128,128,3)),('goal_cam_pos',(4,3)),
                            ('goal_cam_rot',(4,4)),('goal_cam_intrinsic',(3,3))]:
            if field not in state or state[field].shape!=shape:
                raise InvalidArtifact('Pinch camera goal fields required')
        if np.any(state['goal_depths']<0) or np.any(state['goal_rgbs']<0) or np.any(state['goal_rgbs']>255):
            raise InvalidArtifact('Invalid Pinch depth or RGB')
    if fixture['env_id'] == 'Write-v0':
        if len(state['x']) != 19404:
            raise InvalidArtifact('Unexpected Write material particle count')
        for field in ('goal_height_mm','current_height_mm'):
            image=state.get(field)
            if image is None or image.shape!=(64,64) or image.dtype.kind not in 'iu' or np.any(image<0):
                raise InvalidArtifact('Invalid Write raster telemetry')


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path: Path, value: Any) -> None:
    data = canonical_json(value)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data + b"\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    if path.is_symlink() or path.stat().st_size > MAX_JSON_BYTES:
        raise InvalidArtifact(f"Invalid JSON file: {path}")
    def invalid_constant(value):
        raise InvalidArtifact(f"Non-finite JSON constant: {value}")
    return json.loads(path.read_bytes(), parse_constant=invalid_constant)


def child_file(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise InvalidArtifact("Artifact path escapes recording")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise InvalidArtifact("Artifact symlinks are forbidden")
    if not current.is_file():
        raise InvalidArtifact(f"Missing artifact: {name}")
    return current


def validate_state(state: dict[str, np.ndarray], n: int | None = None) -> int:
    required = {*STATE_SHAPES, "mass", "qpos", "qvel", "sim_state"}
    if not required <= state.keys():
        raise InvalidArtifact(f"Missing state fields: {sorted(required - state.keys())}")
    count = len(state["x"])
    if count < 1 or (n is not None and count != n):
        raise InvalidArtifact("Particle count changed or is zero")
    for key, tail in STATE_SHAPES.items():
        if state[key].shape != (count, *tail):
            raise InvalidArtifact(f"Invalid shape for {key}: {state[key].shape}")
    if state["mass"].shape != (count,) or np.any(state["mass"] <= 0):
        raise InvalidArtifact("Particle masses must be positive and per-particle")
    if "vol" in state and (state["vol"].shape != (count,) or np.any(state["vol"] <= 0)):
        raise InvalidArtifact("Fluid volumes must be positive and per-particle")
    if state["qpos"].ndim != 1 or state["qvel"].shape != state["qpos"].shape:
        raise InvalidArtifact("Invalid joint state")
    if state["sim_state"].ndim != 1:
        raise InvalidArtifact("Native simulation state must be a flat numeric array")
    for key, value in state.items():
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise InvalidArtifact(f"Non-finite or nonnumeric state: {key}")
    return count


def load_frame(path: Path) -> dict[str, np.ndarray]:
    if path.stat().st_size > MAX_FRAME_BYTES:
        raise InvalidArtifact("Compressed frame exceeds size limit")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > 128 or sum(x.file_size for x in members) > MAX_FRAME_BYTES:
            raise InvalidArtifact("Expanded frame exceeds size limit")
        if len({x.filename for x in members}) != len(members):
            raise InvalidArtifact("Duplicate archive members")
    with np.load(path, allow_pickle=False) as arrays:
        result = {key: arrays[key] for key in arrays.files}
    validate_state(result)
    return result


class TraceWriter:
    def __init__(self, root: Path, *, fixture: dict, provenance: dict, steps: int):
        if steps < 1:
            raise ValueError("steps must be positive")
        self.root = root
        root.mkdir(parents=True, exist_ok=False)
        (root / "frames").mkdir()
        self.manifest = {
            "schema_version": SCHEMA_VERSION, "status": "running",
            "fixture": json_value(fixture), "fixture_sha256": digest_json(fixture),
            "provenance": json_value(provenance), "requested_steps": steps,
            "samples": [], "actions": [],
        }
        self.count = None
        write_json(root / "manifest.json", self.manifest)

    def frame(self, *, time_s: float, state: dict, metrics: dict, action=None) -> None:
        self.count = validate_state(state, self.count)
        index = len(self.manifest["samples"])
        if not math.isfinite(time_s) or (index == 0 and time_s != 0):
            raise InvalidArtifact("Initial sample must be at time zero")
        if index and time_s <= self.manifest["samples"][-1]["time_s"]:
            raise InvalidArtifact("Simulation time must increase")
        if (index == 0) != (action is None):
            raise InvalidArtifact("Each subsequent sample needs its actual action")
        name = f"frames/{index:06d}.npz"
        np.savez_compressed(root_path := self.root / name, **state)
        self.manifest["samples"].append({"step": index, "time_s": time_s,
                                        "path": name, "sha256": sha256(root_path),
                                        "metrics": json_value(metrics)})
        if action is not None:
            self.manifest["actions"].append(json_value(action))
        write_json(self.root / "manifest.json", self.manifest)

    def finish(self, error: str | None = None) -> None:
        complete = len(self.manifest["samples"]) == self.manifest["requested_steps"] + 1
        self.manifest["status"] = "complete" if complete and error is None else "failed"
        self.manifest["error"] = error or (None if complete else "Incomplete rollout")
        write_json(self.root / "manifest.json", self.manifest)


def validate_trace(root: Path) -> dict:
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise InvalidArtifact("Unsupported recording schema")
    if manifest.get("status") != "complete" or manifest.get("error") is not None:
        raise InvalidArtifact("Recording failed or is incomplete")
    if manifest.get("fixture_sha256") != digest_json(manifest["fixture"]):
        raise InvalidArtifact("Fixture checksum mismatch")
    if "material_sha256" in manifest["fixture"]:
        material_path = child_file(root, "material.npz")
        if sha256(material_path) != manifest.get("material_file_sha256"):
            raise InvalidArtifact("Material file checksum mismatch")
        if material_path.stat().st_size > MAX_FRAME_BYTES:
            raise InvalidArtifact("Material file exceeds size limit")
        with zipfile.ZipFile(material_path) as archive:
            if sum(x.file_size for x in archive.infolist()) > MAX_FRAME_BYTES:
                raise InvalidArtifact("Expanded material exceeds size limit")
        with np.load(material_path, allow_pickle=False) as arrays:
            if digest_arrays(dict(arrays)) != manifest["fixture"]["material_sha256"]:
                raise InvalidArtifact("Material values differ from the fixture")
    steps = manifest["requested_steps"]
    if not isinstance(steps, int) or steps < 1 or steps > 100000:
        raise InvalidArtifact("Invalid step count")
    if len(manifest["samples"]) != steps + 1 or len(manifest["actions"]) != steps:
        raise InvalidArtifact("Missing trajectory samples/actions")
    previous = -1.0
    count = None
    masses = None
    paths = set()
    for i, sample in enumerate(manifest["samples"]):
        time = sample["time_s"]
        if sample["step"] != i or not math.isfinite(time) or time <= previous:
            raise InvalidArtifact("Invalid sample ordering/time")
        if i == 0 and time != 0:
            raise InvalidArtifact("Missing initial state")
        previous = time
        if sample["path"] in paths:
            raise InvalidArtifact("Repeated frame path")
        paths.add(sample["path"])
        path = child_file(root, sample["path"])
        if sha256(path) != sample["sha256"]:
            raise InvalidArtifact(f"Frame checksum mismatch at {i}")
        state = load_frame(path)
        count = validate_state(state, count)
        from .telemetry import validate_telemetry
        validate_telemetry(state, manifest['fixture'])
        if manifest['fixture'].get('initial_state_contract', {}).get('version') == 2:
            validate_portable_state(state, manifest['fixture'])
        for field in manifest["fixture"].get("additional_state_fields", []):
            if field != "vol" or field not in state:
                raise InvalidArtifact("Missing or unsupported additional particle state field")
        if i == 0 and "initial_numeric_sha256" in manifest["fixture"]:
            if digest_arrays(initial_numeric_state(state, manifest["fixture"])) != manifest["fixture"]["initial_numeric_sha256"]:
                raise InvalidArtifact("Initial numeric values differ from fixture")
        if masses is None:
            masses = state["mass"]
        elif not np.array_equal(masses, state["mass"]):
            raise InvalidArtifact("Particle masses changed during rollout")
    return manifest

"""Pure evaluator: no candidate imports, no simulator imports, no self-reported verdicts."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .artifacts import (InvalidArtifact, child_file, digest_json, digest_arrays,
                        initial_numeric_state, load_frame, validate_trace, PORTABLE_FIELDS)

FIELDS = ("x", "v", "F", "C", "vc", "qpos", "qvel", "drive_position", "drive_velocity",
          "rigid_pose", "rigid_velocity")
METRICS = (*[f"{key}_max_abs" for key in FIELDS], "com_distance_m", "mass_abs_kg")
INITIAL_METRICS = ("position_m", "quaternion_component", "linear_velocity_m_s", "angular_velocity_rad_s")


def protocol_metrics(fixture):
    extra = fixture.get("additional_state_fields", [])
    if extra not in ([], ["vol"]):
        raise InvalidArtifact("Unsupported additional particle state fields")
    portable = PORTABLE_FIELDS if fixture.get('initial_state_contract', {}).get('version') == 2 else ()
    images = ('goal_height_mm','current_height_mm') if fixture.get('env_id')=='Write-v0' else ()
    return (*METRICS, *[f"{key}_max_abs" for key in (*extra, *portable, *images)], *(['reward_abs'] if portable else []))


def derived_initial_errors(left, right, fixture):
    indices = fixture["initial_state_contract"]["derived_rigid_indices"]
    pose = abs(left["rigid_pose"][indices].astype(float) - right["rigid_pose"][indices].astype(float))
    velocity = abs(left["rigid_velocity"][indices].astype(float) - right["rigid_velocity"][indices].astype(float))
    if fixture['initial_state_contract'].get('version') == 2:
        pose = np.concatenate([pose, abs(left['root_pose'].astype(float) - right['root_pose'].astype(float))])
    return dict(zip(INITIAL_METRICS, [float(np.max(a, initial=0)) for a in
                                     (pose[:, :3], pose[:, 3:], velocity[:, :3], velocity[:, 3:])]))


def validate_limits(limits, metrics):
    if set(limits) != set(metrics):
        raise InvalidArtifact("Protocol must set every metric limit exactly once")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in limits.values()):
        raise InvalidArtifact("Invalid metric limit")


def frame_errors(reference: dict, candidate: dict) -> dict[str, float]:
    errors = {}
    fields = (*FIELDS, "vol") if "vol" in reference or "vol" in candidate else FIELDS
    if any(k in reference or k in candidate for k in PORTABLE_FIELDS):
        fields = (*fields, *PORTABLE_FIELDS)
    if any(k in reference or k in candidate for k in ('goal_height_mm','current_height_mm')):
        fields = (*fields, 'goal_height_mm', 'current_height_mm')
    for name in fields:
        if name not in reference or name not in candidate:
            raise InvalidArtifact(f"Missing comparable field: {name}")
        if reference[name].shape != candidate[name].shape:
            raise InvalidArtifact(f"Shape mismatch: {name}")
        errors[f"{name}_max_abs"] = float(np.max(np.abs(
            reference[name].astype(np.float64) - candidate[name].astype(np.float64)
        ), initial=0))
    rm, cm = reference["mass"].sum(), candidate["mass"].sum()
    rc = np.average(reference["x"], axis=0, weights=reference["mass"])
    cc = np.average(candidate["x"], axis=0, weights=candidate["mass"])
    errors["com_distance_m"] = float(np.linalg.norm(rc - cc))
    errors["mass_abs_kg"] = float(abs(rm - cm))
    return errors


def compare(reference: Path, candidate: Path, protocol: dict) -> dict:
    """A strict short-horizon/identity protocol. Long-horizon gates are separate."""
    result = {"passed": False, "protocol_sha256": digest_json(protocol),
              "scope": "short-horizon-particle-identity", "failures": [], "max_errors": {}}
    try:
        if protocol.get("schema_version") != 1 or protocol.get("calibrated") is not True:
            raise InvalidArtifact("Protocol must be calibrated against reference repeats")
        limits = protocol["limits"]
        baseline = validate_trace(reference)
        trial = validate_trace(candidate)
        if protocol.get("reference_manifest_sha256") != digest_json(baseline):
            raise InvalidArtifact("Protocol is for a different reference recording or horizon")
        metrics = protocol_metrics(baseline["fixture"])
        validate_limits(limits, metrics)
        if baseline["fixture_sha256"] != trial["fixture_sha256"]:
            raise InvalidArtifact("Fixtures differ")
        if protocol.get("fixture_sha256") != baseline["fixture_sha256"]:
            raise InvalidArtifact("Protocol is for a different fixture")
        has_derived = "initial_state_contract" in baseline["fixture"]
        if has_derived:
            validate_limits(protocol["initial_derived_limits"], INITIAL_METRICS)
        if len(baseline["samples"]) != len(trial["samples"]):
            raise InvalidArtifact("Trajectory lengths differ")
        if baseline["actions"] != trial["actions"]:
            raise InvalidArtifact("Applied actions differ")
        maxima = dict.fromkeys(metrics, 0.0)
        for a, b in zip(baseline["samples"], trial["samples"]):
            if a["time_s"] != b["time_s"]:
                raise InvalidArtifact("Simulation sample times differ")
            left = load_frame(child_file(reference, a["path"]))
            right = load_frame(child_file(candidate, b["path"]))
            if "vol_max_abs" in metrics and ("vol" not in left or "vol" not in right):
                raise InvalidArtifact("Missing fluid particle volume state")
            errors = frame_errors(left, right)
            if 'reward_abs' in metrics and a['step'] > 0:
                rewards = [s['metrics'].get('reward') for s in (a, b)]
                if any(type(v) not in (int, float) or not math.isfinite(v) for v in rewards):
                    raise InvalidArtifact('Missing or invalid reward metric')
                errors['reward_abs'] = abs(rewards[0] - rewards[1])
                for key in ('terminated', 'truncated'):
                    values = [s['metrics'].get(key) for s in (a, b)]
                    if any(type(v) is not bool for v in values):
                        raise InvalidArtifact(f'Missing or invalid {key} metric')
                    if values[0] != values[1]:
                        result['failures'].append({'step': a['step'], 'metric': key})
            for key, value in errors.items():
                maxima[key] = max(maxima[key], value)
            if a["step"] == 0:
                if digest_arrays(initial_numeric_state(left, baseline["fixture"])) != digest_arrays(initial_numeric_state(right, baseline["fixture"])):
                    raise InvalidArtifact("Independent initial state differs")
                if has_derived:
                    initial_errors = derived_initial_errors(left, right, baseline["fixture"])
                    result["initial_derived_errors"] = initial_errors
                    for key, value in initial_errors.items():
                        limit = protocol["initial_derived_limits"][key]
                        if value > limit:
                            result["failures"].append({"step": 0, "metric": f"initial_{key}", "value": value, "limit": limit})
            for key, value in errors.items():
                if value > limits[key]:
                    result["failures"].append({"step": a["step"], "metric": key,
                                                "value": value, "limit": limits[key]})
            if any(type(sample["metrics"].get("success")) is not bool for sample in (a, b)):
                raise InvalidArtifact("Missing or non-boolean task success metric")
            if a["metrics"]["success"] != b["metrics"]["success"]:
                result["failures"].append({"step": a["step"], "metric": "legacy_success"})
        result["max_errors"] = maxima
        result["reference_provenance"] = baseline["provenance"]
        result["candidate_provenance"] = trial["provenance"]
        result["passed"] = not result["failures"]
    except (InvalidArtifact, KeyError, TypeError, ValueError, OSError) as error:
        result["failures"].append({"metric": "artifact_integrity", "error": str(error)})
    return result


def calibrate(reference: Path, repeats: list[Path], *, floors: dict,
              maximums: dict, multiplier: float = 3.0,
              initial_floors: dict | None = None, initial_maximums: dict | None = None) -> dict:
    """Measure repeat noise; refuse fixtures requiring physically unacceptable limits."""
    if len(repeats) < 2:
        raise ValueError("At least two additional independent reference runs are required")
    baseline = validate_trace(reference)
    metrics = protocol_metrics(baseline["fixture"])
    if set(floors) != set(metrics) or set(maximums) != set(metrics):
        raise ValueError("Every metric needs an explicit floor and maximum")
    if not math.isfinite(multiplier) or multiplier < 1:
        raise ValueError("Noise multiplier must be finite and >= 1")
    for key in metrics:
        if not 0 <= floors[key] <= maximums[key] or not math.isfinite(maximums[key]):
            raise ValueError(f"Invalid calibration range: {key}")
    provenance = baseline["provenance"]
    if provenance.get("role") != "reference" or not provenance.get("run_id"):
        raise ValueError("Calibration requires an identified reference run")
    protocol = {"schema_version": 1, "calibrated": True,
                "fixture_sha256": baseline["fixture_sha256"], "limits": maximums,
                "reference_manifest_sha256": digest_json(baseline)}
    has_derived = "initial_state_contract" in baseline["fixture"]
    initial_noise = dict.fromkeys(INITIAL_METRICS, 0.)
    if has_derived:
        if initial_floors is None or initial_maximums is None:
            raise ValueError("Derived initialization needs explicit floors and ceilings")
        validate_limits(initial_floors, INITIAL_METRICS)
        validate_limits(initial_maximums, INITIAL_METRICS)
        if any(initial_floors[k] > initial_maximums[k] for k in INITIAL_METRICS):
            raise ValueError("Initial floor exceeds ceiling")
        protocol["initial_derived_limits"] = initial_maximums
    noise = dict.fromkeys(metrics, 0.0)
    ids = {provenance["run_id"]}
    evidence = []
    for path in repeats:
        trial = validate_trace(path)
        p = trial["provenance"]
        if p.get("role") != "reference" or not p.get("run_id") or p["run_id"] in ids:
            raise ValueError("Reference repeats must have distinct run IDs")
        ids.add(p["run_id"])
        for key in ("source_commit", "source_dirty", "runtime", "capture_version"):
            if key not in provenance or p.get(key) != provenance[key]:
                raise ValueError(f"Reference provenance differs: {key}")
        result = compare(reference, path, protocol)
        if not result["passed"]:
            raise ValueError(f"Unstable or incompatible reference: {result['failures'][:3]}")
        for key in metrics:
            noise[key] = max(noise[key], result["max_errors"][key])
        if has_derived:
            for key in INITIAL_METRICS:
                initial_noise[key] = max(initial_noise[key], result["initial_derived_errors"][key])
        evidence.append({"run_id": p["run_id"], "manifest_sha256": digest_json(trial)})
    limits = {key: max(floors[key], multiplier * noise[key]) for key in metrics}
    if any(limits[key] > maximums[key] for key in metrics):
        raise ValueError("Reference variability exceeds the declared calibration ceiling")
    if has_derived:
        initial_limits = {k: max(initial_floors[k], multiplier * initial_noise[k]) for k in INITIAL_METRICS}
        if any(initial_limits[k] > initial_maximums[k] for k in INITIAL_METRICS):
            raise ValueError("Derived initialization variability exceeds its ceiling")
        protocol.update(initial_derived_limits=initial_limits,
                        initial_measured_noise=initial_noise,
                        initial_floors=initial_floors, initial_maximums=initial_maximums)
    return {**protocol, "limits": limits, "measured_noise": noise,
            "noise_multiplier": multiplier, "floors": floors, "maximums": maximums,
            "reference_manifest_sha256": digest_json(baseline), "repeats": evidence}

"""Measure native SAPIEN 3 pose setter/getter residuals without stepping physics.

Only numpy and sapien are imported from outside the standard library. Fill is
never imported. Exported numbers go directly into sapien.Pose, without rounding,
normalization, sign alignment, matrix conversion, or corrective assignments.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import sapien


CHECKOUT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = CHECKOUT / "mani_skill/envs/softbody/legacy_fill_physics.json"
INFRA = CHECKOUT.parents[3]
DEFAULT_ASSET_DOCS = INFRA / "docs/asset-library.md"
DEFAULT_ASSET_CATALOG = (
    INFRA / "packages/sim-infra/src/sim_infra/asset_library/catalog.json"
)
IDENTITY = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
# Authored numeric controls, not robot states or recovered outcomes. Decimal
# literals intentionally exercise the Python float64 -> native float32 boundary.
REPRESENTATIVE_POSES = {
    "identity": IDENTITY,
    "binary_exact_multiaxis": [0.125, -0.25, 0.5, 0.5, -0.5, 0.5, -0.5],
    "negative_quaternion": [0.125, -0.25, 0.5, -0.5, 0.5, -0.5, 0.5],
    "decimal_multiaxis": [
        0.12345678901234566, -0.23456789012345677, 0.3456789012345679,
        0.18257418583505536, 0.3651483716701107,
        0.5477225575051661, 0.7302967433402214,
    ],
    "half_turn_small_translation": [
        7.450580596923828e-09, -0.08249999582767487,
        0.3330000340938568, 0.0, 1.0, 0.0, 0.0,
    ],
}


def file_record(path):
    path = Path(path).resolve()
    payload = path.read_bytes()
    return {
        "path": str(path), "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def git_head(path):
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def inspect_asset_library(docs_path, catalog_path):
    # This happens before constructing any scene. No catalog payload is opened.
    docs = docs_path.read_text()
    catalog = json.loads(catalog_path.read_text())
    if not docs.strip() or not isinstance(catalog.get("assets"), dict):
        raise ValueError("Expected asset-library documentation and an assets catalog")
    return {
        "docs": file_record(docs_path), "catalog": file_record(catalog_path),
        "catalog_asset_ids": sorted(catalog["assets"]),
        "decision": "No geometry is needed for native pose setters; no assets loaded.",
    }


def pose_values(pose):
    # Copy getters immediately. Widening float32 to float64 is lossless; subtract
    # in float64 so the diagnostic does not round away small residuals.
    return np.concatenate((pose.p.copy(), pose.q.copy())).astype(np.float64)


def residual(actual, requested):
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(requested, dtype=np.float64)
    return {
        "components_xyz_wxyz": delta.tolist(),
        "position_max_abs_m": float(np.max(np.abs(delta[:3]))),
        "quaternion_component_max_abs": float(np.max(np.abs(delta[3:]))),
    }


def float32_bits(values):
    # Encoding only, never fed back into a setter. Includes the sign of zero.
    return [f"{int(v):08x}" for v in np.asarray(values, dtype=np.float32).view(np.uint32)]


def measure_pose(case_id, group, target, attribute, requested, tokens=None):
    requested = list(requested)
    pose = sapien.Pose(requested[:3], requested[3:])
    assigned = pose_values(pose)
    setattr(target, attribute, pose)
    returned_pose = getattr(target, attribute)
    readback = pose_values(returned_pose)
    reread = pose_values(getattr(target, attribute))
    method_readback = pose_values(getattr(target, "get_" + attribute)())
    # Repeat the original input, not the getter output. This is still setup;
    # neither scene.step nor articulation/root/qpos setters are called anywhere.
    getattr(target, "set_" + attribute)(pose)
    reassigned = pose_values(getattr(target, attribute))
    pose_after = pose_values(pose)
    return {
        "id": case_id, "group": group,
        "native_target": type(target).__module__ + "." + type(target).__name__,
        "attribute": attribute,
        "input_xyz_wxyz": requested,
        "input_decimal_tokens": tokens,
        "sapien_pose_xyz_wxyz": assigned.tolist(),
        "readback_xyz_wxyz": readback.tolist(),
        "reread_xyz_wxyz": reread.tolist(),
        "method_readback_xyz_wxyz": method_readback.tolist(),
        "same_input_reassignment_xyz_wxyz": reassigned.tolist(),
        "input_pose_after_assignment_xyz_wxyz": pose_after.tolist(),
        "native_dtypes": {"p": str(returned_pose.p.dtype), "q": str(returned_pose.q.dtype)},
        "sapien_pose_float32_bits": float32_bits(assigned),
        "readback_float32_bits": float32_bits(readback),
        "input_quaternion_norm_float64": float(np.linalg.norm(requested[3:])),
        "readback_quaternion_norm_float64": float(np.linalg.norm(readback[3:])),
        "residuals": {
            "pose_minus_input": residual(assigned, requested),
            "readback_minus_pose": residual(readback, assigned),
            "readback_minus_input": residual(readback, requested),
            "reread_minus_readback": residual(reread, readback),
            "method_minus_property": residual(method_readback, readback),
            "reassignment_minus_readback": residual(reassigned, readback),
            "input_pose_mutation": residual(pose_after, assigned),
        },
    }


def make_fixture(parent_record, child_record, joint_type, include_dynamic=False):
    # Explicit systems avoids Scene()'s default RenderSystem. Direct components
    # avoid builders, URDF loaders, collision cooking and auto-derived inertia.
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    parent = sapien.physx.PhysxArticulationLinkComponent()
    child = sapien.physx.PhysxArticulationLinkComponent(parent)
    parent.joint.type = "fixed"
    child.joint.type = joint_type
    bodies = [("parent", parent, parent_record), ("child", child, child_record)]
    if include_dynamic:
        bodies.append(("dynamic", sapien.physx.PhysxRigidDynamicComponent(), child_record))
    entities = []
    for role, body, record in bodies:
        entity = sapien.Entity()
        entity.name = role
        entity.add_component(body)
        body.mass = record["mass"]
        body.inertia = record["inertia"]
        entities.append(entity)
    for entity in entities:
        scene.add_entity(entity)
    # Structural checks only; there are no numerical acceptance thresholds.
    if len(scene.physx_system.articulation_link_components) != 2:
        raise RuntimeError("Expected two registered native articulation links")
    if any(body.collision_shapes for _, body, _ in bodies):
        raise RuntimeError("Setter diagnostic must not contain collision geometry")
    return scene, child.joint, bodies


def mass_record(role, body, requested):
    mass, inertia = float(body.mass), body.inertia.astype(np.float64)
    return {
        "role": role, "source_link": requested["name"],
        "input_mass_kg": requested["mass"], "readback_mass_kg": mass,
        "mass_residual_kg": mass - requested["mass"],
        "input_inertia_kg_m2": requested["inertia"],
        "readback_inertia_kg_m2": inertia.tolist(),
        "inertia_residual_kg_m2": (inertia - requested["inertia"]).tolist(),
        "auto_compute_mass_readback": bool(body.auto_compute_mass),
    }


def summarize(records):
    result = {}
    for group in sorted({r["group"] for r in records}):
        selected = [r for r in records if r["group"] == group]
        stages = {}
        for stage in selected[0]["residuals"]:
            stages[stage] = {}
            for metric in ("position_max_abs_m", "quaternion_component_max_abs"):
                worst = max(selected, key=lambda r: r["residuals"][stage][metric])
                stages[stage][metric] = {
                    "value": worst["residuals"][stage][metric], "case": worst["id"],
                }
        result[group] = {"count": len(selected), "residual_maxima": stages}
    return result


def run(args):
    asset_inspection = inspect_asset_library(args.asset_docs, args.asset_catalog)
    raw_input = args.parameters.read_text()
    document = json.loads(raw_input)
    parameters = document["robot_parameters"]
    # Retain original decimal spellings as well as parsed numeric values.
    tokens = json.loads(raw_input, parse_float=str, parse_int=str)["robot_parameters"]
    links, joints = parameters["links"], parameters["joints"]
    if len(links) != 10 or len(joints) != 9:
        raise ValueError("This focused probe expects the exported Fill 10 links / 9 joints")
    records, fixtures = [], []

    # Each exported COM is exercised on a root link, child link, and free rigid
    # body, using that record's unchanged mass and principal inertia.
    com_cases = [
        ("exported_com/" + r["name"], "exported_com", r, r["com"], t["com"])
        for r, t in zip(links, tokens["links"])
    ] + [
        ("representative_com/" + name, "representative_com", links[0], pose, None)
        for name, pose in REPRESENTATIVE_POSES.items()
    ]
    for case_id, group, record, requested, decimal_tokens in com_cases:
        scene, _, bodies = make_fixture(record, record, "fixed", include_dynamic=True)
        try:
            for role, body, source in bodies:
                records.append(measure_pose(
                    case_id + "/" + role, group, body, "cmass_local_pose",
                    requested, decimal_tokens,
                ))
            fixtures.append({
                "id": case_id, "joint_type": "fixed", "registered_bodies": len(bodies),
                "mass_properties": [mass_record(*b) for b in bodies],
            })
        finally:
            scene.clear()

    # The export has no topology/types. Neighboring records supply COM contexts
    # only, not a claim to reconstruct Fill FK. Test BOTH native joint types for
    # every frame. The identity-COM fixtures are explicitly separate controls.
    joint_cases = []
    for i, (record, token_record) in enumerate(zip(joints, tokens["joints"])):
        for context in ("exported_com", "identity_com_control"):
            joint_cases.append((
                "exported_joint/" + record["name"] + "/" + context,
                "exported_joint" if context == "exported_com" else "exported_joint_identity_com_control",
                links[i], links[i + 1], context, record, token_record,
            ))
    for name, pose in REPRESENTATIVE_POSES.items():
        for context in ("exported_com", "identity_com_control"):
            joint_cases.append((
                "representative_joint/" + name + "/" + context,
                "representative_joint_" + context, links[0], links[1], context,
                {"parent_pose": pose, "child_pose": pose},
                {"parent_pose": None, "child_pose": None},
            ))
    for case_id, group, parent, child, context, frame, frame_tokens in joint_cases:
        for joint_type in ("fixed", "revolute_unwrapped"):
            fixture_id = case_id + "/" + joint_type
            scene, joint, bodies = make_fixture(parent, child, joint_type)
            try:
                com_context = []
                for role, body, source in bodies:
                    requested_com = source["com"] if context == "exported_com" else IDENTITY
                    com = sapien.Pose(requested_com[:3], requested_com[3:])
                    body.cmass_local_pose = com
                    com_context.append({
                        "role": role, "source_link": source["name"],
                        "input_com_xyz_wxyz": requested_com,
                        "readback_com_xyz_wxyz": pose_values(body.cmass_local_pose).tolist(),
                    })
                measured = []
                for key, attribute in (("parent_pose", "pose_in_parent"), ("child_pose", "pose_in_child")):
                    measurement = measure_pose(
                        fixture_id + "/" + attribute, group, joint, attribute,
                        frame[key], frame_tokens[key],
                    )
                    records.append(measurement)
                    measured.append(measurement)
                # Also expose any cross-effect from assigning the other frame.
                for measurement in measured:
                    final = pose_values(getattr(joint, measurement["attribute"]))
                    measurement["after_both_frames_xyz_wxyz"] = final.tolist()
                    measurement["residuals"]["after_both_frames_minus_readback"] = residual(
                        final, measurement["readback_xyz_wxyz"],
                    )
                fixtures.append({
                    "id": fixture_id, "joint_type": joint.type,
                    "registered_bodies": len(bodies), "com_context": com_context,
                    "mass_properties": [mass_record(*b) for b in bodies],
                })
            finally:
                scene.clear()

    package = Path(sapien.__file__).resolve().parent
    source_paths = [
        "wrapper/scene.py", "include/sapien/math/pose.h", "include/sapien/math/quat.h",
        "include/sapien/math/vec3.h", "include/sapien/math/conversion.h",
        "include/sapien/physx/articulation_link_component.h",
        "include/sapien/physx/rigid_component.h",
    ]
    header_records = [file_record(package / p) for p in source_paths if (package / p).is_file()]
    native_libraries = sorted({
        path
        for directory in (package, package / "libs")
        for pattern in ("libsapien*", "sapien.dll", "libPhysX*", "PhysX*.dll")
        for path in directory.glob(pattern)
        if path.is_file()
    })
    mass_values = [m for f in fixtures for m in f["mass_properties"]]
    return {
        "schema_version": 1,
        "scope": "Native pose assignment/readback diagnosis only; no parity verdict or acceptance thresholds.",
        "setup": {
            "scene_systems": ["sapien.physx.PhysxCpuSystem"],
            "max_bodies_per_scene": 3, "collision_shapes": 0,
            "renderer_created": False, "gpu_simulation": False,
            "physics_steps": 0, "root_or_qpos_assignments": 0,
            "pose_assignments": "Setup only, repeated original input once through explicit set_* method.",
            "quaternion_convention": "wxyz; raw components, no normalization or sign alignment by probe",
            "residual_arithmetic": "float64 subtraction of copied native getter values; JSON round-trip precision",
            "joint_context": "Adjacent exported link records supply COM contexts; both diagnostic joint types tested; no FK reconstruction or joint limits configured.",
        },
        "provenance": {
            "utc": datetime.now(timezone.utc).isoformat(),
            "python_executable": sys.executable, "python_version": sys.version,
            "platform": platform.platform(), "machine": platform.machine(),
            "numpy_version": np.__version__,
            "sapien_version": importlib.metadata.version("sapien"),
            "sapien_direct_url": importlib.metadata.distribution("sapien").read_text("direct_url.json"),
            "sapien_native_module": file_record(sapien.pysapien.__file__),
            "sapien_native_libraries": [file_record(p) for p in native_libraries],
            "available_dependency_source": header_records,
            "checkout_head": git_head(CHECKOUT), "script": file_record(__file__),
            "runtime_parameter_assignment_source": file_record(CHECKOUT / "mani_skill/envs/softbody/bucket.py"),
            "parameters": file_record(args.parameters),
            "export_metadata": {k: v for k, v in document.items() if k not in ("bucket", "robot_parameters")},
            "asset_library_inspection": asset_inspection,
        },
        "robot_parameters": parameters,
        "representative_inputs": REPRESENTATIVE_POSES,
        "summary": summarize(records),
        "mass_inertia_summary": {
            "mass_max_abs_kg": max(abs(m["mass_residual_kg"]) for m in mass_values),
            "inertia_component_max_abs_kg_m2": max(
                abs(v) for m in mass_values for v in m["inertia_residual_kg_m2"]
            ),
        },
        "fixtures": fixtures, "measurements": records,
        "limitations": [
            "No SAPIEN 2 process or reference outcomes loaded; source_capture_sha256 is exported provenance only.",
            "No full robot FK, reset/restore lifecycle, geometry, contacts, MPM, drives, dynamics or two-way coupling exercised.",
            "A nonzero setter/getter residual does not identify the exact internal setter versus getter contribution.",
            "COM-dependent native round trips can contribute to initial FK differences; this probe does not attribute the observed Fill discrepancy or establish a corrective formula.",
            "Available native headers declare the component setters/getters but do not contain their implementations; no specific runtime defect or physics fix is established.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--asset-docs", type=Path, default=DEFAULT_ASSET_DOCS)
    parser.add_argument("--asset-catalog", type=Path, default=DEFAULT_ASSET_CATALOG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Keep all diagnostic output inside the checkout and avoid silently replacing
    # a previous run. Inputs and installed dependencies are read-only.
    output = args.output.resolve()
    if not output.is_relative_to(CHECKOUT):
        parser.error("--output must be inside this checkout")
    if output.exists():
        parser.error("--output already exists; choose a new report path")
    report = run(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "report": str(output), "measurements": len(report["measurements"]),
        "mass_inertia_summary": report["mass_inertia_summary"],
        "assignment_readback_maxima": {
            group: item["residual_maxima"]["readback_minus_pose"]
            for group, item in report["summary"].items()
        },
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

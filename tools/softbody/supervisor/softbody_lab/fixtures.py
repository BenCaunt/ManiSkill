"""Export only initialization and controls to simulation workers."""

from pathlib import Path
import shutil

import numpy as np

from .artifacts import (
    InvalidArtifact, child_file, digest_arrays, digest_json, load_frame,
    read_json, sha256, validate_trace, write_json, initial_numeric_state,
)


def export_fixture(trace: Path, destination: Path) -> Path:
    manifest = validate_trace(trace)
    state = load_frame(child_file(trace, manifest["samples"][0]["path"]))
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination / "initial.npz", **state)
    np.save(destination / "actions.npy", np.asarray(manifest["actions"], dtype=np.float32))
    record = {
        "schema_version": 1, "fixture": manifest["fixture"],
        "fixture_sha256": manifest["fixture_sha256"],
        "steps": manifest["requested_steps"],
        "initial_sha256": sha256(destination / "initial.npz"),
        "actions_sha256": sha256(destination / "actions.npy"),
    }
    if "material_sha256" in manifest["fixture"]:
        shutil.copyfile(child_file(trace, "material.npz"), destination / "material.npz")
        record["material_file_sha256"] = sha256(destination / "material.npz")
    write_json(destination / "fixture.json", record)
    return destination


def load_fixture(root: Path):
    record = read_json(child_file(root, "fixture.json"))
    if record.get("schema_version") != 1:
        raise InvalidArtifact("Unknown fixture schema")
    if record["fixture_sha256"] != digest_json(record["fixture"]):
        raise InvalidArtifact("Fixture checksum mismatch")
    for name in ("initial", "actions"):
        path = child_file(root, "initial.npz" if name == "initial" else "actions.npy")
        if sha256(path) != record[f"{name}_sha256"]:
            raise InvalidArtifact(f"Fixture {name} checksum mismatch")
    state = load_frame(root / "initial.npz")
    if "initial_numeric_sha256" in record["fixture"]:
        digest = digest_arrays(initial_numeric_state(state, record["fixture"]))
        if digest != record["fixture"]["initial_numeric_sha256"]:
            raise InvalidArtifact("Initial values differ from fixture")
    actions_path = root / "actions.npy"
    if actions_path.stat().st_size > 64 * 1024 * 1024:
        raise InvalidArtifact("Action file exceeds size limit")
    actions = np.load(actions_path, allow_pickle=False)
    if actions.ndim != 2 or len(actions) != record["steps"] or not np.isfinite(actions).all():
        raise InvalidArtifact("Invalid fixture actions")
    material = None
    if "material_sha256" in record["fixture"]:
        path = child_file(root, "material.npz")
        if sha256(path) != record.get("material_file_sha256"):
            raise InvalidArtifact("Fixture material file checksum mismatch")
        # Material arrays have their own shapes; reuse the bounded NPZ checks
        # from trace validation by checking size before NumPy loads any array.
        import zipfile
        from .artifacts import MAX_FRAME_BYTES
        if path.stat().st_size > MAX_FRAME_BYTES:
            raise InvalidArtifact("Material file exceeds size limit")
        with zipfile.ZipFile(path) as z:
            members = z.infolist()
            if len(members) > 128 or sum(x.file_size for x in members) > MAX_FRAME_BYTES:
                raise InvalidArtifact("Expanded material exceeds size limit")
            if len({m.filename for m in members}) != len(members):
                raise InvalidArtifact("Duplicate material archive members")
        with np.load(path, allow_pickle=False) as z:
            material = dict(z)
        if digest_arrays(material) != record["fixture"]["material_sha256"]:
            raise InvalidArtifact("Fixture material values differ")
    return record, state, actions, material

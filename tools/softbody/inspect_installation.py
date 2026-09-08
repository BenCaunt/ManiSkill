"""Inspect an explicit soft-body site-packages tree without importing its code."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys
import sysconfig


MANIFEST = "mani_skill/envs/softbody/native-bundle.json"
NATIVE_SOURCE = "tools/softbody/native/"
NATIVE_INSTALLED = "mani_skill/envs/softbody/native/"
MAX_MANIFEST_BYTES = 1024**2
MAX_FILE_BYTES = 256 * 1024**2  # Same per-library bound as wheel_support.py.
MAX_TOTAL_BYTES = 1024**3
SOURCE_FILES = (
    "mani_skill/envs/softbody/__init__.py",
    "warp_maniskill/__init__.py",
    "warp_maniskill/warp/__init__.py",
    "warp_maniskill/mpm/mpm_simulator.py",
    "warp_maniskill/build_lib.py",
)
LICENSE_FILES = (
    "warp_maniskill/LICENSE.md",
    "warp_maniskill/PROVENANCE.json",
    "mani_skill/envs/softbody/NOTICE.md",
    NATIVE_INSTALLED + "vendor/LICENSE",
    NATIVE_INSTALLED + "cooked/vendor/LICENSE",
)
# The bundle records its entire build tree. These two upstream files are not
# installed by setup.py/MANIFEST.in; they are not missing runtime dependencies.
BUILD_ONLY_SOURCES = {
    "mani_skill/agents/robots/_template/template_robot.py":
        "Robot-authoring scaffold outside the discovered Python packages.",
    "warp_maniskill/CHANGELOG.md":
        "Build-tree changelog outside the declared Warp package data.",
}


def relative_parts(value):
    """Accept only canonical, portable relative paths, before filesystem access."""
    if (not isinstance(value, str) or not value or len(value) > 1024
            or "\\" in value or ":" in value
            or any(ord(char) < 32 for char in value)
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError("unsafe relative path: " + repr(value))
    return value.split("/")


def installed_source(value):
    relative_parts(value)
    if value.startswith(NATIVE_SOURCE):
        return NATIVE_INSTALLED + value[len(NATIVE_SOURCE):]
    if value.startswith(("mani_skill/", "warp_maniskill/")):
        return value
    raise ValueError("unexpected source prefix: " + value)


class Reader:
    """Bound reads and reject links/special files, including parent symlinks."""

    def __init__(self, root):
        self.root = root
        self.remaining = MAX_TOTAL_BYTES

    def path(self, relative):
        path = self.root
        for part in relative_parts(relative):
            path = path / part
            if stat.S_ISLNK(path.lstat().st_mode):
                raise ValueError("symlink is not allowed: " + relative)
        return path

    def read(self, relative, limit, collect=False):
        path = self.path(relative)
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("expected an ordinary file: " + relative)
        if info.st_size > limit or info.st_size > self.remaining:
            raise ValueError("file exceeds diagnostic read limit: " + relative)
        # O_NONBLOCK also prevents a concurrently substituted FIFO from hanging.
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        with os.fdopen(os.open(path, flags), "rb") as stream:
            opened = os.fstat(stream.fileno())
            if (not stat.S_ISREG(opened.st_mode)
                    or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino)):
                raise ValueError("file changed during inspection: " + relative)
            digest = hashlib.sha256()
            data = bytearray()
            header = b""
            size = 0
            while True:
                chunk = stream.read(min(1024**2, limit - size + 1, self.remaining + 1))
                if not chunk:
                    break
                size += len(chunk)
                self.remaining -= len(chunk)
                if size > limit or self.remaining < 0:
                    raise ValueError("file exceeds diagnostic read limit: " + relative)
                digest.update(chunk)
                if len(header) < 64:
                    header += chunk[:64 - len(header)]
                if collect:
                    data.extend(chunk)
            if size != opened.st_size:
                raise ValueError("file changed during inspection: " + relative)
        return size, digest.hexdigest(), bytes(data) if collect else header


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(reader, path):
    _, _, data = reader.read(path, MAX_MANIFEST_BYTES, collect=True)
    record = json.loads(data.decode("utf-8"), object_pairs_hook=unique_object,
                        parse_constant=lambda value: (_ for _ in ()).throw(
                            ValueError("invalid JSON constant: " + value)))
    if not isinstance(record, dict):
        raise ValueError("expected a JSON object: " + path)
    return record


def inspect_installation(site_packages, require_native=False):
    root = Path(site_packages).absolute()
    host = dict(python_tag=sys.implementation.cache_tag,
                python_version=platform.python_version(),
                platform=platform.system().lower() + "_" + platform.machine(),
                soabi=sysconfig.get_config_var("SOABI"),
                extension_suffix=sysconfig.get_config_var("EXT_SUFFIX"))
    report = dict(schema_version=1, site_packages=str(root), installation="missing",
                  ok=False, require_native=require_native, inspector=host,
                  native_metadata=None, abi=None, files=[], errors=[], build_only_sources=[],
                  licenses={"files": list(LICENSE_FILES), "provenance": None},
                  limitations=[
                      "Static file integrity only; no code or native libraries are loaded.",
                      "Dependencies, CUDA, loaders, external assets and physics support/parity are not tested.",
                      "Manifest hashes are not signatures; inspect a trusted, quiescent installation.",
                      "Legacy solver research/evaluation restrictions and third-party licenses still apply.",
                  ])
    errors = report["errors"]
    checked = {}
    reader = Reader(root)

    def check(path, kind, expected=None, native=False):
        entry = dict(path=path, kind=kind, expected_sha256=expected)
        try:
            size, actual, header = reader.read(path, MAX_FILE_BYTES)
            entry.update(size=size, sha256=actual, status="present" if expected is None else "ok")
            if expected is not None and actual != expected:
                entry["status"] = "hash-mismatch"
                errors.append("checksum mismatch: " + path)
            if native and (size < 64 or header[:6] != b"\x7fELF\x02\x01"
                           or int.from_bytes(header[16:18], "little") != 3
                           or int.from_bytes(header[18:20], "little") != 62):
                entry["status"] = "invalid-native"
                errors.append("expected a little-endian x86_64 ELF library: " + path)
        except (OSError, ValueError) as exc:
            entry["status"] = "missing" if isinstance(exc, FileNotFoundError) else "invalid"
            errors.append(path + ": " + str(exc))
        checked[path] = entry

    try:
        if not root.is_dir():
            raise ValueError("site-packages root is not a directory")
        # Resolve only the explicitly supplied root; children must never be links.
        root = root.resolve(strict=True)
        reader.root = root
        try:
            manifest_path = reader.path(MANIFEST)
        except FileNotFoundError:
            manifest_path = None
        if manifest_path is None:
            report["installation"] = "source-only"
            report["limitations"].append(
                "No native bundle manifest; source presence does not establish physics support. "
                "Locally compiled Warp libraries are not authenticated by this check.")
            if require_native:
                errors.append("a native bundle is required, but this installation is source-only")
            # Adapter libraries without the required manifest indicate an incomplete bundle.
            with os.scandir(root) as entries:
                for index, entry in enumerate(entries):
                    if index >= 10000:
                        raise ValueError("site-packages exceeds diagnostic directory limit")
                    if entry.name.startswith("sapien303_") and "bridge" in entry.name and entry.name.endswith(".so"):
                        errors.append("native adapter without native-bundle.json: " + entry.name)
        else:
            report["installation"] = "native-bundle"
            record = read_json(reader, MANIFEST)
            report["native_metadata"] = {key: value for key, value in record.items()
                                         if key not in ("files", "source_files")}
            if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
                raise ValueError("unsupported native bundle schema_version")
            tag = record.get("python_tag")
            if record.get("platform") != "linux_x86_64" or not isinstance(tag, str) or not re.fullmatch(r"cpython-[0-9]+", tag):
                raise ValueError("invalid native bundle platform or Python tag")
            files = record.get("files")
            sources = record.get("source_files")
            if not isinstance(files, dict) or not isinstance(sources, dict) or not sources:
                raise ValueError("expected native files and nonempty source_files hash maps")
            suffix = "." + tag + "-x86_64-linux-gnu.so"
            # Free-threaded CPython includes 't' in EXT_SUFFIX, but not cache_tag.
            if "sapien303_actor_bridge." + tag + "t-x86_64-linux-gnu.so" in files:
                suffix = "." + tag + "t-x86_64-linux-gnu.so"
            expected = {"warp.so": "warp_maniskill/warp/bin/warp.so"}
            for adapter in ("actor", "cooked"):
                name = "sapien303_" + adapter + "_bridge" + suffix
                expected[name] = name
            report["abi"] = dict(platform=record["platform"], python_tag=tag,
                                 extension_suffix=suffix,
                                 matches_inspector=(record["platform"] == host["platform"]
                                                    and tag == host["python_tag"]
                                                    and suffix == host["extension_suffix"]))
            # Validate the entire inventory before trusting any manifest-derived path.
            mapped_sources = {}
            for inventory in (files, sources):
                for name, digest in inventory.items():
                    relative_parts(name)
                    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                        raise ValueError("invalid SHA-256 for " + name)
            for name, digest in sources.items():
                installed = installed_source(name)
                if name in BUILD_ONLY_SOURCES:
                    report["build_only_sources"].append(dict(
                        path=name, sha256=digest, reason=BUILD_ONLY_SOURCES[name]))
                    continue
                if installed in mapped_sources or installed in expected.values() or installed == MANIFEST:
                    raise ValueError("duplicate or invalid installed source path: " + installed)
                mapped_sources[installed] = digest
            if set(files) != set(expected):
                errors.append("native file inventory differs; expected: " + ", ".join(sorted(expected)))
            for name, destination in expected.items():
                check(destination, "native", files.get(name), native=True)
            for path, digest in sorted(mapped_sources.items()):
                check(path, "source", digest)
            for path in SOURCE_FILES + LICENSE_FILES:
                if path not in mapped_sources:
                    errors.append("required source/license omitted from manifest: " + path)

        for path in SOURCE_FILES + LICENSE_FILES:
            if path not in checked:
                check(path, "license" if path in LICENSE_FILES else "source")
        provenance = read_json(reader, "warp_maniskill/PROVENANCE.json")
        report["licenses"]["provenance"] = {
            key: value for key, value in provenance.items()
            if key in ("license", "source_url", "source_commit", "creator")
        }
    except (OSError, ValueError, RecursionError) as exc:
        errors.append(str(exc))
    report["files"] = list(checked.values())
    report["ok"] = not errors
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-packages", required=True, type=Path,
                        help="explicit installed site-packages root (never added to sys.path)")
    parser.add_argument("--require-native", action="store_true")
    args = parser.parse_args(argv)
    report = inspect_installation(args.site_packages, args.require_native)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

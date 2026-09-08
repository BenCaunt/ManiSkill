"""Synthetic file checks only: no runtime imports or executable native fixtures."""
import builtins
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import inspect_installation as diagnostic


class InspectionTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[2] / ".softbody-tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.provenance = {"license": "Research/evaluation terms retained", "creator": "Fixture"}

    def write(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def source(self):
        for relative in diagnostic.SOURCE_FILES + diagnostic.LICENSE_FILES:
            self.write(relative, b"raise RuntimeError('installed code must never execute')\n")
        self.write("warp_maniskill/PROVENANCE.json", json.dumps(self.provenance).encode())

    def manifest(self, record):
        return self.write(diagnostic.MANIFEST, json.dumps(record).encode())

    def native(self):
        self.source()
        self.write(diagnostic.NATIVE_INSTALLED + "actor_bridge.cpp", b"noncompiled source")
        header = bytearray(64)
        header[:6] = b"\x7fELF\x02\x01"
        header[16:20] = b"\x03\x00\x3e\x00"
        names = {"warp.so": "warp_maniskill/warp/bin/warp.so"}
        for adapter in ("actor", "cooked"):
            name = "sapien303_" + adapter + "_bridge.cpython-310-x86_64-linux-gnu.so"
            names[name] = name
        for relative in names.values():
            self.write(relative, header)
        sources = {}
        for relative in diagnostic.SOURCE_FILES + diagnostic.LICENSE_FILES + (diagnostic.NATIVE_INSTALLED + "actor_bridge.cpp",):
            name = relative.replace(diagnostic.NATIVE_INSTALLED, diagnostic.NATIVE_SOURCE, 1)
            sources[name] = hashlib.sha256((self.root / relative).read_bytes()).hexdigest()
        record = dict(schema_version=1, platform="linux_x86_64", python_tag="cpython-310",
                      files={name: hashlib.sha256(header).hexdigest() for name in names},
                      source_files=sources, actor_build={"sapien": "3.0.3", "python": "3.10",
                      "files": {"/never/read/build/paths": "metadata only"}},
                      commands=[["never-execute-this-command"]])
        self.manifest(record)
        return record

    def inspect(self, **kwargs):
        return diagnostic.inspect_installation(self.root, **kwargs)

    def cli(self, *args):
        result = subprocess.run([sys.executable, "-I", "-S", diagnostic.__file__,
                                 "--site-packages", str(self.root), *args],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.stderr, "")
        return result.returncode, json.loads(result.stdout)

    def test_source_only_cli_and_require_native(self):
        self.source()
        code, report = self.cli()
        self.assertEqual(code, 0)
        self.assertEqual(report["installation"], "source-only")
        self.assertEqual(report["licenses"]["provenance"], self.provenance)
        self.assertIn("physics support", " ".join(report["limitations"]))
        self.assertEqual(self.cli("--require-native")[0], 1)

    def test_native_mapping_metadata_and_foreign_abi(self):
        record = self.native()
        code, report = self.cli("--require-native")
        self.assertEqual(code, 0, report["errors"])
        self.assertEqual(report["native_metadata"]["actor_build"], record["actor_build"])
        self.assertTrue(any(item["path"] == diagnostic.NATIVE_INSTALLED + "actor_bridge.cpp"
                            and item["status"] == "ok" for item in report["files"]))
        with patch.object(diagnostic.platform, "system", return_value="Darwin"):
            report = self.inspect()
        self.assertTrue(report["ok"])
        self.assertFalse(report["abi"]["matches_inspector"])

    def test_no_runtime_imports(self):
        self.native()
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            self.assertFalse(name.split(".")[0] in {"mani_skill", "warp_maniskill", "warp", "sapien"}
                             or name.startswith("sapien303_"), name)
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded):
            self.assertTrue(self.inspect()["ok"])

    def test_build_only_inventory_does_not_hide_missing_runtime_source(self):
        record = self.native()
        for name in diagnostic.BUILD_ONLY_SOURCES:
            record["source_files"][name] = hashlib.sha256(b"build-tree input").hexdigest()
        self.manifest(record)
        report = self.inspect(require_native=True)
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual({item["path"] for item in report["build_only_sources"]},
                         set(diagnostic.BUILD_ONLY_SOURCES))
        for item in report["build_only_sources"]:
            self.assertEqual(item["sha256"], record["source_files"][item["path"]])
            self.assertTrue(item["reason"])
        (self.root / "warp_maniskill/mpm/mpm_simulator.py").unlink()
        report = self.inspect(require_native=True)
        self.assertFalse(report["ok"])
        self.assertTrue(any(item["path"] == "warp_maniskill/mpm/mpm_simulator.py"
                            and item["status"] == "missing" for item in report["files"]))

    def test_missing_root_empty_install_and_missing_library(self):
        self.assertFalse(self.inspect()["ok"])
        self.assertFalse(diagnostic.inspect_installation(self.root / "absent")["ok"])
        self.native()
        (self.root / "warp_maniskill/warp/bin/warp.so").unlink()
        code, report = self.cli()
        self.assertEqual(code, 1)
        self.assertTrue(any(item["status"] == "missing" for item in report["files"]))

    def test_native_and_source_hash_mismatches(self):
        self.native()
        self.write("warp_maniskill/warp/bin/warp.so", b"corrupt")
        self.write(diagnostic.NATIVE_INSTALLED + "actor_bridge.cpp", b"changed")
        report = self.inspect()
        self.assertFalse(report["ok"])
        self.assertEqual(sum("checksum mismatch" in error for error in report["errors"]), 2)

    def test_bad_elf_even_with_correct_hash(self):
        record = self.native()
        self.write("warp_maniskill/warp/bin/warp.so", b"invalid")
        record["files"]["warp.so"] = hashlib.sha256(b"invalid").hexdigest()
        self.manifest(record)
        self.assertFalse(self.inspect()["ok"])

    def test_missing_inventory_manifest_and_license(self):
        record = self.native()
        del record["files"]["warp.so"]
        self.manifest(record)
        self.assertFalse(self.inspect()["ok"])
        (self.root / diagnostic.MANIFEST).unlink()
        self.assertFalse(self.inspect()["ok"])
        self.native()
        (self.root / "warp_maniskill/LICENSE.md").unlink()
        self.assertFalse(self.inspect()["ok"])

    def test_local_warp_is_not_a_native_bundle(self):
        self.source()
        self.write("warp_maniskill/warp/bin/warp.dylib", b"nonexecuted local build")
        self.assertTrue(self.inspect()["ok"])
        self.assertFalse(self.inspect(require_native=True)["ok"])

    def test_bad_json_and_manifest_types(self):
        self.source()
        for data in (b"{", b"[]", b"null", b'{"x":1,"x":1}', b'{"x":NaN}', b"\xff", b"[" * 1500):
            with self.subTest(data=data[:20]):
                self.write(diagnostic.MANIFEST, data)
                self.assertFalse(self.inspect()["ok"])
        record = self.native()
        for key, value in (("schema_version", True), ("platform", "win32"),
                           ("python_tag", []), ("files", []), ("source_files", {})):
            self.manifest(dict(record, **{key: value}))
            self.assertFalse(self.inspect()["ok"])

    def test_unsafe_paths_before_manifest_derived_reads(self):
        record = self.native()
        for name in ("../outside", "/outside", "warp_maniskill/../outside", "warp_maniskill//bad",
                     "warp_maniskill/./bad", "C:/outside", "warp_maniskill\\bad", "warp_maniskill/\x00bad", "unrelated/file"):
            with self.subTest(name=name):
                self.manifest(dict(record, source_files={name: "0" * 64}))
                report = self.inspect()
                self.assertFalse(report["ok"])
                self.assertEqual(report["files"], [])

    def test_symlinks_and_parent_symlinks(self):
        for relative in (diagnostic.MANIFEST, "warp_maniskill/warp/bin/warp.so"):
            self.native()
            path = self.root / relative
            saved = self.write("saved", path.read_bytes())
            path.unlink()
            path.symlink_to(saved)
            self.assertFalse(self.inspect()["ok"])
            path.unlink()
        self.native()
        directory = self.root / "warp_maniskill/warp/bin"
        directory.rename(self.root / "saved-bin")
        directory.symlink_to(self.root / "saved-bin", target_is_directory=True)
        self.assertFalse(self.inspect()["ok"])

    def test_bounds_and_special_files(self):
        self.native()
        with (self.root / diagnostic.MANIFEST).open("wb") as stream:
            stream.truncate(diagnostic.MAX_MANIFEST_BYTES + 1)
        self.assertFalse(self.inspect()["ok"])
        self.native()
        path = self.root / "warp_maniskill/warp/bin/warp.so"
        with path.open("wb") as stream:
            stream.truncate(diagnostic.MAX_FILE_BYTES + 1)
        self.assertFalse(self.inspect()["ok"])
        if hasattr(os, "mkfifo"):
            path.unlink()
            os.mkfifo(path)
            self.assertFalse(self.inspect()["ok"])


if __name__ == "__main__":
    unittest.main()

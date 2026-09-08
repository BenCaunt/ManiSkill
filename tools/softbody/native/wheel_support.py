"""Package source by default; include only an explicitly pinned native bundle.

Binary wheels use the building interpreter's native platform/ABI tags. This
does not claim manylinux portability or bundle external benchmark assets.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import sysconfig

from setuptools import Distribution
from setuptools.command.build_py import build_py


def binary_directory():
    value = os.environ.get('MANISKILL_SOFTBODY_BINARY_DIR')
    return Path(value).resolve() if value else None


def source_inventory(root):
    result = {}
    for prefix in ('mani_skill', 'warp_maniskill', 'tools/softbody/native'):
        for path in sorted((root/prefix).rglob('*')):
            if not path.is_file() or any(part in ('__pycache__', 'tests') for part in path.parts):
                continue
            if path.suffix not in ('.py', '.cpp', '.cu', '.h', '.json', '.md', '.txt', '.patch') and path.name != 'LICENSE':
                continue
            if path.is_symlink():
                raise ValueError('Native source identity cannot follow symlinks')
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def bundle_files(directory, source_root=None):
    """Reject stale, incomplete and incompatible bundles before copying bytes."""
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('Native bundles are currently verified only on Linux x86_64')
    manifest = directory/'bundle.json'
    if manifest.is_symlink() or not manifest.is_file() or manifest.stat().st_size > 1024**2:
        raise ValueError('Expected an ordinary bounded native bundle manifest')
    record = json.loads(manifest.read_text())
    source_root = Path(__file__).resolve().parents[3] if source_root is None else source_root
    if record.get('source_files') != source_inventory(source_root):
        raise ValueError('Native bundle was built from different source files')
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    expected = {'warp.so': 'warp_maniskill/warp/bin/warp.so',
        'sapien303_actor_bridge'+suffix: 'sapien303_actor_bridge'+suffix,
        'sapien303_cooked_bridge'+suffix: 'sapien303_cooked_bridge'+suffix}
    if (record.get('schema_version') != 1 or record.get('platform') != 'linux_x86_64'
            or record.get('python_tag') != sys.implementation.cache_tag
            or set(record.get('files', {})) != set(expected)):
        raise ValueError('Native bundle platform, Python ABI or file inventory differs')
    result = []
    for name, destination in expected.items():
        source = directory/name
        if source.is_symlink() or not source.is_file() or not 0 < source.stat().st_size <= 256*1024**2:
            raise ValueError('Invalid native bundle file: '+name)
        value = source.read_bytes()
        if hashlib.sha256(value).hexdigest() != record['files'][name]:
            raise ValueError('Native bundle checksum mismatch: '+name)
        if (len(value) < 64 or value[:6] != b'\x7fELF\x02\x01'
                or int.from_bytes(value[16:18], 'little') != 3
                or int.from_bytes(value[18:20], 'little') != 62):
            raise ValueError('Expected a little-endian x86_64 ELF library: '+name)
        result.append((source, destination))
    result.append((manifest, 'mani_skill/envs/softbody/native-bundle.json'))
    return result


class SoftbodyDistribution(Distribution):
    def has_ext_modules(self):
        return binary_directory() is not None or super().has_ext_modules()


class BuildSoftbodyPython(build_py):
    def run(self):
        directory = binary_directory()
        files = bundle_files(directory) if directory is not None else []
        if directory is None:
            build_root=Path(self.build_lib)
            stale=list(build_root.glob('sapien303_*bridge*.so'))
            stale+=list((build_root/'warp_maniskill/warp/bin').glob('*'))
            if stale or (build_root/'mani_skill/envs/softbody/native-bundle.json').exists():
                raise ValueError('Stale native bundle in the build directory; use a clean build directory')
        super().run()
        self._softbody_outputs = []
        for source, relative in files:
            destination = Path(self.build_lib)/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            self._softbody_outputs.append(str(destination))

    def get_outputs(self, include_bytecode=1):
        return super().get_outputs(include_bytecode) + getattr(self, '_softbody_outputs', [])

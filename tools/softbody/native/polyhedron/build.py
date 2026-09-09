"""Build the offline explicit-polyhedron cooker against pinned SAPIEN 3.0.3."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(eigen, output, physx=None):
    if version('sapien') != '3.0.3':
        raise RuntimeError('Expected SAPIEN 3.0.3')
    import sapien
    import torch
    source = Path(__file__).resolve().parent
    config = json.loads((source / 'source.json').read_text())
    for name, digest in config['files'].items():
        if sha(source / name) != digest:
            raise RuntimeError('Cooking source checksum mismatch: ' + name)
    macros = (eigen / 'Eigen/src/Core/util/Macros.h').read_text()
    for name, value in [('WORLD', 3), ('MAJOR', 4), ('MINOR', 0)]:
        if f'#define EIGEN_{name}_VERSION {value}' not in macros:
            raise RuntimeError('Expected Eigen 3.4.0')
    package = Path(sapien.__file__).parent
    include = package / 'include'
    for name, digest in config['sdk_headers'].items():
        if sha(include / name) != digest:
            raise RuntimeError('Installed cooking SDK differs: ' + name)
    system = platform.system()
    static_libraries = []
    if system == 'Darwin':
        native = package / 'libs/libsapien.dylib'
        flags = ['-undefined', 'dynamic_lookup']
    elif system == 'Linux':
        native = package.parent / 'sapien.libs/libsapien.so'
        flags = ['-DSAPIEN_CUDA', '-D_GLIBCXX_USE_CXX11_ABI=1', '-DPX_PHYSX_STATIC_LIB']
        if physx is None:
            raise RuntimeError('Linux cooking requires --physx with the pinned SAPIEN PhysX release')
        for name, digest in config['linux_static_libraries'].items():
            path = physx.resolve() / name
            if sha(path) != digest:
                raise RuntimeError('PhysX static cooking dependency differs: ' + name)
            static_libraries.append(str(path))
        static_libraries = ['-Wl,--start-group', *static_libraries, '-Wl,--end-group']
    else:
        raise RuntimeError('Unsupported cooking platform')
    if sha(native) != config['native_libraries'][system]:
        raise RuntimeError('Native SAPIEN library differs from the pinned wheel')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source / 'PHYSX-NOTICE.txt', output / 'PHYSX-NOTICE.txt')
    target = output / ('sapien303_polyhedron_bridge' + sysconfig.get_config_var('EXT_SUFFIX'))
    roots = [include, include / 'physx/include', eigen.resolve(),
             Path(torch.__file__).parent / 'include', Path(sysconfig.get_paths()['include'])]
    if system == 'Linux':
        roots.append(Path('/usr/local/cuda/include'))
    command = ['c++', '-std=c++20', '-shared', '-fPIC', '-O2', '-DNDEBUG', *flags,
               *[f'-I{p}' for p in roots], str(source / 'bridge.cpp'), *static_libraries, str(native),
               '-Wl,-rpath,' + str(native.parent), '-o', str(target)]
    (output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (output / 'build.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    if result.returncode:
        raise RuntimeError('Native cooking build failed; see ' + str(output / 'build.log'))
    # Shared-library linking can succeed with an unavailable native symbol.
    # Verify the actual import in a bounded child before recording a usable build.
    with (output / 'import.log').open('w') as log:
        result = subprocess.run([sys.executable, '-c',
            'import sys; sys.path.insert(0,sys.argv[1]); import sapien; import sapien303_polyhedron_bridge',
            str(output)], stdout=log, stderr=subprocess.STDOUT, timeout=20)
    if result.returncode:
        raise RuntimeError('Cooking extension import failed; see ' + str(output / 'import.log'))
    report = dict(sapien=version('sapien'), torch=version('torch'), python=sys.version,
                  platform=platform.platform(), source=config, command=command,
                  extension=target.name, extension_sha256=sha(target),
                  native_library_sha256=sha(native),
                  scope='Explicit convex cooking only; GPU simulation is verified separately')
    (output / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eigen', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physx', type=Path, help='Pinned SAPIEN PhysX static release root (Linux)')
    args = parser.parse_args()
    print(build(args.eigen, args.output, args.physx))

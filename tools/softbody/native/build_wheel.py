"""Build a complete Linux soft-body wheel using installed, offline toolchains.

Run from a writable source checkout with NumPy, Torch, SAPIEN 3.0.3, setuptools,
CUDA 11.8, a C++20 compiler and the pinned Eigen headers already available.
The resulting wheel contains native code; benchmark assets remain external.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig

from wheel_support import source_inventory, bundle_files


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eigen',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if platform.system()!='Linux' or platform.machine()!='x86_64':
        raise RuntimeError('Complete native wheels are currently supported on Linux x86_64')
    root=Path(__file__).resolve().parents[3]
    if not (root/'setup.py').is_file() or not (root/'warp_maniskill/build_lib.py').is_file():
        raise RuntimeError('Build the complete wheel from its source distribution or checkout')
    out=args.output.resolve()
    if out.is_relative_to(root):
        raise ValueError('Use a new output directory outside the source checkout')
    out.mkdir(parents=True,exist_ok=False)
    commands=[
        [sys.executable,str(root/'warp_maniskill/build_lib.py'),'--cuda_path','/usr/local/cuda'],
        [sys.executable,str(root/'tools/softbody/native/build.py'),'--eigen',str(args.eigen.resolve()),'--output',str(out/'actors')],
        [sys.executable,str(root/'tools/softbody/native/cooked/build.py'),'--eigen',str(args.eigen.resolve()),'--output',str(out/'cooked')],
    ]
    for index,command in enumerate(commands):
        with (out/f'build-{index}.log').open('w') as log:
            subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    bundle=out/'bundle';bundle.mkdir()
    suffix=sysconfig.get_config_var('EXT_SUFFIX')
    inputs=[root/'warp_maniskill/warp/bin/warp.so',out/'actors'/('sapien303_actor_bridge'+suffix),
            out/'cooked'/('sapien303_cooked_bridge'+suffix)]
    for source in inputs:shutil.copyfile(source,bundle/source.name)
    record=dict(schema_version=1,platform='linux_x86_64',python_tag=sys.implementation.cache_tag,
        files={p.name:sha(p) for p in inputs},source_files=source_inventory(root),
        commands=commands,actor_build=json.loads((out/'actors/build.json').read_text()),
        cooked_build=json.loads((out/'cooked/build.json').read_text()),
        scope='Experimental Linux native runtime; external assets and physical acceptance are separate.')
    (bundle/'bundle.json').write_text(json.dumps(record,indent=2)+'\n')
    bundle_files(bundle,root)
    environment=dict(os.environ,MANISKILL_SOFTBODY_BINARY_DIR=str(bundle),PYTHONDONTWRITEBYTECODE='1')
    wheel_dir=out/'dist';wheel_dir.mkdir()
    command=[sys.executable,'-c','from setuptools.build_meta import build_wheel; import sys; print(build_wheel(sys.argv[1]))',str(wheel_dir)]
    with (out/'wheel.log').open('w') as log:
        subprocess.run(command,cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    wheels=list(wheel_dir.glob('*.whl'))
    if len(wheels)!=1 or wheels[0].name.endswith('-any.whl'):
        raise RuntimeError('Expected one platform-specific native wheel')
    (out/'result.json').write_text(json.dumps(dict(wheel=wheels[0].name,sha256=sha(wheels[0]),
        bundle_sha256=sha(bundle/'bundle.json')),indent=2)+'\n')
    print(wheels[0])


if __name__=='__main__':main()

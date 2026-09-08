"""Build the pinned Linux SAPIEN actor adapter without downloading dependencies."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eigen',type=Path,required=True,help='Eigen 3.4.0 source/include root')
    parser.add_argument('--output',type=Path,required=True,help='New build directory to add to PYTHONPATH')
    args=parser.parse_args()
    if version('sapien')!='3.0.3' or platform.system()!='Linux':
        raise RuntimeError('This adapter is verified only for Linux SAPIEN 3.0.3')
    import sapien
    import torch
    eigen=args.eigen.resolve()
    macros=(eigen/'Eigen/src/Core/util/Macros.h').read_text()
    for name,value in [('WORLD',3),('MAJOR',4),('MINOR',0)]:
        if f'#define EIGEN_{name}_VERSION {value}' not in macros:
            raise RuntimeError('Expected the Eigen 3.4.0 headers pinned by SAPIEN')
    source=Path(__file__).resolve().parent
    provenance=json.loads((source/'vendor/provenance.json').read_text())
    for record in provenance['files']:
        if hashlib.sha256((source/'vendor'/record['path']).read_bytes()).hexdigest()!=record['sha256']:
            raise RuntimeError('Vendored interoperability header/license hash mismatch')
    package=Path(sapien.__file__).parent
    native=package.parent/'sapien.libs/libsapien.so'
    if not native.is_file():raise RuntimeError('Expected the audited SAPIEN 3.0.3 Linux wheel library')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    target=output/('sapien303_actor_bridge'+sysconfig.get_config_var('EXT_SUFFIX'))
    include=package/'include'
    roots=[include,include/'physx/include',eigen,Path(torch.__file__).parent/'include',
           Path(sysconfig.get_paths()['include']),Path('/usr/local/cuda/include')]
    command=['c++','-std=c++20','-shared','-fPIC','-O2','-DNDEBUG','-DSAPIEN_CUDA',
             '-D_GLIBCXX_USE_CXX11_ABI=1',*[f'-I{p}' for p in roots],str(source/'actor_bridge.cpp'),
             str(native),'-L/usr/local/cuda/lib64','-lcudart',f'-Wl,-rpath,{native.parent}','-o',str(target)]
    (output/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    subprocess.run(command,check=True)
    manifest=dict(sapien=version('sapien'),torch=version('torch'),python=sys.version,
                  command=command,eigen='3.4.0',scope='Native actor adapter only; run GPU verification before use',
                  files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [target,native,source/'actor_bridge.cpp']})
    (output/'build.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(target)


if __name__=='__main__':main()

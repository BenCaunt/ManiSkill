"""Experimental ABI adapter for the additive SAPIEN 3.0.3 mesh factory.

This does not rebuild the whole wheel. The SDK overlay adds one static method;
no fields, virtual slots or existing declarations change. Full-library source
integration is supplied separately and remains the preferred upstream route.
Only the pinned wheel/header combination is accepted.
"""
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

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--eigen',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if version('sapien')!='3.0.3':raise RuntimeError('Expected SAPIEN 3.0.3')
    import sapien
    import torch
    source=Path(__file__).resolve().parent
    config=json.loads((source/'source.json').read_text())
    for name,digest in config['files'].items():
        if sha(source/name)!=digest:raise RuntimeError('Source hash mismatch: '+name)
    include=Path(sapien.__file__).parent/'include'
    original=include/'sapien/physx/mesh.h'
    if sha(original)!=config['original_header_sha256']:
        raise RuntimeError('Installed SAPIEN mesh declaration does not match audited source')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    overlay=out/'include'
    shutil.copytree(include/'sapien',overlay/'sapien')
    shutil.copy2(source/'mesh.h',overlay/'sapien/physx/mesh.h')
    # Verify the whole overlay has only the intended one-file change.
    changed=[str(f.relative_to(overlay)) for f in overlay.rglob('*') if f.is_file()
             and sha(f)!=sha(include/f.relative_to(overlay))]
    if changed!=['sapien/physx/mesh.h']:raise RuntimeError('Unexpected SDK overlay changes')
    package=Path(sapien.__file__).parent
    if platform.system()=='Darwin':
        native=package/'libs/libsapien.dylib';flags=['-undefined','dynamic_lookup']
    elif platform.system()=='Linux':
        native=package.parent/'sapien.libs/libsapien.so';flags=['-DSAPIEN_CUDA','-D_GLIBCXX_USE_CXX11_ABI=1']
    else:raise RuntimeError('Unsupported diagnostic platform')
    if sha(native)!=config['native_libraries'][platform.system()]:
        raise RuntimeError('Native SAPIEN library differs from the pinned wheel')
    target=out/('sapien303_cooked_bridge'+sysconfig.get_config_var('EXT_SUFFIX'))
    roots=[overlay,include,include/'physx/include',args.eigen.resolve(),
           Path(torch.__file__).parent/'include',Path(sysconfig.get_paths()['include'])]
    if platform.system()=='Linux':roots.append(Path('/usr/local/cuda/include'))
    rpath = '$ORIGIN/sapien.libs' if platform.system() == 'Linux' else str(native.parent)
    command=['c++','-std=c++20','-shared','-fPIC','-O2','-DNDEBUG',*flags,
             *[f'-I{r}' for r in roots],str(source/'bridge.cpp'),str(source/'convex_mesh_cooked.cpp'),
             str(native),f'-Wl,-rpath,{rpath}','-o',str(target)]
    (out/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    subprocess.run(command,check=True,timeout=120)
    report=dict(platform=platform.platform(),sapien=version('sapien'),python=sys.version,
                factory_mode='Additive static method compiled into experimental adapter; original wheel unchanged',
                source=config,files={str(f):sha(f) for f in [target,native,original]},command=command)
    (out/'build.json').write_text(json.dumps(report,indent=2)+'\n');print(target)

if __name__=='__main__':main()

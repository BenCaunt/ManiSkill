"""Stage the pinned cooked-mesh adapter; never load a worker ELF on the host."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

FILENAME='sapien303_cooked_bridge.cpython-310-x86_64-linux-gnu.so'
LIBRARY_SHA256='57b9dbf776bd216a2a71c86c34fadeab12469cc9067f6bf2338234499d80f097'


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest(source):
    return hashlib.sha256(json.dumps(source,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_spec(spec):
    if not isinstance(spec,dict) or set(spec)!={'build','sha256','source_config_sha256'}:
        raise ValueError('Cooked extension requires build, binary and source checksums')
    if not isinstance(spec['build'],str) or not re.fullmatch('[A-Za-z0-9_-]+',spec['build']):
        raise ValueError('Invalid cooked extension build identifier')
    for key in ('sha256','source_config_sha256'):
        if not isinstance(spec[key],str) or not re.fullmatch('[a-f0-9]{64}',spec[key]):
            raise ValueError('Invalid cooked extension checksum')


def validate_build(spec,manifest):
    validate_spec(spec)
    if manifest.get('sapien')!='3.0.3' or not manifest.get('python','').startswith('3.10.'):
        raise ValueError('Cooked extension SAPIEN/Python ABI differs')
    source=manifest['source']
    if source_digest(source)!=spec['source_config_sha256']:
        raise ValueError('Cooked extension source configuration differs')
    if source['native_libraries']['Linux']!=LIBRARY_SHA256:
        raise ValueError('Cooked extension wheel differs')
    for name,expected in ((FILENAME,spec['sha256']),('libsapien.so',LIBRARY_SHA256),('mesh.h',source['original_header_sha256'])):
        if [v for k,v in manifest['files'].items() if Path(k).name==name]!=[expected]:
            raise ValueError('Cooked extension build provenance differs: '+name)


def stage(spec,worker_root,destination):
    validate_spec(spec)
    root=Path(worker_root).absolute();build=root/'records'/spec['build']/'build'
    binary=build/FILENAME;manifest_path=build/'build.json'
    for path,limit in [(binary,64*1024**2),(manifest_path,2*1024**2)]:
        if (path.is_symlink() or any(p.is_symlink() for p in path.parents if p!=root.parent)
                or not path.is_file() or not 0<path.stat().st_size<=limit):
            raise ValueError('Invalid cooked extension file or symlink')
    if digest(binary)!=spec['sha256']: raise ValueError('Cooked binary checksum mismatch')
    manifest=json.loads(manifest_path.read_text());validate_build(spec,manifest)
    out=Path(destination);out.mkdir(parents=True,exist_ok=False)
    for path in (binary,manifest_path): shutil.copyfile(path,out/path.name)
    if digest(out/FILENAME)!=spec['sha256']: raise ValueError('Cooked binary changed during staging')
    record=dict(**spec,module='sapien303_cooked_bridge',filename=FILENAME,
        build_manifest_sha256=digest(out/'build.json'),sapien_library_sha256=LIBRARY_SHA256)
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def pack_path(spec,worker_root,expected_sha256):
    validate_spec(spec)
    if not isinstance(expected_sha256,str) or not re.fullmatch('[a-f0-9]{64}',expected_sha256):
        raise ValueError('Cooked pack requires a pinned manifest checksum')
    root=Path(worker_root).absolute();path=root/'probe-inputs'/spec['build']/'pack'
    manifest=path/'pack.json'
    if (manifest.is_symlink() or any(p.is_symlink() for p in manifest.parents if p!=root.parent)
            or not manifest.is_file() or not 0<manifest.stat().st_size<=1024**2
            or digest(manifest)!=expected_sha256):
        raise ValueError('Cooked pack manifest differs or is not an ordinary file')
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('request','worker_root','output'): p.add_argument(name,type=Path)
    a=p.parse_args();print(json.dumps(stage(json.loads(a.request.read_text())['native_cooked_extension'],a.worker_root,a.output)))

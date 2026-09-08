"""Inspect wheel bytes independently of the package's setup/build code."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile


def expected_runtime_files(source_files):
    expected={}
    packages={str(PurePosixPath(n).parent) for n in source_files if n.endswith('/__init__.py')}
    for name,digest in source_files.items():
        p=PurePosixPath(name)
        if '__pycache__' in p.parts or 'tests' in p.parts:
            continue
        if name.startswith('mani_skill/') and (name.endswith('.py') and (
                str(p.parent) in packages or name.startswith(('mani_skill/envs/','mani_skill/utils/')))
                                               or name.endswith('softbody/NOTICE.md')):
            expected[name]=digest
        elif name.startswith('warp_maniskill/') and (name.endswith('.py')
                or name in ['warp_maniskill/LICENSE.md','warp_maniskill/PROVENANCE.json','warp_maniskill/README.md','warp_maniskill/VERSION.md']
                or name.startswith('warp_maniskill/licenses/')
                or '/warp/native/' in name and p.suffix in ('.h','.cpp','.cu')):
            expected[name]=digest
        elif name.startswith('tools/softbody/native/'):
            expected[name.replace('tools/softbody/native/','mani_skill/envs/softbody/native/',1)]=digest
    return expected


def verify(wheel,source_files,*,binary=False):
    wheel=Path(wheel);failures=[]
    with zipfile.ZipFile(wheel) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or sum(i.file_size for i in archive.infolist())>1024**3:
            raise ValueError('Invalid wheel file inventory or size')
        for name in names:
            p=PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or str(p)!=name:
                raise ValueError('Invalid wheel member path')
        wheels=[n for n in names if n.endswith('.dist-info/WHEEL')]
        metas=[n for n in names if n.endswith('.dist-info/METADATA')]
        if len(wheels)!=1 or len(metas)!=1:
            raise ValueError('Missing unambiguous wheel metadata')
        metadata=archive.read(wheels[0]).decode();package_metadata=archive.read(metas[0]).decode()
        required=expected_runtime_files(source_files)
        for name,digest in required.items():
            if name not in names or hashlib.sha256(archive.read(name)).hexdigest()!=digest:
                failures.append('Missing or changed runtime/source/license file: '+name)
        cache=[n for n in names if '__pycache__' in PurePosixPath(n).parts or n.endswith(('.pyc','.pyo','.o','.obj'))]
        if cache:failures.append('Build/cache files leaked into wheel')
        libraries=[n for n in names if n.endswith(('.so','.dylib','.dll'))]
        bundle=None
        if not binary:
            if libraries or 'Root-Is-Purelib: true' not in metadata or 'Tag: py3-none-any' not in metadata or not wheel.name.endswith('-py3-none-any.whl'):
                failures.append('Source wheel contains binaries or has invalid compatibility tags')
        else:
            name='mani_skill/envs/softbody/native-bundle.json'
            if name not in names:raise ValueError('Binary wheel lacks its build manifest')
            bundle=json.loads(archive.read(name));python=bundle['python_tag'].replace('cpython-','cp')
            tag=f'{python}-{python}-linux_x86_64'
            if 'Root-Is-Purelib: false' not in metadata or 'Tag: '+tag not in metadata or not wheel.name.endswith('-'+tag+'.whl'):
                failures.append('Native wheel has incorrect platform/ABI tags')
            if not re.search(r'^Requires-Dist: sapien\s*==\s*3\.0\.3\s*$',package_metadata,re.M):
                failures.append('Native wheel must pin the SAPIEN ABI for every installation')
            expected={('warp_maniskill/warp/bin/'+n if n=='warp.so' else n):h for n,h in bundle['files'].items()}
            if set(libraries)!=set(expected) or len(expected)!=3:
                failures.append('Native wheel binary inventory differs')
            for name,digest in expected.items():
                if name not in names or hashlib.sha256(archive.read(name)).hexdigest()!=digest:
                    failures.append('Native library checksum mismatch: '+name);continue
                data=archive.read(name)
                if (len(data)<64 or data[:6]!=b'\x7fELF\x02\x01' or int.from_bytes(data[16:18],'little')!=3
                        or int.from_bytes(data[18:20],'little')!=62):
                    failures.append('Native library is not an x86_64 ELF shared object: '+name)
            expected_bundle={n:h for n,h in source_files.items()
                if n.startswith(('mani_skill/','warp_maniskill/','tools/softbody/native/'))
                and not {'__pycache__','tests'}.intersection(PurePosixPath(n).parts)
                and (PurePosixPath(n).suffix in ('.py','.cpp','.cu','.h','.json','.md','.txt','.patch')
                     or PurePosixPath(n).name=='LICENSE')}
            if bundle.get('source_files')!=expected_bundle:
                failures.append('Native bundle source differs from frozen candidate input')
        return dict(passed=not failures,failures=failures,wheel=wheel.name,
            sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),file_count=len(names),
            source_files_checked=len(required),native_libraries=libraries,
            excluded_python_scaffolds=[n for n in source_files if n.startswith('mani_skill/') and n.endswith('.py') and n not in required],
            scope='Wheel contents, compatibility metadata and source identity; separate installed physics validation required.')

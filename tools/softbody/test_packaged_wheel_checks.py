import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from verification.wheel_checks import verify


@pytest.fixture
def source_wheel(tmp_path):
    files={'mani_skill/__init__.py':b'', 'mani_skill/module.py':b'x=1',
           'warp_maniskill/__init__.py':b'', 'warp_maniskill/LICENSE.md':b'Complete original license',
           'mani_skill/utils/namespace/module.py':b'x=2'}
    source={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    files['mani_skill-3.0.1.dist-info/WHEEL']=b'Root-Is-Purelib: true\nTag: py3-none-any\n'
    files['mani_skill-3.0.1.dist-info/METADATA']=b'Name: mani_skill\n'
    return tmp_path,files,source


def write(path,files):
    with zipfile.ZipFile(path,'w') as z:
        for n,b in files.items():z.writestr(n,b)
    return path


def test_source_wheel_checks_code_namespace_modules_and_license(source_wheel):
    root,files,source=source_wheel
    r=verify(write(root/'mani_skill-3.0.1-py3-none-any.whl',files),source)
    assert r['passed'] and r['source_files_checked']==5


@pytest.mark.parametrize('name',['mani_skill/module.py','warp_maniskill/LICENSE.md','mani_skill/utils/namespace/module.py'])
def test_missing_required_files_fail(source_wheel,name):
    root,files,source=source_wheel;del files[name]
    assert not verify(write(root/'mani_skill-3.0.1-py3-none-any.whl',files),source)['passed']


@pytest.mark.parametrize('name',['accidental.so','cache.pyc','native.o'])
def test_accidental_binary_or_build_cache_fails(source_wheel,name):
    root,files,source=source_wheel;files[name]=b'not executable'
    assert not verify(write(root/'mani_skill-3.0.1-py3-none-any.whl',files),source)['passed']


@pytest.mark.parametrize('name',['../escape','/absolute','a\\b'])
def test_invalid_member_path_is_rejected(source_wheel,name):
    root,files,source=source_wheel;files[name]=b'bad'
    with pytest.raises(ValueError,match='member path'):
        verify(write(root/'mani_skill-3.0.1-py3-none-any.whl',files),source)


@pytest.fixture
def native_wheel(source_wheel):
    root,files,source=source_wheel
    header=bytearray(64);header[:6]=b'\x7fELF\x02\x01';header[16:18]=(3).to_bytes(2,'little');header[18:20]=(62).to_bytes(2,'little')
    binaries={n:hashlib.sha256(header).hexdigest() for n in ['warp.so','sapien303_actor_bridge.cpython-310-x86_64-linux-gnu.so','sapien303_cooked_bridge.cpython-310-x86_64-linux-gnu.so']}
    for n in binaries:files['warp_maniskill/warp/bin/'+n if n=='warp.so' else n]=bytes(header)
    manifest=dict(schema_version=1,python_tag='cpython-310',files=binaries,source_files=source)
    files['mani_skill/envs/softbody/native-bundle.json']=json.dumps(manifest).encode()
    files['mani_skill-3.0.1.dist-info/WHEEL']=b'Root-Is-Purelib: false\nTag: cp310-cp310-linux_x86_64\n'
    files['mani_skill-3.0.1.dist-info/METADATA']=b'Name: mani_skill\nRequires-Dist: sapien==3.0.3\n'
    return root,files,source


def test_native_wheel_requires_matching_platform_binary_and_abi(native_wheel):
    root,files,source=native_wheel
    assert verify(write(root/'mani_skill-3.0.1-cp310-cp310-linux_x86_64.whl',files),source,binary=True)['passed']


@pytest.mark.parametrize('fault',['optional-only','source-truncated','tag','binary'])
def test_native_wheel_faults(native_wheel,fault):
    root,files,source=native_wheel
    if fault=='optional-only':files['mani_skill-3.0.1.dist-info/METADATA']=b'Requires-Dist: sapien==3.0.3; extra == "softbody"\n'
    elif fault=='source-truncated':
        name='mani_skill/envs/softbody/native-bundle.json';m=json.loads(files[name]);m['source_files']={};files[name]=json.dumps(m).encode()
    elif fault=='tag':files['mani_skill-3.0.1.dist-info/WHEEL']=b'Root-Is-Purelib: true\nTag: py3-none-any\n'
    else:files['warp_maniskill/warp/bin/warp.so']+=b'corruption'
    assert not verify(write(root/'mani_skill-3.0.1-cp310-cp310-linux_x86_64.whl',files),source,binary=True)['passed']

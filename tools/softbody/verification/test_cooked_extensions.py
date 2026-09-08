import copy
import json

import pytest

from tools.softbody.verification import cooked_extensions as native


def fixture(root):
    build=root/'records/cooked-v1/build';build.mkdir(parents=True)
    binary=build/native.FILENAME;binary.write_bytes(b'controlled test bytes, not executable')
    source=dict(files={'bridge.cpp':'a'*64},original_header_sha256='b'*64,
                native_libraries={'Linux':native.LIBRARY_SHA256})
    spec=dict(build='cooked-v1',sha256=native.digest(binary),source_config_sha256=native.source_digest(source))
    manifest=dict(sapien='3.0.3',python='3.10.12',source=source,
        files={native.FILENAME:spec['sha256'],'libsapien.so':native.LIBRARY_SHA256,'mesh.h':'b'*64})
    (build/'build.json').write_text(json.dumps(manifest))
    return spec,build,manifest


def test_stage_exact_binary_and_full_source_provenance(tmp_path):
    spec,build,_=fixture(tmp_path);(build/'unrelated').write_text('not staged')
    out=tmp_path/'staged';record=native.stage(spec,tmp_path,out)
    assert {p.name for p in out.iterdir()}=={native.FILENAME,'build.json','record.json'}
    assert native.digest(out/native.FILENAME)==spec['sha256']==record['sha256']
    assert record['source_config_sha256']==spec['source_config_sha256']


@pytest.mark.parametrize('kind',['binary','source','header','library','python','sapien','duplicate','symlink'])
def test_changed_build_fails_before_staging(tmp_path,kind):
    spec,build,manifest=fixture(tmp_path)
    if kind=='binary':(build/native.FILENAME).write_bytes(b'changed')
    elif kind=='source':manifest['source']['files']['bridge.cpp']='c'*64
    elif kind=='header':manifest['files']['mesh.h']='c'*64
    elif kind=='library':manifest['files']['libsapien.so']='c'*64
    elif kind=='python':manifest['python']='3.12.10'
    elif kind=='sapien':manifest['sapien']='3.0.4'
    elif kind=='duplicate':manifest['files']['other/mesh.h']='b'*64
    else:
        p=build/native.FILENAME;p.rename(build/'elsewhere');p.symlink_to(build/'elsewhere')
    (build/'build.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):native.stage(spec,tmp_path,tmp_path/'staged')
    assert not (tmp_path/'staged').exists()


@pytest.mark.parametrize('change',[{'build':'../escape'},{'build':'/absolute'},{'sha256':'bad'},
                                  {'source_config_sha256':None},{'extra':True}])
def test_bad_spec_rejected(tmp_path,change):
    spec,_,_=fixture(tmp_path);spec.update(change)
    with pytest.raises(ValueError):native.stage(spec,tmp_path,tmp_path/'staged')

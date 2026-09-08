"""Packaging fault controls; synthetic ELF bytes are never executed."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import sysconfig

import pytest

_spec=importlib.util.spec_from_file_location('softbody_wheel_support',Path(__file__).parent/'native/wheel_support.py')
support=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(support)


@pytest.fixture
def bundle(tmp_path,monkeypatch):
    monkeypatch.setattr(support.platform,'system',lambda:'Linux')
    monkeypatch.setattr(support.platform,'machine',lambda:'x86_64')
    source=tmp_path/'source';(source/'mani_skill').mkdir(parents=True)
    (source/'mani_skill/runtime.py').write_text('version=1\n')
    directory=tmp_path/'bundle';directory.mkdir()
    suffix=sysconfig.get_config_var('EXT_SUFFIX')
    names=['warp.so','sapien303_actor_bridge'+suffix,'sapien303_cooked_bridge'+suffix]
    header=bytearray(64);header[:6]=b'\x7fELF\x02\x01';header[16:18]=(3).to_bytes(2,'little');header[18:20]=(62).to_bytes(2,'little')
    for n in names:(directory/n).write_bytes(header)
    record=dict(schema_version=1,platform='linux_x86_64',python_tag=sys.implementation.cache_tag,
        source_files=support.source_inventory(source),files={n:hashlib.sha256(header).hexdigest() for n in names})
    (directory/'bundle.json').write_text(json.dumps(record))
    return directory,source,record


def test_bundle_is_exactly_three_platform_libraries_and_a_manifest(bundle):
    directory,source,_=bundle
    rows=support.bundle_files(directory,source)
    assert len(rows)==4 and rows[0][1]=='warp_maniskill/warp/bin/warp.so'


@pytest.mark.parametrize('field,value',[('schema_version',2),('platform','linux_aarch64'),('python_tag','cpython-999')])
def test_reject_wrong_platform_or_abi(bundle,field,value):
    directory,source,record=bundle;record[field]=value
    (directory/'bundle.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='platform, Python ABI'):
        support.bundle_files(directory,source)


def test_changed_source_cannot_reuse_old_native_bundle(bundle):
    directory,source,_=bundle;(source/'mani_skill/runtime.py').write_text('version=2\n')
    with pytest.raises(ValueError,match='different source'):
        support.bundle_files(directory,source)


@pytest.mark.parametrize('mode',['missing','extra','checksum','symlink','architecture','executable'])
def test_reject_invalid_native_payload(bundle,mode):
    directory,source,record=bundle;path=directory/'warp.so'
    if mode=='missing':path.unlink()
    elif mode=='extra':record['files']['extra.so']='a'*64
    elif mode=='checksum':path.write_bytes(path.read_bytes()+b'x')
    elif mode=='symlink':
        target=directory.parent/'elsewhere.so';path.rename(target);path.symlink_to(target)
    else:
        value=bytearray(path.read_bytes());value[18 if mode=='architecture' else 16]=1
        path.write_bytes(value);record['files']['warp.so']=hashlib.sha256(value).hexdigest()
    (directory/'bundle.json').write_text(json.dumps(record))
    with pytest.raises(ValueError):support.bundle_files(directory,source)


def test_ordinary_source_wheel_is_pure_and_binary_bundle_is_not(monkeypatch,tmp_path):
    monkeypatch.delenv('MANISKILL_SOFTBODY_BINARY_DIR',raising=False)
    assert not support.SoftbodyDistribution().has_ext_modules()
    monkeypatch.setenv('MANISKILL_SOFTBODY_BINARY_DIR',str(tmp_path))
    assert support.SoftbodyDistribution().has_ext_modules()


def test_source_build_rejects_leftover_native_build_outputs(monkeypatch,tmp_path):
    monkeypatch.delenv('MANISKILL_SOFTBODY_BINARY_DIR',raising=False)
    command=support.BuildSoftbodyPython(support.SoftbodyDistribution())
    command.build_lib=str(tmp_path)
    (tmp_path/'sapien303_actor_bridge.stale.so').write_bytes(b'stale')
    with pytest.raises(ValueError,match='Stale native bundle'):command.run()

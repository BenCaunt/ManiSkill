import hashlib
import json
from types import SimpleNamespace

import pytest

from mani_skill.envs.softbody import cooked_geometry as module


def pack(tmp_path,change=None):
    (tmp_path/'blobs').mkdir();(tmp_path/'blobs/a.bin').write_bytes(b'controlled test bytes')
    config=dict(units='metres',leaves=['a'],files={'blobs/a.bin':hashlib.sha256(b'controlled test bytes').hexdigest()},
                physical={'mass':1.},source_geometry_sha256='a'*64)
    if change: change(config)
    raw=json.dumps(config).encode();(tmp_path/'pack.json').write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_load_pinned_bytes_once(tmp_path):
    digest=pack(tmp_path);p=module.CookedConvexPack.load(tmp_path,digest,1)
    assert p.blobs==(b'controlled test bytes',) and p.leaves==('a',)
    (tmp_path/'blobs/a.bin').write_bytes(b'changed afterward')
    assert p.blobs==(b'controlled test bytes',)
    with pytest.raises(ValueError,match='checksum'):module.CookedConvexPack.load(tmp_path,digest,1)


@pytest.mark.parametrize('kind',['hash','units','duplicate','missing','extra','traversal','symlink','count'])
def test_invalid_pack_never_reaches_native_code(tmp_path,kind):
    def change(c):
        if kind=='units':c['units']='mm'
        if kind=='duplicate':c['leaves']=['a','a']
        if kind=='missing':c['leaves']=['b']
        if kind=='extra':c['files']['blobs/b.bin']=c['files']['blobs/a.bin']
        if kind=='traversal':c['files']['../escape']=c['files'].pop('blobs/a.bin')
    digest=pack(tmp_path,change)
    if kind=='hash':digest='0'*64
    if kind=='symlink':
        p=tmp_path/'blobs/a.bin';p.rename(tmp_path/'elsewhere');p.symlink_to(tmp_path/'elsewhere')
    with pytest.raises((ValueError,FileNotFoundError)):
        module.CookedConvexPack.load(tmp_path,digest,2 if kind in ('duplicate','count') else 1)


def test_each_row_owns_shapes_while_clones_share_meshes(tmp_path,monkeypatch):
    p=module.CookedConvexPack.load(tmp_path,pack(tmp_path),1)
    system=object();bodies=[SimpleNamespace(collision_shapes=[],entity=SimpleNamespace(scene=SimpleNamespace(physx_system=system))) for _ in range(3)]
    calls=[]
    def attach(system,body,blobs,material,density):
        calls.append(('attach',body,blobs,density));body.collision_shapes.append(object())
    def clone(system,source,target):
        calls.append(('clone',source,target));target.collision_shapes.append(object())
    monkeypatch.setattr(module,'version',lambda _: '3.0.3')
    monkeypatch.setattr(module,'import_module',lambda _:SimpleNamespace(attach=attach,clone_to=clone))
    p.attach(system,bodies,object(),300.)
    assert [c[0] for c in calls]==['attach','clone','clone']
    assert calls[0][2]==list(p.blobs) and calls[0][3]==300.
    assert all(c[1] is bodies[0] for c in calls)


def test_nonempty_or_foreign_rows_rejected_before_mutation(tmp_path,monkeypatch):
    p=module.CookedConvexPack.load(tmp_path,pack(tmp_path),1);system=object();calls=[]
    monkeypatch.setattr(module,'version',lambda _: '3.0.3')
    monkeypatch.setattr(module,'import_module',lambda _:SimpleNamespace(attach=lambda *a:calls.append(a)))
    def body(owner,shapes):return SimpleNamespace(collision_shapes=shapes,entity=SimpleNamespace(scene=SimpleNamespace(physx_system=owner)))
    for rows in [[],[body(system,[]),body(system,[object()])],[body(system,[]),body(object(),[])]]:
        with pytest.raises(ValueError):p.attach(system,rows,object(),300.)
    assert not calls

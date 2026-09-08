"""Load pinned, offline PhysX cooking output into ordinary native shapes."""
from dataclasses import dataclass
import hashlib
from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path, PurePosixPath
import re


@dataclass(frozen=True)
class CookedConvexPack:
    manifest_sha256: str
    leaves: tuple
    blobs: tuple
    physical: dict
    source_geometry_sha256: str

    @classmethod
    def load(cls, directory, expected_sha256, expected_count):
        directory=Path(directory).resolve()
        manifest=directory/'pack.json'
        if manifest.is_symlink() or not 0<manifest.stat().st_size<=1024**2:
            raise ValueError('Invalid cooked convex manifest')
        raw=manifest.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected_sha256:
            raise ValueError('Cooked convex manifest checksum mismatch')
        config=json.loads(raw)
        leaves=config['leaves'];files=config['files']
        if (config['units']!='metres' or len(leaves)!=expected_count
                or len(set(leaves))!=len(leaves)
                or any(not isinstance(x,str) or not re.fullmatch('[a-z0-9_]+',x) for x in leaves)
                or not expected_count<=len(files)<=expected_count+32):
            raise ValueError('Invalid cooked convex pack layout')
        blobs={};total=0
        for name,digest in files.items():
            relative=PurePosixPath(name);path=directory/name
            if (relative.is_absolute() or '..' in relative.parts or '\\' in name
                    or name!=relative.as_posix() or path.is_symlink()
                    or not path.resolve().is_relative_to(directory)
                    or any(p.is_symlink() for p in path.parents if p!=directory.parent)
                    or not path.is_file() or not 0<path.stat().st_size<=4*1024**2):
                raise ValueError('Invalid cooked convex pack file: '+name)
            value=path.read_bytes();total+=len(value)
            if total>64*1024**2 or hashlib.sha256(value).hexdigest()!=digest:
                raise ValueError('Cooked convex pack file checksum/size mismatch: '+name)
            if name.startswith('blobs/'):
                blobs[name]=value
        expected_names=['blobs/'+name+'.bin' for name in leaves]
        if set(blobs)!=set(expected_names):
            raise ValueError('Cooked convex pack has missing or extra blobs')
        return cls(expected_sha256,tuple(leaves),tuple(blobs[n] for n in expected_names),
                   config['physical'],config['source_geometry_sha256'])

    def attach(self, system, bodies, material, density):
        """Build the first row, then share immutable meshes through native clones."""
        if version('sapien')!='3.0.3':
            raise RuntimeError('Cooked mesh compatibility currently requires SAPIEN 3.0.3')
        try:
            bridge=import_module('sapien303_cooked_bridge')
        except ModuleNotFoundError as exc:
            if exc.name!='sapien303_cooked_bridge': raise
            raise RuntimeError('Pour requires the native cooked-mesh adapter; see tools/softbody/native/cooked/README.md') from exc
        bodies=tuple(bodies)
        if not bodies or len({id(body) for body in bodies})!=len(bodies):
            raise ValueError('Expected distinct native bottle bodies')
        if any(body.collision_shapes or body.entity.scene.physx_system is not system for body in bodies):
            raise ValueError('Cooked bottle bodies must be empty and belong to the supplied system')
        bridge.attach(system,bodies[0],list(self.blobs),material,density)
        for body in bodies[1:]: bridge.clone_to(system,bodies[0],body)
        if any(len(body.collision_shapes)!=len(self.leaves) for body in bodies):
            raise RuntimeError('Native cooked bottle shape count differs from the pinned pack')

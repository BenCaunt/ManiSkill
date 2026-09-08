"""Metric mesh-to-MPM conversion adapted from ManiSkill 2 v0.5.3.

Source: mani_skill2/envs/mpm/utils.py at 493be36121a9dd06071a57172274babe617b789f.
Meshes remain in each rigid body's local coordinates. Dense signed-distance
volumes preserve open cavities; no convex hull replaces the MPM collision mesh.
"""

import hashlib
from pathlib import Path

import numpy as np
import sapien
import trimesh

from .mpm import wp
from warp.sim.model import DenseVolume


def visual_meshes(body):
    component = body.entity.find_component_by_type(sapien.render.RenderBodyComponent)
    if component is None:
        raise ValueError(f"No visual collision mesh for {body.name}")
    meshes = []
    for shape in component.render_shapes:
        if not isinstance(shape, sapien.render.RenderShapeTriangleMesh):
            raise NotImplementedError("Visual SDF conversion currently requires triangle mesh shapes")
        for part in shape.parts:
            mesh = trimesh.Trimesh(part.vertices * shape.scale, part.triangles)
            mesh.apply_transform(shape.local_pose.to_transformation_matrix())
            meshes.append(mesh)
    return meshes


def convex_collision_meshes(body):
    """Read actual cooked hulls in body coordinates for legacy reward geometry."""
    meshes = []
    for shape in body.get_collision_shapes():
        if not isinstance(shape, sapien.physx.PhysxCollisionShapeConvexMesh):
            raise NotImplementedError('Expected convex triangle-mesh collision geometry')
        mesh = trimesh.Trimesh(shape.vertices * shape.scale, shape.triangles)
        mesh.apply_transform(shape.local_pose.to_transformation_matrix())
        meshes.append(mesh)
    if not meshes:
        raise ValueError('Missing convex collision geometry')
    return meshes


def mesh_signature(meshes):
    digest = hashlib.sha256(b'maniskill2-v0.5.3-sdf-v2')
    for mesh in meshes:
        digest.update(np.asarray(mesh.vertices, dtype=np.float64).tobytes())
        digest.update(np.asarray(mesh.faces, dtype=np.int64).tobytes())
    return digest.hexdigest()


def mesh_sdf(meshes):
    """Preserve the reference sampling and face-normal convention exactly."""
    if not meshes:
        raise ValueError("Cannot build an SDF without meshes")
    bbox = trimesh.util.concatenate(meshes).bounds
    dx = min(.01, float(np.max(bbox[1] - bbox[0])) / 40)
    if dx <= 0 or not np.isfinite(bbox).all():
        raise ValueError("Invalid metric mesh bounds")
    margin = max(dx * 3, .01)
    dim = np.ceil((bbox[1] - bbox[0] + margin * 2) / dx).astype(int)
    if int(np.prod(dim)) > 4_000_000:
        raise ValueError("SDF exceeds the per-body diagnostic allocation limit")
    lower = (bbox[0] + bbox[1]) / 2 - dim * dx / 2
    points = np.zeros((*dim, 3))
    points[..., 0] += (np.arange(.5, dim[0]) * dx + lower[0])[:, None, None]
    points[..., 1] += (np.arange(.5, dim[1]) * dx + lower[1])[None, :, None]
    points[..., 2] += (np.arange(.5, dim[2]) * dx + lower[2])[None, None, :]
    points = points.reshape(-1, 3)
    fields, normals = [], []
    for mesh in meshes:
        query = trimesh.proximity.ProximityQuery(mesh)
        sdf = -query.signed_distance(points)
        surface, _, indices = query.on_surface(points)
        normal = (points - surface) * np.sign(sdf)[:, None]
        # Historical code uses 1e6 here (not 1e-6). Keep its face normals for
        # compatibility; correcting that reference behavior needs a new protocol.
        mask = np.linalg.norm(normal, axis=-1) < 1e6
        normal[mask] = mesh.face_normals[indices][mask]
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8
        fields.append(sdf.reshape(dim))
        normals.append(normal.reshape((*dim, 3)))
    fields, normals = np.stack(fields), np.stack(normals)
    index = fields.argmin(0)[None]
    return dict(sdf=np.take_along_axis(fields, index, 0)[0],
                normal=np.take_along_axis(normals, index[..., None], 0)[0],
                position=lower, scale=np.ones(3) * dx, dim=dim)


def register_visual_body(builder, body, cache_dir):
    meshes = visual_meshes(body)
    signature = mesh_signature(meshes)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f'{signature}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            sdf = dict(data)
    else:
        sdf = mesh_sdf(meshes)
        np.savez_compressed(path, **sdf)
    if not all(np.isfinite(v).all() for v in sdf.values()):
        raise ValueError("Non-finite signed-distance volume")
    index = builder.add_body(origin=wp.transform_identity())
    volume = DenseVolume(np.concatenate([sdf['normal'], sdf['sdf'][..., None]], -1),
                         sdf['position'], sdf['scale'], mass=1., I=np.eye(3), com=np.zeros(3))
    builder.add_shape_dense_volume(index, volume=volume)
    com = body.cmass_local_pose
    rotation = com.to_transformation_matrix()[:3, :3]
    builder.set_body_mass(index, float(body.mass), rotation @ np.diag(body.inertia) @ rotation.T, com.p)
    return dict(body=body.name, signature=signature, cache_file=str(path),
                mesh_bounds_m=trimesh.util.concatenate(meshes).bounds.tolist(),
                grid_dim=sdf['dim'].tolist(), grid_spacing_m=sdf['scale'].tolist())

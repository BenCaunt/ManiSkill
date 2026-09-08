"""World-wrench projection for fixed-base native SAPIEN articulations.

Only generalized forces are written. This module neither changes robot state nor
steps the world. Callers own force scheduling and must supply a fresh complete
baseline qf buffer on every application. Floating bases are unsupported here.
"""
import numpy as np
import torch
from sapien import physx


def rotate_wxyz(quaternion, vector):
    """Rotate by an engine quaternion without normalizing reported state."""
    xyz, scalar = quaternion[..., 1:], quaternion[..., :1]
    xyz, vector = torch.broadcast_tensors(xyz, vector)
    return vector + 2 * torch.linalg.cross(xyz, torch.linalg.cross(xyz, vector) + scalar * vector)


class ArticulationWrenchProjector:
    """Project world [torque-about-COM, force] to native active-joint order.

    Capture invariant topology, COM offsets and parent joint frames once after
    model construction. Rebuild this object if those inputs change during reset.
    `project` accepts one articulation or arbitrary leading batch dimensions.
    It includes fixed-link descendants and correctly excludes sibling branches.
    """
    def __init__(self, articulation, *, device='cpu', dtype=torch.float32):
        self.links = list(articulation.links)
        self.joints = list(articulation.active_joints)
        self.link_count, self.dof = len(self.links), int(articulation.dof)
        self.device, self.dtype = torch.device(device), dtype
        if articulation.root.joint.type != 'fixed':
            raise NotImplementedError('Floating-base wrenches require a separate root momentum path')
        if self.dof != len(self.joints) or any(
            j.dof != 1 or j.type not in ('revolute', 'revolute_unwrapped', 'prismatic') for j in self.joints
        ):
            raise NotImplementedError('Expected one-DOF revolute/prismatic active joints')
        indices = {link: i for i, link in enumerate(self.links)}
        child_to_joint = {joint.child_link: i for i, joint in enumerate(self.joints)}
        ancestors = np.zeros((self.dof, self.link_count), dtype=bool)
        for column, link in enumerate(self.links):
            current, visited = link, set()
            while current is not None:
                if current in visited or current not in indices:
                    raise ValueError('Invalid articulation tree')
                visited.add(current)
                if current in child_to_joint: ancestors[child_to_joint[current], column] = True
                current = current.parent
        tensor = lambda value: torch.as_tensor(np.asarray(value), dtype=dtype, device=self.device)
        self.ancestors = torch.as_tensor(ancestors, device=self.device)
        self.parent_indices = torch.tensor([indices[j.parent_link] for j in self.joints],
                                           dtype=torch.long, device=self.device)
        self.revolute = torch.tensor([j.type != 'prismatic' for j in self.joints], dtype=torch.bool, device=self.device)
        self.com_local = tensor([body.cmass_local_pose.p for body in self.links]).reshape(-1, 3)
        self.joint_parent_p = tensor([j.pose_in_parent.p for j in self.joints]).reshape(-1, 3)
        rotations = tensor([j.pose_in_parent.q for j in self.joints]).reshape(-1, 4)
        self.joint_parent_axis = rotate_wxyz(rotations, tensor([1., 0., 0.]))

    def project(self, link_poses_wxyz, wrenches_torque_force):
        poses = torch.as_tensor(link_poses_wxyz, dtype=self.dtype, device=self.device)
        wrenches = torch.as_tensor(wrenches_torque_force, dtype=self.dtype, device=self.device)
        if poses.shape[-2:] != (self.link_count, 7) or wrenches.shape != (*poses.shape[:-1], 6):
            raise ValueError('Expected matching (..., link_count, 7) poses and (..., link_count, 6) wrenches')
        if not torch.isfinite(poses).all() or not torch.isfinite(wrenches).all():
            raise ValueError('Non-finite link state or wrench')
        norms = torch.sum(poses[..., 3:]**2, dim=-1)
        if torch.any(torch.abs(norms-1) > 1e-3):
            raise ValueError('Link pose has a non-unit quaternion')
        com = poses[..., :3] + rotate_wxyz(poses[..., 3:], self.com_local)
        parents = poses.index_select(-2, self.parent_indices)
        origins = parents[..., :3] + rotate_wxyz(parents[..., 3:], self.joint_parent_p)
        axes = rotate_wxyz(parents[..., 3:], self.joint_parent_axis)
        offsets = com.unsqueeze(-3) - origins.unsqueeze(-2)
        forces = wrenches[..., 3:].unsqueeze(-3).expand_as(offsets)
        moments = wrenches[..., :3].unsqueeze(-3) + torch.linalg.cross(offsets, forces)
        angular = torch.sum(moments * axes.unsqueeze(-2), dim=-1)
        linear = torch.sum(forces * axes.unsqueeze(-2), dim=-1)
        contribution = torch.where(self.revolute.unsqueeze(-1), angular, linear)
        return torch.sum(contribution * self.ancestors, dim=-1)


class GPUArticulationForces:
    """Bind projectors to actual initialized GPU pose and generalized-force rows.

    The supplied articulations must share this PhysxGpuSystem. Construct only
    after gpu_init, and rebuild after GPU topology/index changes. Explicit
    baseline_qf covers the entire system, preserving independent controllers and
    other articulations. Call apply once per physical step, after accumulating
    all external wrenches. No force is retained implicitly between calls.
    """
    def __init__(self, system, articulations):
        if not isinstance(system, physx.PhysxGpuSystem):
            raise TypeError('GPUArticulationForces requires a native PhysxGpuSystem')
        self.system = system
        self.articulations = list(articulations)
        if len(set(self.articulations)) != len(self.articulations):
            raise ValueError('An articulation may be bound only once')
        self.buffer = system.cuda_articulation_qf.torch()
        self.projectors = [ArticulationWrenchProjector(a, device=self.buffer.device)
                           for a in self.articulations]
        self.rows, self.pose_rows, self.native_pose_rows = [], [], []
        for a, projector in zip(self.articulations, self.projectors):
            if any(link not in system.articulation_link_components for link in a.links):
                raise ValueError('Articulation does not belong to the supplied GPU system')
            row = int(a.gpu_index)
            indices = [int(link.gpu_pose_index) for link in a.links]
            if row < 0 or row >= self.buffer.shape[0] or projector.dof > self.buffer.shape[1] or min(indices) < 0:
                raise ValueError('GPU articulation buffers are not initialized')
            self.rows.append(row)
            self.native_pose_rows.append(indices)
            self.pose_rows.append(torch.tensor(indices, device=self.buffer.device))

    def fetch_link_poses(self):
        self.system.gpu_fetch_articulation_link_pose()
        data = self.system.cuda_rigid_body_data.torch()
        return [data.index_select(0, rows)[..., :7] for rows in self.pose_rows]

    def project(self, wrenches_by_articulation):
        if len(wrenches_by_articulation) != len(self.articulations):
            raise ValueError('Provide one full link-wrench array per bound articulation')
        # Native index changes invalidate cached rows; never silently load a
        # different articulation after topology was rebuilt elsewhere.
        if any(int(a.gpu_index) != row or [int(l.gpu_pose_index) for l in a.links] != indices
               for a, row, indices in zip(self.articulations, self.rows, self.native_pose_rows)):
            raise RuntimeError('GPU topology changed; rebuild the force adapter')
        poses = self.fetch_link_poses()
        projected = torch.zeros_like(self.buffer)
        for row, projector, pose, wrench in zip(self.rows, self.projectors, poses, wrenches_by_articulation):
            projected[row, :projector.dof] = projector.project(pose, wrench)
        return projected

    def apply(self, wrenches_by_articulation, *, baseline_qf):
        baseline = torch.as_tensor(baseline_qf, dtype=self.buffer.dtype, device=self.buffer.device)
        if baseline.shape != self.buffer.shape or not torch.isfinite(baseline).all():
            raise ValueError('Supply a finite complete baseline qf buffer for this physical step')
        total = baseline + self.project(wrenches_by_articulation)
        if not torch.isfinite(total).all():
            raise ValueError('Non-finite combined generalized force')
        self.buffer.copy_(total)
        self.system.gpu_apply_articulation_qf()
        return total

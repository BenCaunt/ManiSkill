"""Fixed-base gravity/Coriolis compensation using native-model Pinocchio.

This computes controller forces from observed joint state. It never changes
physical state, creates a second simulated robot, or advances a physics engine.
Rebuild after changing root orientation, mass properties or joint frames.
"""
import numpy as np


class FixedBasePassiveForces:
    """Compute the qddot=0 inverse-dynamics force in native active-joint order.

    The native URDF exporter places the fixed root at identity, so world gravity
    must first be expressed in that root's frame. Supply the actual fixed-root
    quaternion, including when the articulation lives in a GPU PhysX system.
    Pinocchio uses the articulation's mass, principal inertia, COM frames and
    joint frames, exported after all reference parameters have been applied.
    """
    def __init__(self, articulation, *, gravity_world, root_quaternion_wxyz):
        if articulation.root.joint.type != 'fixed':
            raise NotImplementedError('Passive-force model requires a fixed root')
        self.dof = int(articulation.dof)
        joints = list(articulation.active_joints)
        if self.dof != len(joints) or any(j.dof != 1 or j.type not in
                ('revolute', 'revolute_unwrapped', 'prismatic') for j in joints):
            raise NotImplementedError('Expected one-DOF revolute/prismatic joints')
        if any(link.disable_gravity for link in articulation.links):
            raise NotImplementedError('Per-link gravity disabling requires a separate model')
        gravity = np.asarray(gravity_world, dtype=np.float64)
        quaternion = np.asarray(root_quaternion_wxyz, dtype=np.float64)
        if gravity.shape != (3,) or quaternion.shape != (4,) or not (
                np.isfinite(gravity).all() and np.isfinite(quaternion).all()):
            raise ValueError('Expected finite gravity and root quaternion')
        if abs(np.dot(quaternion, quaternion) - 1) > 1e-3:
            raise ValueError('Fixed root must have a unit quaternion')
        # Inverse quaternion rotation; do not normalize native reported state.
        xyz = -quaternion[1:]
        self.gravity_root = gravity + 2 * np.cross(xyz, np.cross(xyz, gravity) + quaternion[0] * gravity)
        self.model = articulation.create_pinocchio_model(gravity=self.gravity_root.tolist())
        self.zero = np.zeros(self.dof, dtype=np.float64)

    def compute(self, qpos, qvel, *, gravity=True, coriolis_and_centrifugal=True):
        q = np.asarray(qpos, dtype=np.float64)
        v = np.asarray(qvel, dtype=np.float64)
        if q.shape != (self.dof,) or v.shape != (self.dof,) or not (
                np.isfinite(q).all() and np.isfinite(v).all()):
            raise ValueError('Expected finite native-order joint position and velocity')
        if not gravity and not coriolis_and_centrifugal:
            return self.zero.copy()
        static = None
        if not coriolis_and_centrifugal or not gravity:
            static = np.asarray(self.model.compute_inverse_dynamics(q, self.zero, self.zero)).copy()
        if coriolis_and_centrifugal:
            total = np.asarray(self.model.compute_inverse_dynamics(q, v, self.zero)).copy()
            if not gravity:
                total -= static
        else:
            total = static
        if total.shape != (self.dof,) or not np.isfinite(total).all():
            raise RuntimeError('Pinocchio returned invalid passive forces')
        return total


class GPUPassiveForces:
    """Read actual GPU joint state and return compensation in system qf layout.

    Construct after gpu_init and reset state application. The returned buffer
    contains compensation for the supplied articulations and zeros elsewhere;
    callers combine it with other controller forces and MPM reactions before
    applying qf once. This class itself never writes a native GPU force buffer.
    Rebuild after reset changes the root pose, topology or model parameters.
    """
    def __init__(self, system, articulations, *, gravity_world):
        from sapien import physx
        if not isinstance(system, physx.PhysxGpuSystem):
            raise TypeError('GPUPassiveForces requires native GPU PhysX')
        self.system = system
        self.articulations = list(articulations)
        if len(set(self.articulations)) != len(self.articulations):
            raise ValueError('An articulation may be bound only once')
        system.gpu_fetch_articulation_link_pose()
        poses = system.cuda_rigid_body_data.torch().cpu().numpy()
        self.rows, self.root_rows, self.root_quaternions, self.models = [], [], [], []
        for robot in self.articulations:
            if any(link not in system.articulation_link_components for link in robot.links):
                raise ValueError('Articulation does not belong to this GPU system')
            row, root_row = int(robot.gpu_index), int(robot.root.gpu_pose_index)
            buffer = system.cuda_articulation_qf.torch()
            if row < 0 or row >= buffer.shape[0] or robot.dof > buffer.shape[1] or root_row < 0 or root_row >= len(poses):
                raise ValueError('GPU articulation buffers are not initialized')
            quaternion = poses[root_row, 3:7].copy()
            self.models.append(FixedBasePassiveForces(robot, gravity_world=gravity_world,
                root_quaternion_wxyz=quaternion))
            self.rows.append(row); self.root_rows.append(root_row); self.root_quaternions.append(quaternion)

    def compute(self):
        import torch
        system = self.system
        if any(int(robot.gpu_index) != row or int(robot.root.gpu_pose_index) != root_row
                or int(robot.dof) != model.dof for robot, row, root_row, model in
                zip(self.articulations, self.rows, self.root_rows, self.models)):
            raise RuntimeError('GPU topology changed; rebuild passive-force models')
        system.gpu_fetch_articulation_qpos(); system.gpu_fetch_articulation_qvel()
        system.gpu_fetch_articulation_link_pose()
        q = system.cuda_articulation_qpos.torch().cpu().numpy()
        v = system.cuda_articulation_qvel.torch().cpu().numpy()
        poses = system.cuda_rigid_body_data.torch().cpu().numpy()
        if any(not np.array_equal(poses[row, 3:7], saved) for row, saved in zip(self.root_rows, self.root_quaternions)):
            raise RuntimeError('Fixed root orientation changed; rebuild passive-force models')
        result = torch.zeros_like(system.cuda_articulation_qf.torch())
        for row, model in zip(self.rows, self.models):
            force = model.compute(q[row, :model.dof], v[row, :model.dof])
            result[row, :model.dof] = torch.as_tensor(force, dtype=result.dtype, device=result.device)
        return result


class LegacyPassiveForceMixin:
    """Preserve legacy compensation while the MPM world owns GPU qf application."""
    def reset(self, init_qpos=None):
        self._mpm_passive_adapter = None
        self.mpm_baseline_qf = None
        return super().reset(init_qpos)

    def before_simulation_step(self):
        if self.scene.gpu_sim_enabled:
            if getattr(self, '_mpm_passive_adapter', None) is None:
                self._mpm_passive_adapter = GPUPassiveForces(self.scene.px, self.robot._objs,
                    gravity_world=self.scene.sim_config.scene_config.gravity)
            self.mpm_baseline_qf = self._mpm_passive_adapter.compute()
            super().before_simulation_step()
        else:
            robot = self.robot._objs[0]
            passive = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
            super().before_simulation_step()
            robot.set_qf(passive)

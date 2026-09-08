"""Two-way coupling between the preserved legacy MPM solver and SAPIEN 3.

Scheduling follows ManiSkill 2 v0.5.3 MPMBaseEnv.step_action: synchronize rigid
state, integrate MPM substeps, average reaction wrenches, apply forces, step
PhysX, and rotate the MPM state buffers. All wrenches are in world coordinates;
torques are about each body's center of mass, as produced by the MPM kernels.

This module is intentionally explicit about its current CPU PhysX scope. MPM
may run on CUDA or CPU. GPU PhysX needs a different force/synchronization adapter.
"""

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import sapien
from sapien import physx

_ROOT = Path(__file__).resolve().parents[3]
_WARP_ROOT = _ROOT / "warp_maniskill"
if "warp" in sys.modules and _WARP_ROOT not in Path(sys.modules["warp"].__file__).resolve().parents:
    raise ImportError("Legacy MPM requires the fork's bundled Warp in a separate process")
sys.path.insert(0, str(_WARP_ROOT))
import warp as wp
from mpm.mpm_model import MPMModelBuilder
from mpm.mpm_simulator import Simulator


def warp_pose(pose):
    """SAPIEN wxyz -> Warp xyzw, retaining the body's origin rather than its COM."""
    return np.r_[pose.p, pose.q[1:], pose.q[0]].astype(np.float32)


def register_collision_body(builder, body):
    """Copy metric primitive colliders and actual inertial properties into MPM.

    No SAPIEN state is assigned here. Mesh/SDF conversion is a separate adapter
    and must preserve open cavities; unsupported shapes never become boxes.
    """
    supported = (physx.PhysxCollisionShapeBox, physx.PhysxCollisionShapeSphere,
                 physx.PhysxCollisionShapeCapsule)
    shapes = body.get_collision_shapes()
    if not shapes or any(not isinstance(s, supported) for s in shapes):
        raise NotImplementedError("Primitive adapter requires box, sphere, or capsule colliders")
    index = builder.add_body(origin=wp.transform_identity())
    for shape in shapes:
        local = warp_pose(shape.local_pose)
        kwargs = dict(body=index, pos=tuple(local[:3]), rot=tuple(local[3:]))
        if isinstance(shape, physx.PhysxCollisionShapeBox):
            builder.add_shape_box(**kwargs, hx=shape.half_size[0], hy=shape.half_size[1], hz=shape.half_size[2])
        elif isinstance(shape, physx.PhysxCollisionShapeSphere):
            builder.add_shape_sphere(**kwargs, radius=shape.radius)
        else:
            # Both SAPIEN and this customized Warp capsule use the local X axis.
            builder.add_shape_capsule(**kwargs, radius=shape.radius, half_width=shape.half_length)
    if isinstance(body, physx.PhysxRigidBodyComponent):
        com = body.cmass_local_pose
        rotation = com.to_transformation_matrix()[:3, :3]
        builder.set_body_mass(index, float(body.mass),
                              rotation @ np.diag(body.inertia) @ rotation.T, com.p)
    else:
        builder.set_body_mass(index, 0., np.zeros((3, 3)), np.zeros(3))
    return index


@dataclass
class CouplingStep:
    rigid_pose_xyzw: np.ndarray
    rigid_twist_angular_linear: np.ndarray
    substep_wrench_torque_force: np.ndarray
    mean_wrench_torque_force: np.ndarray
    rigid_dt: float


class MPMCoupler:
    """A single CPU-PhysX scene coupled to a legacy MPM model.

    Call step() once per PhysX timestep. For an environment integration, pass its
    before_simulation_step callback so drive targets/forces retain legacy order.
    Pose/state assignment belongs to environment reset, never this step method.
    """

    def __init__(self, scene, model, states, bodies, *, mpm_dt, rigid_dt=None):
        if not isinstance(scene.physx_system, physx.PhysxCpuSystem):
            raise NotImplementedError("This adapter requires CPU PhysX; GPU force coupling is not implemented here")
        self.scene = scene
        self.model = model
        self.states = list(states)
        self.bodies = list(bodies)
        self.mpm_dt = float(mpm_dt)
        self.rigid_dt = float(scene.get_timestep() if rigid_dt is None else rigid_dt)
        if not np.isfinite([self.mpm_dt, self.rigid_dt]).all() or min(self.mpm_dt, self.rigid_dt) <= 0:
            raise ValueError("Timesteps must be positive and finite")
        self.substeps = round(self.rigid_dt / self.mpm_dt)
        if self.substeps < 1 or not np.isclose(self.substeps * self.mpm_dt, self.rigid_dt, rtol=1e-7, atol=1e-10):
            raise ValueError("Rigid timestep must be an integer multiple of the MPM timestep")
        if len(self.states) != self.substeps + 1:
            raise ValueError("Provide one MPM buffer per substep plus the initial state")
        if len(self.bodies) != model.body_count:
            raise ValueError("SAPIEN body order/count must match the MPM builder")
        if len({id(body) for body in self.bodies}) != len(self.bodies):
            raise ValueError("A rigid body cannot be coupled twice")
        self.simulator = Simulator(device=model.device)
        self.failed = False
        self.pending_step = None

    def prepare_step(self, before_physx=None):
        """Integrate MPM and apply its reaction, leaving PhysX stepping to the caller.

        Pair with complete_step() after exactly one PhysX step. This allows the
        existing ManiSkill simulation loop to retain ownership of its timestep.
        """
        if self.failed:
            raise RuntimeError("Coupler failed previously; reset/rebuild before continuing")
        if self.pending_step is not None:
            raise RuntimeError("Complete the pending PhysX step before preparing another")
        if not np.isclose(self.scene.get_timestep(), self.rigid_dt, rtol=1e-7, atol=1e-10):
            raise RuntimeError("PhysX timestep changed after coupling was configured")
        poses = np.asarray([warp_pose(b.entity_pose) for b in self.bodies], dtype=np.float32).reshape(-1, 7)
        twists = np.asarray([np.r_[b.angular_velocity, b.linear_velocity]
                             if isinstance(b, physx.PhysxRigidBodyComponent) else np.zeros(6)
                             for b in self.bodies], dtype=np.float32).reshape(-1, 6)
        if not np.isfinite(poses).all() or not np.isfinite(twists).all():
            self.failed = True
            raise RuntimeError("Non-finite rigid state")
        try:
            if self.bodies:
                for state in self.states:
                    state.body_q.assign(poses)
                    state.body_qd.assign(twists)
            for i in range(self.substeps):
                self.simulator.simulate(self.model, self.states[i], self.states[i + 1], self.mpm_dt)
            wrenches = (np.stack([s.ext_body_f.numpy() for s in self.states[:-1]])
                        if self.bodies else np.zeros((self.substeps, 0, 6), dtype=np.float32))
            if not np.isfinite(wrenches).all() or self.states[-1].struct.error.numpy()[0] != 0:
                raise RuntimeError("MPM kernel failed or produced a non-finite wrench")
            n = self.model.struct.n_particles
            for field in ("particle_q", "particle_qd", "particle_F", "particle_C", "particle_volume_correction"):
                if not np.isfinite(getattr(self.states[-1].struct, field).numpy()[:n]).all():
                    raise RuntimeError(f"MPM produced non-finite {field}")
            mean = wrenches.mean(axis=0)
            if before_physx is not None:
                before_physx()
            for body, wrench in zip(self.bodies, mean):
                if isinstance(body, physx.PhysxRigidBodyComponent) and not getattr(body, "kinematic", False):
                    body.add_force_torque(wrench[3:], wrench[:3], mode="force")
        except BaseException:
            self.failed = True
            raise
        self.pending_step = CouplingStep(poses, twists, wrenches, mean, self.rigid_dt)
        return self.pending_step

    def complete_step(self):
        """Publish the new MPM state after the caller's PhysX step succeeds."""
        if self.failed or self.pending_step is None:
            raise RuntimeError("No valid prepared step to complete")
        detail = self.pending_step
        self.states = [self.states[-1], *self.states[:-1]]
        self.pending_step = None
        return detail

    def step(self, before_physx=None):
        self.prepare_step(before_physx)
        try:
            self.scene.step()
        except BaseException:
            self.failed = True
            raise
        return self.complete_step()

    def particle_state(self):
        n = self.model.struct.n_particles
        state = self.states[0].struct
        return {k: getattr(state, field).numpy()[:n].copy() for k, field in (
            ("x", "particle_q"), ("v", "particle_qd"), ("F", "particle_F"),
            ("C", "particle_C"), ("vc", "particle_volume_correction"))}

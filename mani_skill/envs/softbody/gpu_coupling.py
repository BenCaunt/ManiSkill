"""Explicit shared-world scheduling for CUDA MPM and native GPU PhysX.

This correctness path transfers MPM body data through CPU arrays. It does not
claim high-throughput vectorized environments. Rebuild the world and couplers
after a reset that changes topology, model parameters or GPU buffer allocation.
"""
import numpy as np
import torch
from sapien import physx

from .mpm import MPMCoupler, warp_pose
from .gpu_wrenches import GPUArticulationForces


class _SceneBodies:
    def __init__(self, world, scene, bodies):
        self.world, self.scene, self.bodies = world, scene, list(bodies)

    def read_rigid_state(self):
        if not self.world._active:
            raise RuntimeError('Read GPU state only within the shared world step')
        poses, twists = [], []
        for body in self.bodies:
            if isinstance(body, physx.PhysxRigidBodyComponent):
                row = self.world._rigid_data[int(body.gpu_pose_index)]
                # Native SAPIEN GPU fetch already removes set_scene_offset.
                # Subtracting it again displaces MPM colliders by whole scenes.
                poses.append(np.r_[row[:3], row[4:7], row[3]])
                twists.append(np.r_[row[10:13], row[7:10]])
            else:
                # Static entity poses remain in their owning SAPIEN scene frame.
                poses.append(warp_pose(body.entity_pose)); twists.append(np.zeros(6))
        return np.asarray(poses,dtype=np.float32).reshape(-1,7), np.asarray(twists,dtype=np.float32).reshape(-1,6)

    def apply_wrenches(self, wrenches):
        if not self.world._active:
            raise RuntimeError('Accumulate reaction forces only within the shared world step')
        for body,wrench in zip(self.bodies,wrenches):self.world._accumulate(body,wrench)


class MPMGPUWorld:
    """One native PhysX step for all attached MPM scene models.

    Call system.gpu_init before creating this object. Add fresh MPM models, then
    call step with explicit complete controller/baseline force buffers. All MPM
    substeps finish before one force application and one shared PhysX step.
    Static and kinematic bodies receive no reaction force. Fixed-base articulated
    links use generalized-force projection; free actors use native GPU forces.
    """
    def __init__(self, system):
        if not isinstance(system,physx.PhysxGpuSystem):
            raise TypeError('MPMGPUWorld requires native GPU PhysX')
        self.system=system
        self.couplers=[]
        self._adapters=[]
        self._active=False
        self.failed=False
        self.steps=0
        self._force_adapter=None
        self._link_mapping={}

    def add_model(self, scene, model, states, bodies, *, mpm_dt):
        if self.steps or self._active or self.failed:
            raise RuntimeError('Add models only to a new world before stepping')
        if scene.physx_system is not self.system:
            raise ValueError('All MPM scenes must share this native PhysxGpuSystem')
        if any(body.entity.scene is not scene for body in bodies):
            raise ValueError('Coupled bodies must belong to the MPM model scene')
        adapter=_SceneBodies(self,scene,bodies)
        coupler=MPMCoupler(scene,model,states,bodies,mpm_dt=mpm_dt,rigid_adapter=adapter)
        self._adapters.append(adapter);self.couplers.append(coupler)
        # Topology capture is repeated only while constructing this world.
        articulations=[]
        for a in self._adapters:
            for body in a.bodies:
                if isinstance(body,physx.PhysxArticulationLinkComponent) and body.articulation not in articulations:
                    articulations.append(body.articulation)
        self._force_adapter=GPUArticulationForces(self.system,articulations) if articulations else None
        self._link_mapping={body:(i,j) for i,a in enumerate(articulations) for j,body in enumerate(a.links)}
        return coupler

    def _accumulate(self, body, wrench):
        value=torch.as_tensor(wrench,dtype=torch.float32,device=self._rigid_data_device)
        if isinstance(body,physx.PhysxArticulationLinkComponent):
            i,j=self._link_mapping[body]
            self._link_wrenches[i][j]+=value
        elif isinstance(body,physx.PhysxRigidDynamicComponent) and not body.kinematic:
            self._dynamic_force[int(body.gpu_index),:3]+=value[3:]
            self._dynamic_torque[int(body.gpu_index),:3]+=value[:3]

    def step(self, *, baseline_qf=None, baseline_force=None, baseline_torque=None, before_physx=None):
        if self.failed or self._active or not self.couplers:
            raise RuntimeError('Need a nonfailed configured world with no pending step')
        self._active=True
        try:
            px=self.system
            px.gpu_fetch_rigid_dynamic_data()
            px.gpu_fetch_articulation_link_pose();px.gpu_fetch_articulation_link_velocity()
            data=px.cuda_rigid_body_data.torch()
            self._rigid_data=data.cpu().numpy().copy()
            self._rigid_data_device=data.device
            self._dynamic_force=torch.zeros_like(px.cuda_rigid_dynamic_force.torch())
            self._dynamic_torque=torch.zeros_like(px.cuda_rigid_dynamic_torque.torch())
            self._link_wrenches=([torch.zeros((p.link_count,6),dtype=torch.float32,device=data.device)
                                 for p in self._force_adapter.projectors] if self._force_adapter else [])
            for coupler in self.couplers:coupler.prepare_step()
            if before_physx is not None:before_physx()
            def resolve(baseline, target, required=False):
                if callable(baseline):baseline=baseline()
                if baseline is None:
                    if required:raise ValueError('Provide the complete baseline articulation qf explicitly')
                    baseline=torch.zeros_like(target)
                baseline=torch.as_tensor(baseline,dtype=target.dtype,device=target.device)
                if baseline.shape!=target.shape or not torch.isfinite(baseline).all():
                    raise ValueError('Invalid baseline force buffer')
                total=baseline+target
                if not torch.isfinite(total).all():raise ValueError('Non-finite combined force')
                return total
            if self._force_adapter:
                baseline=resolve(baseline_qf,torch.zeros_like(self._force_adapter.buffer),required=True)
                self.last_applied_qf=self._force_adapter.apply(self._link_wrenches,baseline_qf=baseline)
            elif baseline_qf is not None:
                total=resolve(baseline_qf,torch.zeros_like(px.cuda_articulation_qf.torch()))
                px.cuda_articulation_qf.torch().copy_(total);px.gpu_apply_articulation_qf()
                self.last_applied_qf=total
            force=resolve(baseline_force,self._dynamic_force)
            torque=resolve(baseline_torque,self._dynamic_torque)
            if force.shape[0]:
                px.cuda_rigid_dynamic_force.torch().copy_(force);px.gpu_apply_rigid_dynamic_force()
                px.cuda_rigid_dynamic_torque.torch().copy_(torque);px.gpu_apply_rigid_dynamic_torque()
            px.step()
            details=[coupler.complete_step() for coupler in self.couplers]
            self.steps+=1
            return details
        except BaseException:
            self.failed=True
            for coupler in self.couplers:coupler.failed=True
            raise
        finally:
            self._active=False
            self._rigid_data=None

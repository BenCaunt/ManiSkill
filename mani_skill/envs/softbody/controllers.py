"""Legacy end-effector actions on native SAPIEN 3 joint drives.

Adapted from ManiSkill 2 v0.5.3 pd_ee_pose.py and Panda configuration. Legacy
rotations are rotation vectors, with ee/base/ee_align composition semantics.
These differ from the current ManiSkill controller's Euler-angle conventions.
See NOTICE.md for source terms. Pose assignment here changes only IK targets.
"""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import sapien
import torch
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from mani_skill.agents.controllers import (
    PDJointPosController, PDJointPosControllerConfig, PDJointPosVelControllerConfig,
    PDJointVelControllerConfig,
)
from mani_skill.agents.controllers.base_controller import ControllerConfig
from mani_skill.utils import gym_utils
from mani_skill.utils.structs import Pose


class LegacyEEPosController(PDJointPosController):
    def _initialize_joints(self):
        if self.scene.gpu_sim_enabled or self.scene.num_envs != 1:
            raise NotImplementedError('Legacy IK currently requires one CPU PhysX scene')
        super()._initialize_joints()
        self._native_robot = self.articulation._objs[0]
        self.pmodel = self._native_robot.create_pinocchio_model()
        self.qmask = np.zeros(self._native_robot.dof, dtype=bool)
        self._joint_indices = self.active_joint_indices.cpu().numpy()
        self.qmask[self._joint_indices] = True
        self.ee_link = self.articulation.links_map[self.config.ee_link]
        self.ee_link_idx = self._native_robot.links.index(self.ee_link._objs[0])

    def _initialize_action_space(self):
        self.single_action_space = spaces.Box(np.float32(np.broadcast_to(self.config.lower,3)),
                                               np.float32(np.broadcast_to(self.config.upper,3)),dtype=np.float32)

    @property
    def ee_pose_at_base(self):
        return self.articulation.pose.inv() * self.ee_link.pose

    def reset(self):
        super().reset()
        self._target_pose = self.ee_pose_at_base

    def compute_target_pose(self, previous, action):
        value = action.detach().cpu().numpy()[0]
        delta = sapien.Pose(value)
        if not self.config.use_delta:
            if self.config.frame != 'base':
                raise ValueError('Absolute legacy position requires the base frame')
            result = delta
        elif self.config.frame == 'base':
            result = delta * previous.sp
        elif self.config.frame == 'ee':
            result = previous.sp * delta
        else:
            raise ValueError(f'Unknown legacy position frame: {self.config.frame}')
        return Pose.create(result, device=self.device)

    def set_action(self, action):
        action = self._preprocess_action(action)
        self._step = 0
        self._start_qpos = self.qpos
        previous = self._target_pose if self.config.use_target else self.ee_pose_at_base
        self._target_pose = self.compute_target_pose(previous,action)
        result, success, _ = self.pmodel.compute_inverse_kinematics(self.ee_link_idx,self._target_pose.sp,
            initial_qpos=self._native_robot.qpos,active_qmask=self.qmask,max_iterations=100)
        self.last_ik_success = bool(success)
        self._target_qpos = torch.as_tensor(result[self._joint_indices],dtype=self.qpos.dtype,device=self.device)[None] if success else self._start_qpos
        if self.config.interpolate:
            self._step_size = (self._target_qpos-self._start_qpos)/self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)

    def get_state(self):
        return {'target_pose':self._target_pose.raw_pose} if self.config.use_target else {}

    def set_state(self, state):
        if self.config.use_target:
            self._target_pose = Pose.create(state['target_pose'],device=self.device)


@dataclass
class LegacyEEPosControllerConfig(ControllerConfig):
    lower: float
    upper: float
    stiffness: float
    damping: float
    ee_link: str
    force_limit: float = 100.
    friction: float = 0.
    frame: str = 'ee'
    use_delta: bool = True
    use_target: bool = False
    interpolate: bool = False
    normalize_action: bool = True
    drive_mode: str = 'force'
    controller_cls = LegacyEEPosController


class LegacyEEPoseController(LegacyEEPosController):
    def _initialize_action_space(self):
        low = np.float32(np.r_[np.broadcast_to(self.config.pos_lower,3),np.broadcast_to(-self.config.rot_bound,3)])
        high = np.float32(np.r_[np.broadcast_to(self.config.pos_upper,3),np.broadcast_to(self.config.rot_bound,3)])
        self.single_action_space = spaces.Box(low,high,dtype=np.float32)

    def _clip_and_scale_action(self, action):
        position = gym_utils.clip_and_scale_action(action[:,:3],self.action_space_low[:3],self.action_space_high[:3])
        rotation = action[:,3:].clone()
        norm = torch.linalg.norm(rotation,dim=1,keepdim=True)
        rotation = rotation / torch.maximum(norm,torch.ones_like(norm)) * self.config.rot_bound
        return torch.cat((position,rotation),dim=1)

    def compute_target_pose(self, previous, action):
        value = action.detach().cpu().numpy()[0]
        quat = Rotation.from_rotvec(value[3:6]).as_quat()[[3,0,1,2]]
        delta = sapien.Pose(value[:3],quat)
        if not self.config.use_delta:
            if self.config.frame != 'base':
                raise ValueError('Absolute legacy pose requires the base frame')
            result = delta
        elif self.config.frame == 'ee':
            result = previous.sp * delta
        elif self.config.frame in ('base','ee_align'):
            result = delta * previous.sp
            if self.config.frame == 'ee_align':
                result.p = previous.sp.p + value[:3]
        else:
            raise ValueError(f'Unknown legacy pose frame: {self.config.frame}')
        return Pose.create(result,device=self.device)


@dataclass
class LegacyEEPoseControllerConfig(ControllerConfig):
    pos_lower: float
    pos_upper: float
    rot_bound: float
    stiffness: float
    damping: float
    ee_link: str
    force_limit: float = 100.
    friction: float = 0.
    frame: str = 'ee'
    use_delta: bool = True
    use_target: bool = False
    interpolate: bool = False
    normalize_action: bool = True
    drive_mode: str = 'force'
    controller_cls = LegacyEEPoseController


def legacy_arm_configs(joints, ee_link, *, absolute_pose=False):
    """All eleven original Panda arm modes, plus Pour's absolute pose mode."""
    position = PDJointPosControllerConfig(joints,None,None,1000.,100.,force_limit=100.,normalize_action=False)
    delta = PDJointPosControllerConfig(joints,-.1,.1,1000.,100.,force_limit=100.,use_delta=True)
    target = deepcopy(delta); target.use_target=True
    ee_position = LegacyEEPosControllerConfig(joints,-.1,.1,1000.,100.,ee_link)
    ee_pose = LegacyEEPoseControllerConfig(joints,-.1,.1,.1,1000.,100.,ee_link)
    ee_target_position = deepcopy(ee_position); ee_target_position.use_target=True
    ee_target_pose = deepcopy(ee_pose); ee_target_pose.use_target=True
    aligned = deepcopy(ee_pose); aligned.frame='ee_align'
    # BaseAgent uses insertion order to choose its default control mode.
    result = dict(pd_joint_delta_pos=delta,pd_joint_pos=position,
        pd_ee_delta_pos=ee_position,pd_ee_delta_pose=ee_pose,pd_ee_delta_pose_align=aligned,
        pd_joint_target_delta_pos=target,
        pd_ee_target_delta_pos=ee_target_position,pd_ee_target_delta_pose=ee_target_pose,
        pd_joint_vel=PDJointVelControllerConfig(joints,-1.,1.,100.,force_limit=100.),
        pd_joint_pos_vel=PDJointPosVelControllerConfig(joints,None,None,1000.,100.,force_limit=100.,normalize_action=False),
        pd_joint_delta_pos_vel=PDJointPosVelControllerConfig(joints,-.1,.1,1000.,100.,force_limit=100.,use_delta=True))
    if absolute_pose:
        result['pd_ee_pose']=LegacyEEPoseControllerConfig(joints,-100.,100.,np.pi,1000.,100.,ee_link,
            use_delta=False,frame='base',normalize_action=False)
    return result

"""Native Panda using the pinned legacy URDF and measured loader outputs.

The parameter values are reference simulation inputs, not physical measurements.
"""
from copy import deepcopy
from pathlib import Path
import sapien

from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.utils import sapien_utils
from .controllers import legacy_arm_configs


def apply_reference_robot_parameters(robot, parameters):
    """Preserve native loader mass properties and joint frames from an export."""
    if set(robot.links_map) != {v['name'] for v in parameters['links']}:
        raise ValueError('Robot link names differ from the reference model')
    for record in parameters['links']:
        body = robot.links_map[record['name']]._objs[0]
        body.mass = record['mass']
        body.inertia = record['inertia']
        com = record['com']
        body.cmass_local_pose = sapien.Pose(com[:3], com[3:])
    for record in parameters['joints']:
        joint = robot.joints_map[record['name']]._objs[0]
        parent, child = record['parent_pose'], record['child_pose']
        joint.pose_in_parent = sapien.Pose(parent[:3], parent[3:])
        joint.pose_in_child = sapien.Pose(child[:3], child[3:])


class LegacyPanda(Panda):
    uid = 'legacy_mpm_panda'
    legacy_urdf_name = 'panda_v2.urdf'

    def __init__(self, *args, legacy_asset_dir, robot_parameters, **kwargs):
        self.urdf_path = str(Path(legacy_asset_dir) / 'descriptions' / self.legacy_urdf_name)
        if not Path(self.urdf_path).is_file():
            raise FileNotFoundError(self.urdf_path)
        self._reference_parameters = robot_parameters
        super().__init__(*args, **kwargs)

    def _load_articulation(self, initial_pose=None):
        # SAPIEN 2 ignored the URDF's finger mimic tag; the original controller
        # drives both joints to the same target without a physical tendon.
        # The SAPIEN 3 default adds a 1e5-stiffness tendon, changing the forces
        # on a grasped rope. Disable that extra constraint at construction.
        if self.scene.num_envs != 1 or self.build_separate:
            raise NotImplementedError('Legacy Panda currently requires one CPU scene')
        loader = self.scene.create_urdf_loader()
        loader.name = self.uid if self._agent_idx is None else f'{self.uid}-agent-{self._agent_idx}'
        loader.fix_root_link = self.fix_root_link
        loader.load_multiple_collisions_from_file = self.load_multiple_collisions
        loader.disable_self_collisions = self.disable_self_collisions
        config = sapien_utils.parse_urdf_config(self.urdf_config)
        sapien_utils.check_urdf_config(config)
        sapien_utils.apply_urdf_config(loader, config)
        parsed = loader.parse(self.urdf_path)
        if len(parsed['articulation_builders']) != 1 or parsed['actor_builders']:
            raise ValueError('Legacy Panda must be one articulation')
        builder = parsed['articulation_builders'][0]
        builder.initial_pose = initial_pose
        self.robot = builder.build(build_mimic_joints=False)
        self.robot_link_names = [link.name for link in self.robot.links]

    def _after_loading_articulation(self):
        apply_reference_robot_parameters(self.robot, self._reference_parameters)

    @property
    def _controller_configs(self):
        gripper = super()._controller_configs['pd_joint_pos']['gripper']
        gripper.friction = 0.
        return {key:dict(arm=value,gripper=deepcopy(gripper),balance_passive_force=False)
                for key,value in legacy_arm_configs(self.arm_joint_names,self.ee_link_name,
                    absolute_pose=getattr(self,'legacy_absolute_pose',False)).items()}

    def before_simulation_step(self):
        robot = self.robot._objs[0]
        passive = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        super().before_simulation_step()
        robot.set_qf(passive)

"""Shared native robot, scene settings, and material settings for bucket tasks.

Preserves the pinned ManiSkill 2 v0.5.3 inputs. See NOTICE.md for source terms.
"""
from pathlib import Path
import json

import sapien

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import PDJointPosControllerConfig
from .legacy_base import LegacyMPMEnv


class LegacyPandaBucket(BaseAgent):
    uid = 'legacy_panda_bucket'
    urdf_config = {}

    def __init__(self, *args, legacy_asset_dir, **kwargs):
        self.urdf_path = str(Path(legacy_asset_dir) / 'descriptions/panda_bucket.urdf')
        if not Path(self.urdf_path).is_file():
            raise FileNotFoundError(self.urdf_path)
        super().__init__(*args, **kwargs)

    def _after_loading_articulation(self):
        # Preserve actual reference model inputs across URDF loader versions.
        # Mesh-cooked inertia, missing-inertia defaults, and joint-frame
        # quaternion conversions otherwise differ between SAPIEN 2 and 3.
        path = Path(__file__).with_name('legacy_fill_physics.json')
        parameters = json.loads(path.read_text())['robot_parameters']
        for link in parameters['links']:
            body = self.robot.links_map[link['name']]._objs[0]
            body.mass = link['mass']
            body.inertia = link['inertia']
            com = link['com']
            body.cmass_local_pose = sapien.Pose(com[:3], com[3:])
        for record in parameters['joints']:
            joint = self.robot.joints_map[record['name']]._objs[0]
            parent, child = record['parent_pose'], record['child_pose']
            joint.pose_in_parent = sapien.Pose(parent[:3], parent[3:])
            joint.pose_in_child = sapien.Pose(child[:3], child[3:])

    @property
    def _controller_configs(self):
        names = [f'panda_joint{i}' for i in range(1, 8)]
        def config(delta, target=False):
            return dict(arm=PDJointPosControllerConfig(names, -.1 if delta else None,
                        .1 if delta else None, 1000., 100., force_limit=100.,
                        friction=0., use_delta=delta, use_target=target,
                        normalize_action=delta), balance_passive_force=False)
        return {'pd_joint_delta_pos': config(True), 'pd_joint_pos': config(False),
                'pd_joint_target_delta_pos': config(True, True)}

    def before_simulation_step(self):
        # Legacy CPU controller cancels gravity AND Coriolis/centrifugal loads.
        # Merely disabling gravity (the normal MS3 default) is not equivalent.
        robot = self.robot._objs[0]
        passive = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        super().before_simulation_step()
        robot.set_qf(passive)


class LegacyBucketEnv(LegacyMPMEnv):
    def _load_agent(self, options):
        self.agent = LegacyPandaBucket(self.scene, self._control_freq, self._control_mode,
                                      legacy_asset_dir=self.legacy_asset_dir,
                                      initial_pose=sapien.Pose([-.6, 0., 0.]))
        self.bucket = self.agent.robot.links_map['bucket']._objs[0]

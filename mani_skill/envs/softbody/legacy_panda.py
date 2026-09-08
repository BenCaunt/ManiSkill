"""Native Panda using the pinned legacy URDF and measured loader outputs.

The parameter values are reference simulation inputs, not physical measurements.
"""
from pathlib import Path
import sapien

from mani_skill.agents.robots.panda.panda import Panda


class LegacyPanda(Panda):
    uid = 'legacy_mpm_panda'

    def __init__(self, *args, legacy_asset_dir, robot_parameters, **kwargs):
        self.urdf_path = str(Path(legacy_asset_dir) / 'descriptions/panda_v2.urdf')
        if not Path(self.urdf_path).is_file():
            raise FileNotFoundError(self.urdf_path)
        self._reference_parameters = robot_parameters
        super().__init__(*args, **kwargs)

    def _after_loading_articulation(self):
        parameters = self._reference_parameters
        if set(self.robot.links_map) != {v['name'] for v in parameters['links']}:
            raise ValueError('Legacy Panda link names differ from the reference model')
        for record in parameters['links']:
            body = self.robot.links_map[record['name']]._objs[0]
            body.mass = record['mass']
            body.inertia = record['inertia']
            com = record['com']
            body.cmass_local_pose = sapien.Pose(com[:3], com[3:])
        for record in parameters['joints']:
            joint = self.robot.joints_map[record['name']]._objs[0]
            parent, child = record['parent_pose'], record['child_pose']
            joint.pose_in_parent = sapien.Pose(parent[:3], parent[3:])
            joint.pose_in_child = sapien.Pose(child[:3], child[3:])

    @property
    def _controller_configs(self):
        configs = super()._controller_configs
        for config in configs.values():
            config['balance_passive_force'] = False
            for key in ('arm', 'gripper'):
                config[key].friction = 0.
        return configs

    def before_simulation_step(self):
        robot = self.robot._objs[0]
        passive = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        super().before_simulation_step()
        robot.set_qf(passive)

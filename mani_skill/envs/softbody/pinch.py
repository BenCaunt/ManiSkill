"""Native Pinch-v0 using explicit numeric reference level and model exports.

Adapted from ManiSkill 2 v0.5.3 pinch_env.py; see NOTICE.md. Official benchmark
levels are supplied separately. Catalog-authored diagnostic levels do not imply
benchmark coverage. Only reset assigns particle, robot or goal state.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
from .base_env import MATERIAL_FIELDS
from .geometry import load_numeric_pack_file, register_reference_collision_body
from .legacy_base import LegacyMPMEnv
from .legacy_panda import LegacyPanda
from .mpm import MPMModelBuilder, wp
from warp.distance import compute_chamfer_distance

PINCH_PACK_MANIFEST_SHA256 = 'fca566f8d96f8c1ff2f1a7c60e93ad34ae71e89294c3179bf2b519257f25858d'
GOAL_FIELDS = ('deformed_distance', 'goal_depths', 'goal_rgbs', 'goal_cam_pos',
               'goal_cam_rot', 'goal_cam_intrinsic')
GOAL_SHAPES = ((2,), (4,128,128), (4,128,128,3), (4,3), (4,4), (3,3))


class LegacyPandaPinch(LegacyPanda):
    uid = 'legacy_mpm_panda_pinch'
    legacy_urdf_name = 'panda_pinch.urdf'

    @property
    def _controller_configs(self):
        configs = super()._controller_configs
        for value in configs.values():
            value['gripper'].upper = .06
        return configs


def project_goal_points(data):
    """Preserve the original four-camera, pixel-center, padded point cloud."""
    clouds = []
    for depth, pos, rot in zip(data['goal_depths'], data['goal_cam_pos'], data['goal_cam_rot']):
        transform = sapien.Pose(pos, rot).to_transformation_matrix()
        pixels = np.stack(np.meshgrid(np.arange(.5,128.5), np.arange(.5,128.5))
                          + [np.ones((128,128))], -1)
        camera = np.linalg.solve(data['goal_cam_intrinsic'], (pixels*depth[...,None]).reshape(-1,3).T).T.reshape(128,128,3)
        camera = (camera*[1,-1,-1]) @ np.array([[0,0,-1],[-1,0,0],[0,1,0]]).T
        world = camera[depth>0] @ transform[:3,:3].T + transform[:3,3]
        clouds.append(np.c_[world,np.ones(len(world))])
    points = np.concatenate(clouds)
    return np.pad(points, ((0,65536-len(points)),(0,0)))


def validate_goal(points, data, count):
    if points.shape != (count,3) or not np.isfinite(points).all():
        raise ValueError('Pinch goal must match the live particle count')
    if set(data) != set(GOAL_FIELDS):
        raise ValueError('Pinch goal camera and distance fields required')
    for key, shape in zip(GOAL_FIELDS, GOAL_SHAPES):
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f'Invalid Pinch goal field: {key}')
    if np.any(data['deformed_distance'] < 0) or data['deformed_distance'].sum() <= 0:
        raise ValueError('Pinch initial deformation distance must have positive total')
    if np.any(data['goal_depths'] < 0) or abs(np.linalg.det(data['goal_cam_intrinsic'])) < 1e-12:
        raise ValueError('Invalid Pinch camera depth or intrinsic matrix')
    if not np.allclose(np.linalg.norm(data['goal_cam_rot'],axis=1),1,rtol=0,atol=1e-5):
        raise ValueError('Pinch camera quaternion must be normalized')
    rgb = data['goal_rgbs']
    if np.any(rgb<0) or np.any(rgb>255) or np.any(rgb!=np.floor(rgb)):
        raise ValueError('Pinch numeric level currently requires uint8-range RGB values')


@register_env('Pinch-v0', max_episode_steps=300)
class PinchEnv(LegacyMPMEnv):
    PARTICLE_CHECKPOINT_GROUPS = (*LegacyMPMEnv.PARTICLE_CHECKPOINT_GROUPS, 'task_particles')

    def __init__(self, *args, level_dir=None, legacy_mpm_data_dir=None, **kwargs):
        directory = legacy_mpm_data_dir or os.environ.get('MANISKILL_LEGACY_MPM_DATA')
        if not directory:
            raise ValueError('Provide the pinned Pinch robot/contact numeric export')
        directory = Path(directory).resolve()
        manifest = directory/'export.json'
        if manifest.stat().st_size>2*1024**2 or hashlib.sha256(manifest.read_bytes()).hexdigest()!=PINCH_PACK_MANIFEST_SHA256:
            raise ValueError('Pinch numeric pack manifest checksum mismatch')
        self.reference_pack = json.loads(manifest.read_text())
        self.reference_geometry = [load_numeric_pack_file(directory,r['file'],self.reference_pack['files'][r['file']])
                                   for r in self.reference_pack['geometry']]
        self.level_dir = Path(level_dir or directory/'levels').resolve()
        manifest = self.level_dir/'levels.json'
        if manifest.stat().st_size > 2*1024**2:
            raise ValueError('Pinch level manifest exceeds size limit')
        self.levels = json.loads(manifest.read_text())['levels']
        if not self.levels or any(not isinstance(k,str) or Path(k).name!=k for k in self.levels):
            raise ValueError('Pinch level names must be nonempty basenames')
        super().__init__(*args, **kwargs)

    def _load_agent(self, options):
        self.agent = LegacyPandaPinch(self.scene,self._control_freq,self._control_mode,
            legacy_asset_dir=self.legacy_asset_dir,robot_parameters=self.reference_pack['robot_parameters'],
            initial_pose=sapien.Pose([-.56,0.,0.]))
        self.grasp_site = self.agent.robot.links_map['panda_hand_tcp']._objs[0]

    def _load_scene(self, options):
        self._load_ground()

    @property
    def _default_sensor_configs(self):
        return [CameraConfig('base_camera',sapien.Pose([.4,0.,.3],euler2quat(0.,np.pi/10,-np.pi)),128,128,np.pi/2,near=.001,far=10.)]

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig('render_camera',sapien.Pose([-.05,.7,.3],euler2quat(0.,np.pi/10,-np.pi/2)),512,512,1.,near=.001,far=10.)

    def _initialize_episode(self, env_idx, options):
        rng = np.random.RandomState(int(self._episode_seed[0]))
        noise = rng.uniform([-.1]*7+[0,0],[.1]*7+[0,0])
        nominal = np.array([0,.01,0,-1.96,0,1.98,0,.06,.06])+noise
        self.agent.reset(torch.as_tensor(nominal,dtype=torch.float32,device=self.device)[None])
        self.agent.controller.reset()
        self.agent.robot.set_pose(sapien.Pose([-.56,0.,0.]))
        filename = options.get('level_file')
        if filename is None:
            filename = str(rng.choice(sorted(self.levels)))
        if filename not in self.levels:
            raise ValueError('Unknown Pinch level_file')
        record = self.levels[filename]
        data = load_numeric_pack_file(self.level_dir,record['file'],record['sha256'])
        n = len(data['x']); self._validate_checkpoint_count(n)
        shapes = {'x':(n,3),'v':(n,3),'F':(n,3,3),'C':(n,3,3),'vc':(n,),
            'root_pose':(7,),'root_velocity':(6,),'qpos':(9,),'qvel':(9,),'ground_pose':(7,),
            'particle_mass':(n,),'particle_vol':(n,),'particle_mu_lam_ys':(n,3),
            'particle_friction_cohesion':(n,3),'particle_type':(n,)}
        for key, shape in shapes.items():
            if data[key].shape!=shape:
                raise ValueError(f'Invalid Pinch initial field: {key}')
        if np.any(data['particle_type']!=0) or np.any(data['particle_mass']<=0) or np.any(data['particle_vol']<=0):
            raise ValueError('Pinch requires positive-mass, positive-volume solid particles')
        goal_data = {key:data[key].reshape(shape) for key,shape in zip(GOAL_FIELDS,GOAL_SHAPES)}
        validate_goal(data['goal'],goal_data,n)
        self.level_file = filename
        self.level_sha256 = record['source_sha256']
        builder = MPMModelBuilder();builder.set_mpm_domain([.5,.5,.5],grid_length=.01)
        bodies = []
        for record, arrays in zip(self.reference_pack['geometry'],self.reference_geometry):
            body = self.agent.robot.links_map[record['name']]._objs[0]
            register_reference_collision_body(builder,body,record,arrays);bodies.append(body)
        for i in range(n):
            builder.add_mpm_particle(data['x'][i],data['v'][i],data['particle_mass'][i],data['particle_vol'][i],0,
                material=data['particle_mu_lam_ys'][i],material2=data['particle_friction_cohesion'][i],
                color=(.65237011,.14198029,.02201299))
        self.rebuild_mpm(builder,bodies)
        self._set_goal(data['goal'],goal_data)
        checkpoint = self.get_state_dict()
        checkpoint['mpm'] = {key:torch.as_tensor(data[key],device=self.device)[None] for key in checkpoint['mpm']}
        robot = np.r_[data['root_pose'],data['root_velocity'],data['qpos'],data['qvel']]
        checkpoint['articulations'][self.agent.robot.name] = torch.as_tensor(robot,device=self.device)[None]
        ground = self.rigid_pose(self.ground._bodies[0])
        if not np.array_equal(np.r_[ground.p,ground.q],data['ground_pose']):
            raise ValueError('Pinch level changes the fixed ground pose')
        # Source restores the level's physical state after nominal agent.reset;
        # the nominal controller/drive targets survive that physical assignment.
        self.defer_initial_state(checkpoint)

    def _configure_mpm_model(self, model):
        super()._configure_mpm_model(model)
        model.gravity = np.array([0,0,-1],dtype=np.float32)
        for prefix, values in [('static',dict(ke=100.,kd=.5,mu=.9,ka=0.)),('body',dict(ke=100.,kd=.2,mu=.5,ka=0.))]:
            for key,value in values.items():setattr(model.struct,f'{prefix}_{key}',value)
        model.adaptive_grid=False;model.particle_contact=True;model.grid_contact=False
        model.struct.ground_sticky=True;model.struct.body_sticky=False;model.struct.particle_radius=.005

    def _set_goal(self, points, data):
        self.goal_particle_points = np.array(points,dtype=np.float32,copy=True)
        self.goal_data = {k:np.array(v,copy=True) for k,v in data.items()}
        self.goal_data['goal_rgbs'] = self.goal_data['goal_rgbs'].astype(np.uint8)
        self.goal_points = project_goal_points(self.goal_data)
        self.total_deformed_distance = self.goal_data['deformed_distance']
        self.target_dist = .3*sum(self.total_deformed_distance)
        n = len(points)
        self.goal_array = wp.array(self.goal_particle_points,dtype=wp.vec3,device=self.mpm_device)
        self.dist1,self.dist2 = [wp.empty(n,dtype=wp.float32,device=self.mpm_device) for _ in range(2)]
        self.index1,self.index2 = [wp.empty(n,dtype=wp.int64,device=self.mpm_device) for _ in range(2)]
        self._chamfer_dist = None

    def _compute_chamfer(self):
        if self._chamfer_dist is None:
            n = len(self.goal_particle_points)
            if self.mpm_device=='cuda':
                compute_chamfer_distance(self.mpm_coupler.states[0].struct.particle_q,n,self.goal_array,n,
                                         self.dist1,self.dist2,self.index1,self.index2)
                wp.synchronize()
                distances = [self.dist1.numpy(),self.dist2.numpy()]
            else:
                # Original custom distance kernel is CUDA-only. CPU fallback
                # uses exact nearest neighbors with float32 norm arithmetic;
                # CPU/GPU rounding equivalence is not claimed.
                from scipy.spatial import cKDTree
                x = self.mpm_coupler.particle_state()['x'];goal=self.goal_particle_points
                distances = []
                for a,b in ((x,goal),(goal,x)):
                    index = cKDTree(b).query(a,k=1)[1]
                    delta = a-b[index]
                    distances.append(np.sum(delta*delta,axis=1,dtype=np.float32))
            # Kernel returns squared distance. Squaring again and taking the
            # fourth root preserves the legacy directed L4 metric. Explicit
            # float widening preserves NumPy 1.x scalar exponent semantics.
            self._chamfer_dist = [float(np.mean(d*d,dtype=np.float32))**.25 for d in distances]
        return self._chamfer_dist

    def _after_control_step(self):
        self._chamfer_dist=None
        super()._after_control_step()

    def evaluate(self):
        distance = sum(self._compute_chamfer())
        return dict(success=torch.tensor([distance<self.target_dist],device=self.device),
            progress=torch.tensor([1-distance/sum(self.total_deformed_distance)],device=self.device,dtype=torch.float64))

    def compute_dense_reward(self,obs,action,info):
        matrix=self.rigid_pose(self.grasp_site).to_transformation_matrix()
        bottom=np.asarray(matrix[:3,3]+matrix[:3,2]*.02,dtype=np.float32)
        distance=np.min(np.linalg.norm(self.mpm_coupler.particle_state()['x']-bottom,axis=-1))
        reach=1-np.tanh(10.*distance)
        angle=np.arcsin(np.clip(np.linalg.norm(np.cross(matrix[:3,2],[0,0,-1])),-1,1))
        return torch.tensor([-100.*sum(self._compute_chamfer())+.1*reach+.1*(1-angle)],device=self.device,dtype=torch.float32)

    def compute_normalized_dense_reward(self,obs,action,info):
        return self.compute_dense_reward(obs,action,info)

    def _get_obs_extra(self,info):
        pose=self.rigid_pose(self.grasp_site)
        return {**super()._get_obs_extra(info),'tcp_pose':torch.as_tensor(np.r_[pose.p,pose.q],device=self.device)[None],
            'target_rgb':torch.as_tensor(self.goal_data['goal_rgbs'].copy(),device=self.device)[None],
            'target_depth':torch.as_tensor(self.goal_data['goal_depths'].copy(),device=self.device,dtype=torch.float32)[None],
            'target_points':torch.as_tensor(self.goal_points.copy(),device=self.device,dtype=torch.float32)[None]}

    def get_state_dict(self):
        return {**super().get_state_dict(),
            'task':{k:torch.as_tensor(v.copy(),device=self.device)[None] for k,v in self.goal_data.items()},
            'task_particles':{'goal':torch.as_tensor(self.goal_particle_points.copy(),device=self.device)[None]}}

    def set_state_dict(self,state,env_idx=None):
        if not self._mpm_reset_active:raise RuntimeError('Pinch goal assignment requires reset')
        count = torch.as_tensor(state['mpm']['x']).shape[1]
        if set(state.get('task_particles',{}))!={'goal'}:raise ValueError('Pinch particle goal required')
        def unbatch(value):
            value = torch.as_tensor(value).detach().cpu().numpy()
            if value.ndim<1 or value.shape[0]!=1:raise ValueError('Pinch goal checkpoint must have batch size1')
            return value[0]
        points = unbatch(state['task_particles']['goal'])
        data = {k:unbatch(v) for k,v in state['task'].items()}
        validate_goal(points,data,count)
        # Flat checkpoints promote leaves; restore each fixed camera field's
        # declared dtype after checking all values, including integer RGB.
        data = {k:v.astype(self.goal_data[k].dtype) for k,v in data.items()}
        super().set_state_dict({k:v for k,v in state.items() if k not in ('task','task_particles')},env_idx)
        self._set_goal(points,data)

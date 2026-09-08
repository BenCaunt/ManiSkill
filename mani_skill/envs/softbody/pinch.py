"""Native Pinch-v0 using explicit numeric reference level and model exports.

Adapted from ManiSkill 2 v0.5.3 pinch_env.py; see NOTICE.md. Official benchmark
levels are supplied separately. Catalog-authored diagnostic levels do not imply
benchmark coverage. Only reset assigns particle, robot or goal state.
"""
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


@dataclass
class _PinchGoal:
    particles: np.ndarray
    data: dict
    projection: np.ndarray
    array: wp.array
    distances: list
    indices: list
    chamfer: Optional[list] = None


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
        pixels = np.stack(list(np.meshgrid(np.arange(.5,128.5), np.arange(.5,128.5)))
                          + [np.ones((128,128))], -1)
        camera = np.linalg.solve(data['goal_cam_intrinsic'], (pixels*depth[...,None]).reshape(-1,3).T).T.reshape(128,128,3)
        camera = (camera*[1,-1,-1]) @ np.array([[0,0,-1],[-1,0,0],[0,1,0]]).T
        world = camera[depth>0] @ transform[:3,:3].T + transform[:3,3]
        clouds.append(np.c_[world,np.ones(len(world))])
    points = np.concatenate(clouds)
    return np.pad(points, ((0,65536-len(points)),(0,0)))


def validate_goal(points, data, count):
    if points.shape != (count,3) or points.dtype.kind not in 'fiu' or not np.isfinite(points).all():
        raise ValueError('Pinch goal must match the live particle count')
    if set(data) != set(GOAL_FIELDS):
        raise ValueError('Pinch goal camera and distance fields required')
    for key, shape in zip(GOAL_FIELDS, GOAL_SHAPES):
        if data[key].shape != shape or data[key].dtype.kind not in 'fiu' or not np.isfinite(data[key]).all():
            raise ValueError(f'Invalid Pinch goal field: {key}')
    if (np.any(data['deformed_distance'] < 0) or not np.isfinite(data['deformed_distance'].sum())
            or data['deformed_distance'].sum() <= 0):
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
    _supports_mpm_batch = True

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
        self.grasp_sites = list(self.agent.robot.links_map['panda_hand_tcp']._objs)
        self.grasp_site = self.grasp_sites[0]

    def _load_scene(self, options):
        self._pinch_goals = [None]*self.num_envs
        self.level_files = [None]*self.num_envs
        self.level_sha256s = [None]*self.num_envs
        self._load_ground()

    @property
    def _default_sensor_configs(self):
        return [CameraConfig('base_camera',sapien.Pose([.4,0.,.3],euler2quat(0.,np.pi/10,-np.pi)),128,128,np.pi/2,near=.001,far=10.)]

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig('render_camera',sapien.Pose([-.05,.7,.3],euler2quat(0.,np.pi/10,-np.pi/2)),512,512,1.,near=.001,far=10.)

    def _initialize_episode(self, env_idx, options):
        indices = self._mpm_batch.indices(env_idx) if self._mpm_batch is not None else [0]
        filenames = options.get('level_file')
        if filenames is None or isinstance(filenames,str):
            filenames = [filenames]*len(indices)
        elif not isinstance(filenames,(list,tuple)) or len(filenames)!=len(indices):
            raise ValueError('Pinch level_file must name one level per selected environment')
        if any(name is not None and (not isinstance(name,str) or name not in self.levels) for name in filenames):
            raise ValueError('Unknown Pinch level_file')
        nominals=[]; levels=[]
        for index,filename in zip(indices,filenames):
            rng = np.random.RandomState(int(self._episode_seed[index]))
            noise = rng.uniform([-.1]*7+[0,0],[.1]*7+[0,0])
            nominals.append(np.array([0,.01,0,-1.96,0,1.98,0,.06,.06])+noise)
            filename = str(rng.choice(sorted(self.levels))) if filename is None else filename
            data,goal_data = self._read_level(filename)
            ground = self.rigid_pose(self.ground._bodies[index])
            if not np.array_equal(np.r_[ground.p,ground.q],data['ground_pose']):
                raise ValueError('Pinch level changes the fixed ground pose')
            levels.append((filename,data,goal_data))
        self.agent.reset(torch.as_tensor(np.array(nominals)[np.argsort(indices)],dtype=torch.float32,device=self.device))
        self.agent.controller.reset()
        self.agent.robot.set_pose(sapien.Pose([-.56,0.,0.]))
        for index,(filename,data,goal_data) in zip(indices,levels):
            builder,bodies = self._pinch_builder(data,index)
            self.rebuild_mpm(builder,bodies,env_idx=index)
            self.level_files[index] = filename
            self.level_sha256s[index] = self.levels[filename]['source_sha256']
            self._set_goal(data['goal'],goal_data,index)
        checkpoint = self.get_state_dict()
        if self._mpm_batch is not None:
            def selected(value):
                return {k:selected(v) for k,v in value.items()} if isinstance(value,dict) else value[indices].clone()
            checkpoint = selected(checkpoint)
            for key,value in checkpoint['mpm'].items():
                value.zero_()
                for row,(_,data,_) in enumerate(levels):
                    value[row,:len(data['x'])] = torch.as_tensor(data[key],device=self.device)
        else:
            data = levels[0][1]
            checkpoint['mpm'] = {key:torch.as_tensor(data[key],device=self.device)[None] for key in checkpoint['mpm']}
        robots = [np.r_[data['root_pose'],data['root_velocity'],data['qpos'],data['qvel']] for _,data,_ in levels]
        checkpoint['articulations'][self.agent.robot.name] = torch.as_tensor(np.array(robots),device=self.device)
        # Preserve nominal controller caches and current drive buffers; apply the physical state
        # only after BaseEnv has completed its final controller reset.
        self.defer_initial_state(checkpoint)

    def _read_level(self,filename):
        record = self.levels[filename]
        data = load_numeric_pack_file(self.level_dir,record['file'],record['sha256'])
        n = len(data['x']); self._validate_checkpoint_count(n)
        if self._mpm_batch is not None and n>self._mpm_batch.capacity:
            raise ValueError('Pinch level exceeds declared batch particle capacity')
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
        return data,goal_data

    def _pinch_builder(self,data,index):
        builder = MPMModelBuilder();builder.set_mpm_domain([.5,.5,.5],grid_length=.01)
        bodies = []
        for record, arrays in zip(self.reference_pack['geometry'],self.reference_geometry):
            body = self.agent.robot.links_map[record['name']]._objs[index]
            register_reference_collision_body(builder,body,record,arrays);bodies.append(body)
        for i in range(len(data['x'])):
            builder.add_mpm_particle(data['x'][i],data['v'][i],data['particle_mass'][i],data['particle_vol'][i],0,
                material=data['particle_mu_lam_ys'][i],material2=data['particle_friction_cohesion'][i],
                color=(.65237011,.14198029,.02201299))
        return builder,bodies

    def _configure_mpm_model(self, model):
        super()._configure_mpm_model(model)
        model.gravity = np.array([0,0,-1],dtype=np.float32)
        for prefix, values in [('static',dict(ke=100.,kd=.5,mu=.9,ka=0.)),('body',dict(ke=100.,kd=.2,mu=.5,ka=0.))]:
            for key,value in values.items():setattr(model.struct,f'{prefix}_{key}',value)
        model.adaptive_grid=False;model.particle_contact=True;model.grid_contact=False
        model.struct.ground_sticky=True;model.struct.body_sticky=False;model.struct.particle_radius=.005

    def _set_goal(self, points, data, index=0):
        points = np.array(points,dtype=np.float32,copy=True)
        data = {k:np.array(v,copy=True) for k,v in data.items()}
        validate_goal(points,data,len(points))
        data['goal_rgbs'] = data['goal_rgbs'].astype(np.uint8)
        n = len(points)
        self._pinch_goals[index] = _PinchGoal(points,data,project_goal_points(data),
            wp.array(points,dtype=wp.vec3,device=self.mpm_device),
            [wp.empty(n,dtype=wp.float32,device=self.mpm_device) for _ in range(2)],
            [wp.empty(n,dtype=wp.int64,device=self.mpm_device) for _ in range(2)])

    # Single-environment capture compatibility; batch state exposes every row.
    @property
    def goal_particle_points(self):return self._pinch_goals[0].particles
    @property
    def goal_data(self):return self._pinch_goals[0].data
    @property
    def goal_points(self):return self._pinch_goals[0].projection
    @property
    def total_deformed_distance(self):return self.goal_data['deformed_distance']
    @property
    def target_dist(self):return .3*sum(self.total_deformed_distance)
    @property
    def goal_array(self):return self._pinch_goals[0].array
    @property
    def dist1(self):return self._pinch_goals[0].distances[0]
    @property
    def dist2(self):return self._pinch_goals[0].distances[1]
    @property
    def index1(self):return self._pinch_goals[0].indices[0]
    @property
    def index2(self):return self._pinch_goals[0].indices[1]
    @property
    def level_file(self):return self.level_files[0]
    @property
    def level_sha256(self):return self.level_sha256s[0]

    def _compute_chamfer(self,index=0):
        goal = self._pinch_goals[index]
        if goal.chamfer is None:
            n = len(goal.particles);coupler = self.mpm_couplers[index]
            if n!=coupler.model.struct.n_particles:
                raise ValueError('Pinch live material and goal counts differ')
            if self.mpm_device=='cuda':
                compute_chamfer_distance(coupler.states[0].struct.particle_q,n,goal.array,n,
                                         *goal.distances,*goal.indices)
                wp.synchronize()
                distances = [array.numpy() for array in goal.distances]
            else:
                # Original custom distance kernel is CUDA-only. CPU fallback
                # uses exact nearest neighbors with float32 norm arithmetic;
                # CPU/GPU rounding equivalence is not claimed.
                from scipy.spatial import cKDTree
                x = coupler.particle_state()['x'];target=goal.particles
                distances = []
                for a,b in ((x,target),(target,x)):
                    nearest = cKDTree(b).query(a,k=1)[1]
                    delta = a-b[nearest]
                    distances.append(np.sum(delta*delta,axis=1,dtype=np.float32))
            # Kernel returns squared distance. Squaring again and taking the
            # fourth root preserves the legacy directed L4 metric. Explicit
            # float widening preserves NumPy 1.x scalar exponent semantics.
            goal.chamfer = [float(np.mean(d*d,dtype=np.float32))**.25 for d in distances]
        return goal.chamfer

    def _after_control_step(self):
        for goal in self._pinch_goals:goal.chamfer=None
        super()._after_control_step()

    def evaluate(self):
        distances = [sum(self._compute_chamfer(i)) for i in range(self.num_envs)]
        totals = [sum(goal.data['deformed_distance']) for goal in self._pinch_goals]
        return dict(success=torch.tensor([bool(d<.3*t) for d,t in zip(distances,totals)],device=self.device),
            progress=torch.tensor([1-d/t for d,t in zip(distances,totals)],device=self.device,dtype=torch.float64))

    def _dense_reward_value(self,index):
        matrix=self.rigid_pose(self.grasp_sites[index]).to_transformation_matrix()
        bottom=np.asarray(matrix[:3,3]+matrix[:3,2]*.02,dtype=np.float32)
        distance=np.min(np.linalg.norm(self.mpm_couplers[index].particle_state()['x']-bottom,axis=-1))
        reach=1-np.tanh(10.*distance)
        angle=np.arcsin(np.clip(np.linalg.norm(np.cross(matrix[:3,2],[0,0,-1])),-1,1))
        return -100.*sum(self._compute_chamfer(index))+.1*reach+.1*(1-angle)

    def compute_dense_reward(self,obs,action,info):
        return torch.tensor([self._dense_reward_value(i) for i in range(self.num_envs)],device=self.device,dtype=torch.float32)

    def compute_normalized_dense_reward(self,obs,action,info):
        return self.compute_dense_reward(obs,action,info)

    def _get_obs_extra(self,info):
        poses=[self.rigid_pose(body) for body in self.grasp_sites]
        return {**super()._get_obs_extra(info),
            'tcp_pose':torch.as_tensor(np.array([np.r_[p.p,p.q] for p in poses]),device=self.device),
            'target_rgb':torch.as_tensor(np.stack([g.data['goal_rgbs'] for g in self._pinch_goals]),device=self.device),
            'target_depth':torch.as_tensor(np.stack([g.data['goal_depths'] for g in self._pinch_goals]),device=self.device,dtype=torch.float32),
            'target_points':torch.as_tensor(np.stack([g.projection for g in self._pinch_goals]),device=self.device,dtype=torch.float32)}

    def get_state_dict(self):
        if self._mpm_batch is not None:
            particles=torch.zeros((self.num_envs,self._mpm_batch.capacity,3),device=self.device,dtype=torch.float32)
            for index,goal in enumerate(self._pinch_goals):
                particles[index,:len(goal.particles)]=torch.as_tensor(goal.particles,device=self.device)
        else:
            particles=torch.as_tensor(self.goal_particle_points.copy(),device=self.device)[None]
        return {**super().get_state_dict(),
            'task':{k:torch.as_tensor(np.stack([g.data[k] for g in self._pinch_goals]),device=self.device) for k in GOAL_FIELDS},
            'task_particles':{'goal':particles}}

    def set_state_dict(self,state,env_idx=None):
        if not self._mpm_reset_active:raise RuntimeError('Pinch goal assignment requires reset')
        indices = self._mpm_batch.indices(env_idx) if self._mpm_batch is not None else [0]
        if self._mpm_batch is not None:
            counts=torch.as_tensor(state['mpm_meta']['count']).detach().cpu().numpy()
            if (counts.shape!=(len(indices),1) or not np.isfinite(counts).all() or np.any(counts!=np.floor(counts))
                    or np.any((counts<1)|(counts>self._mpm_batch.capacity))):
                raise ValueError('Invalid Pinch checkpoint counts')
            counts=counts[:,0].astype(int);width=self._mpm_batch.capacity
        else:
            width=torch.as_tensor(state['mpm']['x']).shape[1];counts=[width]
        if set(state.get('task_particles',{}))!={'goal'}:raise ValueError('Pinch particle goal required')
        def rows(value):
            value = torch.as_tensor(value).detach().cpu().numpy()
            if value.ndim<1 or value.shape[0]!=len(indices):raise ValueError('Pinch goal checkpoint must match selected rows')
            return value
        points = rows(state['task_particles']['goal'])
        if points.shape!=(len(indices),width,3) or not np.isfinite(points).all():raise ValueError('Invalid Pinch goal padding')
        data = {k:rows(v) for k,v in state['task'].items()}
        goals=[]
        for row,(index,count) in enumerate(zip(indices,counts)):
            if np.any(points[row,count:]!=0):raise ValueError('Inactive Pinch goal padding must be zero')
            row_data={k:v[row] for k,v in data.items()}
            validate_goal(points[row,:count],row_data,count)
            # Flat checkpoints promote leaves; restore each camera field's
            # declared dtype after checking all values, including integer RGB.
            row_data = {k:v.astype(self._pinch_goals[index].data[k].dtype) for k,v in row_data.items()}
            goal=points[row,:count].astype(np.float32)
            validate_goal(goal,row_data,count)
            goals.append((goal,row_data))
        super().set_state_dict({k:v for k,v in state.items() if k not in ('task','task_particles')},env_idx)
        for index,(goal,row_data) in zip(indices,goals):self._set_goal(goal,row_data,index)

"""Native Write-v0 preserving the original material, stick and height-map task.

Adapted from ManiSkill 2 v0.5.3 write_env.py; see NOTICE.md. Goals are explicit
numeric HDF5 inputs. No official goal dataset is bundled or downloaded here.
"""
import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
from .controllers import LegacyEEPoseControllerConfig, legacy_arm_configs
from .geometry import load_numeric_pack_file, register_reference_collision_body
from .legacy_base import LegacyMPMEnv
from .legacy_panda import apply_reference_robot_parameters
from .passive_forces import LegacyPassiveForceMixin
from .mpm import MPMModelBuilder, wp
from mpm.height_rasterizer import rasterize_clear_kernel, rasterize_kernel


WRITE_PACK_MANIFEST_SHA256='d27e8453c5adc25c6a97d0d8319dc6bec6e9d245a7d4d241bcd2aa94d517a71c'


@dataclass
class _WriteGoal:
    points: np.ndarray
    image: wp.array
    current: wp.array
    buffer: wp.array
    display: np.ndarray
    iou: Optional[float] = None


@wp.kernel
def success_iou_kernel(goal:wp.array(dtype=int,ndim=2), current:wp.array(dtype=int,ndim=2), out:wp.array(dtype=int)):
    row,column=wp.tid()
    a=goal[row,column]<40
    an=goal[row,column]<50
    b=current[row,column]<40
    bn=current[row,column]<50
    if (an and b) or (a and bn):
        wp.atomic_add(out,0,1)
    if a or b:
        wp.atomic_add(out,1,1)


class LegacyPandaStick(LegacyPassiveForceMixin, BaseAgent):
    uid='legacy_mpm_panda_stick'
    urdf_config={}

    def __init__(self,*args,legacy_asset_dir,robot_parameters,**kwargs):
        self.urdf_path=str(Path(legacy_asset_dir)/'descriptions/panda_stick.urdf')
        if not Path(self.urdf_path).is_file():
            raise FileNotFoundError(self.urdf_path)
        self._reference_parameters=robot_parameters
        super().__init__(*args,**kwargs)

    def _after_loading_articulation(self):
        apply_reference_robot_parameters(self.robot,self._reference_parameters)

    @property
    def _controller_configs(self):
        joints=[f'panda_joint{i}' for i in range(1,8)]
        configs=legacy_arm_configs(joints,'panda_hand')
        configs['pd_ee_delta_pose_demo']=LegacyEEPoseControllerConfig(joints,-.1,.1,.1,1000.,100.,'panda_hand',
            frame='base',normalize_action=False)
        return {key:dict(arm=value,balance_passive_force=False) for key,value in configs.items()}

@register_env('Write-v0',max_episode_steps=200)
class WriteEnv(LegacyMPMEnv):
    _supports_mpm_batch = True

    def __init__(self,*args,level_dir=None,legacy_mpm_data_dir=None,**kwargs):
        directory=legacy_mpm_data_dir or os.environ.get('MANISKILL_LEGACY_MPM_DATA')
        if not directory:
            raise ValueError('Provide the pinned Write robot/contact numeric export')
        directory=Path(directory).resolve()
        manifest=directory/'export.json'
        if manifest.stat().st_size>2*1024**2 or hashlib.sha256(manifest.read_bytes()).hexdigest()!=WRITE_PACK_MANIFEST_SHA256:
            raise ValueError('Write numeric pack manifest checksum mismatch')
        self.reference_pack=json.loads(manifest.read_text())
        self.reference_geometry=[load_numeric_pack_file(directory,r['file'],self.reference_pack['files'][r['file']])
                                 for r in self.reference_pack['geometry']]
        level_dir=level_dir or os.environ.get('MANISKILL_WRITE_LEVELS')
        if not level_dir:
            raise ValueError('Provide level_dir with explicit Write HDF5 goal files')
        self.level_dir=Path(level_dir).resolve()
        self.all_filepaths=sorted(self.level_dir.glob('*.h5'))
        if not self.all_filepaths:
            raise FileNotFoundError('No Write HDF5 goals found; original goals are supplied separately')
        super().__init__(*args,**kwargs)

    def _load_agent(self,options):
        self.agent=LegacyPandaStick(self.scene,self._control_freq,self._control_mode,
            legacy_asset_dir=self.legacy_asset_dir,robot_parameters=self.reference_pack['robot_parameters'],
            initial_pose=sapien.Pose([-.55,0.,0.]))
        self.hands=list(self.agent.robot.links_map['panda_hand']._objs)
        self.hand=self.hands[0]
        # The source's end-effector is the final collision-free TCP link.
        self.end_effectors=list(self.agent.robot.links_map['panda_hand_tcp']._objs)
        self.end_effector=self.end_effectors[0]

    def _load_scene(self,options):
        self._write_goals=[None]*self.num_envs
        self.level_files=[None]*self.num_envs
        self.level_sha256s=[None]*self.num_envs
        self._load_ground()
        poses=[sapien.Pose([0.,-.13,.04]),sapien.Pose([0.,.13,.04]),
               sapien.Pose([-.13,0.,.04],[.7071068,0.,0.,.7071068]),
               sapien.Pose([.13,0.,.04],[.7071068,0.,0.,.7071068])]
        self.walls=[]
        for i,pose in enumerate(poses):
            builder=self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[.15,.02,.04])
            builder.add_box_visual(half_size=[.15,.02,.04])
            builder.initial_pose=pose
            wall=builder.build_kinematic(f'wall_{i}')
            record=self.reference_pack['geometry'][i+1]
            for body in wall._bodies:
                body.mass=record['mass'];body.inertia=record['inertia']
                body.cmass_local_pose=sapien.Pose(record['com'][:3],record['com'][3:])
            self.walls.append(wall)

    @property
    def _default_sensor_configs(self):
        return [CameraConfig('base_camera',sapien.Pose([-.2,0.,.3],euler2quat(0.,np.pi/6,0.)),128,128,np.pi/2,near=.001,far=10.)]

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig('render_camera',sapien.Pose([-.3,0.,.4],euler2quat(0.,np.pi/5,0.)),512,512,1.,near=.001,far=10.)

    def _initialize_episode(self,env_idx,options):
        indices=self._mpm_batch.indices(env_idx) if self._mpm_batch is not None else [0]
        filenames=options.get('level_file')
        if filenames is None or isinstance(filenames,str):
            filenames=[filenames]*len(indices)
        elif not isinstance(filenames,(list,tuple)) or len(filenames)!=len(indices):
            raise ValueError('Write level_file must name one file per selected environment')
        if any(name is not None and (not isinstance(name,str) or not name or Path(name).name!=name)
               for name in filenames):
            raise ValueError('Write level_file must be a filename within level_dir')
        nominal=np.array([-.029177314,.10816099,.03054934,-2.1639752,-.0013982388,2.2785723,.79039097])
        qposes=[]; levels=[]
        # Validate every selected goal before changing robot or particle state.
        for index,filename in zip(indices,filenames):
            rng=np.random.RandomState(int(self._episode_seed[index]))
            qposes.append(nominal+rng.uniform([-.1]*7,[.1]*7))
            path=self.level_dir/filename if filename is not None else Path(rng.choice(self.all_filepaths))
            goal,checksum=self._read_goal(path)
            levels.append((path.name,checksum,goal))
        # Articulation setters consume rows in boolean reset-mask order.
        self.agent.reset(torch.as_tensor(np.array(qposes)[np.argsort(indices)],dtype=torch.float32,device=self.device))
        self.agent.robot.set_pose(sapien.Pose([-.55,0.,0.]))
        for index,(filename,checksum,goal) in zip(indices,levels):
            builder,bodies=self._write_builder(int(self._episode_seed[index]),index)
            if len(builder.mpm_particle_q)!=len(goal):
                raise ValueError('Write goal and material particle counts differ')
            self.rebuild_mpm(builder,bodies,env_idx=index)
            self.level_files[index]=filename
            self.level_sha256s[index]=checksum
            self._set_goal(goal,index)

    @staticmethod
    def _read_goal(path):
        if path.is_symlink() or path.stat().st_size>64*1024**2:
            raise ValueError('Invalid Write level file')
        with path.open('rb') as stream:
            data=stream.read(64*1024**2+1)
        if len(data)>64*1024**2:
            raise ValueError('Invalid Write level file')
        with h5py.File(io.BytesIO(data),'r') as f:
            if not isinstance(f.get('goal',getlink=True),h5py.HardLink):
                raise ValueError('Write goal must be an inline numeric dataset')
            dataset=f['goal']
            if not isinstance(dataset,h5py.Dataset) or dataset.shape!=(19404,3) or dataset.dtype.kind not in 'fiu' or dataset.is_virtual or dataset.external:
                raise ValueError('Write goal must contain 19404 three-dimensional points')
            goal=np.asarray(dataset,dtype=np.float32)
        if not np.isfinite(goal).all():
            raise ValueError('Write goal points must be finite')
        return goal,hashlib.sha256(data).hexdigest()

    def _write_builder(self,seed,index):
        builder=MPMModelBuilder()
        builder.set_mpm_domain([.5,.5,.5],grid_length=.01)
        bodies=[self.hands[index],*[w._bodies[index] for w in self.walls]]
        for body,record,arrays in zip(bodies,self.reference_pack['geometry'],self.reference_geometry):
            register_reference_collision_body(builder,body,record,arrays)
        E,nu=3e5,.1
        # Explicit widening preserves the reference NumPy 1.x scalar promotion.
        # Keeping float32 scalar arithmetic under NumPy 2 drops the top layer.
        heights=np.full((42,42),.05,dtype=np.float32).astype(np.float64)
        builder.add_mpm_from_height_map(pos=(0.,0.,0.),vel=(0.,0.,0.),dx=.005,height_map=heights,density=3e3,
            mu_lambda_ys=(E/(2*(1+nu)),E*nu/((1+nu)*(1-2*nu)),2e3),friction_cohesion=(0.,0.,0.),type=0,
            jitter=True,color=(.65237011,.14198029,.02201299),random_state=np.random.RandomState(seed))
        return builder,bodies

    # Preserve the original single-environment capture API. Batched checkpoints
    # carry every row; these convenience views continue to refer to row zero.
    @property
    def goal_points(self):
        return self._write_goals[0].points

    @property
    def goal_image(self):
        return self._write_goals[0].image

    @property
    def current_image(self):
        return self._write_goals[0].current

    @property
    def iou_buffer(self):
        return self._write_goals[0].buffer

    @property
    def goal_image_display(self):
        return self._write_goals[0].display

    @property
    def level_file(self):
        return self.level_files[0]

    @property
    def level_sha256(self):
        return self.level_sha256s[0]

    def _configure_mpm_model(self,model):
        super()._configure_mpm_model(model)
        model.adaptive_grid=False;model.grid_contact=False;model.particle_contact=True
        model.struct.body_sticky=0;model.struct.ground_sticky=1;model.struct.particle_radius=.005

    def _rasterize(self,points,count,image):
        wp.launch(rasterize_clear_kernel,dim=(64,64),inputs=[image,0],device=self.mpm_device)
        wp.launch(rasterize_kernel,dim=count,inputs=[points,wp.vec3(.105,.105,0.),64/.21,1000.,int(.007*64/.21),64,64,image],device=self.mpm_device)

    def _set_goal(self,points,index=0):
        points=np.array(points,dtype=np.float32,copy=True)
        goal=wp.array(points,dtype=wp.vec3,device=self.mpm_device)
        image=wp.zeros((64,64),dtype=wp.int32,device=self.mpm_device)
        self._rasterize(goal,len(points),image)
        wp.synchronize()
        self._write_goals[index]=_WriteGoal(points,image,
            wp.zeros((64,64),dtype=wp.int32,device=self.mpm_device),
            wp.zeros(2,dtype=wp.int32,device=self.mpm_device),
            np.clip(image.numpy()[:,::-1],0,255).astype(np.uint8))

    def _compute_iou(self,index=0):
        goal=self._write_goals[index]
        if goal.iou is None:
            coupler=self.mpm_couplers[index]
            state=coupler.states[0].struct
            # Checkpoints may change the live material count independently of
            # the fixed goal. Never rasterize inactive particle padding.
            self._rasterize(state.particle_q,coupler.model.struct.n_particles,goal.current)
            goal.buffer.zero_()
            wp.launch(success_iou_kernel,dim=(64,64),inputs=[goal.image,goal.current,goal.buffer],device=self.mpm_device)
            wp.synchronize()
            intersection,union=goal.buffer.numpy()
            goal.iou=float(intersection/union) if union else float('nan')
        return goal.iou

    def _after_control_step(self):
        for goal in self._write_goals:
            goal.iou=None
        super()._after_control_step()

    def evaluate(self):
        values=[self._compute_iou(i) for i in range(self.num_envs)]
        return dict(success=torch.tensor([v>.8 for v in values],device=self.device),iou=torch.tensor(values,device=self.device))

    def _dense_reward_value(self,index):
        matrix=self.rigid_pose(self.end_effectors[index]).to_transformation_matrix()
        bottom=np.asarray(matrix[:3,3]+matrix[:3,2]*.02,dtype=np.float32)
        distance=np.min(np.linalg.norm(self.mpm_couplers[index].particle_state()['x']-bottom,axis=-1))
        reach=1-np.tanh(10.*distance)
        angle=np.arcsin(np.clip(np.linalg.norm(np.cross(matrix[:3,2],[0,0,-1])),-1,1))
        return self._compute_iou(index)+.1*reach+.1*(1-angle)

    def compute_dense_reward(self,obs,action,info):
        return torch.tensor([self._dense_reward_value(i) for i in range(self.num_envs)],dtype=torch.float32,device=self.device)

    def compute_normalized_dense_reward(self,obs,action,info):
        return self.compute_dense_reward(obs,action,info)

    def _get_obs_extra(self,info):
        poses=[self.rigid_pose(body) for body in self.end_effectors]
        return {**super()._get_obs_extra(info),
                'tcp_pose':torch.as_tensor(np.array([np.r_[p.p,p.q] for p in poses]),device=self.device),
                'goal':torch.as_tensor(np.stack([g.display for g in self._write_goals]),device=self.device)}

    def get_state_dict(self):
        return {**super().get_state_dict(),'task':{
            'goal_points':torch.as_tensor(np.stack([g.points for g in self._write_goals]),device=self.device)}}

    def set_state_dict(self,state,env_idx=None):
        if not self._mpm_reset_active:
            raise RuntimeError('Write goal assignment requires reset')
        indices=self._mpm_batch.indices(env_idx) if self._mpm_batch is not None else [0]
        points=torch.as_tensor(state['task']['goal_points']).detach().cpu().numpy()
        if (points.shape!=(len(indices),19404,3) or points.dtype.kind not in 'fiu'
                or not np.isfinite(points).all() or np.any(np.abs(points)>np.finfo(np.float32).max)):
            raise ValueError('Invalid Write checkpoint goal points')
        physical={k:v for k,v in state.items() if k!='task'}
        if indices!=sorted(indices):
            order=np.argsort(indices).tolist()
            def reorder(value):
                if isinstance(value,dict):return {k:reorder(v) for k,v in value.items()}
                value=torch.as_tensor(value)
                if value.ndim<1 or len(value)!=len(indices):raise ValueError('Invalid Write checkpoint row count')
                return value[order]
            physical=reorder(physical)
            env_idx=sorted(indices)
        super().set_state_dict(physical,env_idx)
        for index,goal in zip(indices,points):
            self._set_goal(goal,index)

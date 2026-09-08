"""Native Pour-v0 fluid task adapted from ManiSkill 2 v0.5.3.

Uses original external model inputs and fluid SDFs. The rigid bottle uses an
open-cavity decomposition in place of the reference's closed convex hull; the
original mass, COM and inertia remain explicit simulation inputs. See NOTICE.md.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
from transforms3d.quaternions import axangle2quat, qmult

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.common import np_compute_angle_between
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from .geometry import load_numeric_pack_file, register_reference_collision_body
from .cooked_geometry import CookedConvexPack
from .legacy_base import LegacyMPMEnv
from .legacy_panda import LegacyPanda
from .mpm import MPMModelBuilder

POUR_PACK_MANIFEST_SHA256 = '54d4c40bdfe3184788d5e1f1841c8806611cff1249f0aad8b5c9a150ab982710'
BOTTLE_COOKED_PACK_SHA256 = '57ca9d867a663651257e8318df3ef70aca1fae40127ef29d25944461b3db43bc'


class LegacyPourPanda(LegacyPanda):
    legacy_absolute_pose = True


@register_env('Pour-v0', max_episode_steps=350)
class PourEnv(LegacyMPMEnv):
    _supports_mpm_batch = True
    _batch_particle_capacity = 16384
    # The .048 m diameter/.09 m height recipe at .0025 m spacing fits below
    # this capacity. Only observations are padded; physics/checkpoints contain
    # exactly the live particles, and observations expose the live count.
    observation_particle_capacity = 16384
    def __init__(self, *args, legacy_mpm_data_dir=None, bottle_collision_dir=None, **kwargs):
        directory = legacy_mpm_data_dir or os.environ.get('MANISKILL_LEGACY_MPM_DATA')
        if not directory:
            raise ValueError('Provide the pinned numeric Pour model pack')
        self.legacy_mpm_data_dir = Path(directory).resolve()
        manifest = self.legacy_mpm_data_dir/'export.json'
        if manifest.stat().st_size > 2*1024**2 or hashlib.sha256(manifest.read_bytes()).hexdigest() != POUR_PACK_MANIFEST_SHA256:
            raise ValueError('Pour numeric pack manifest checksum mismatch')
        self.reference_pack = json.loads(manifest.read_text())
        self.reference_geometry = [load_numeric_pack_file(self.legacy_mpm_data_dir, r['file'],
            self.reference_pack['files'][r['file']]) for r in self.reference_pack['geometry']]
        self.bottle_collision = CookedConvexPack.load(
            bottle_collision_dir or os.environ.get('MANISKILL_BOTTLE_COLLISION_DIR')
            or self.legacy_mpm_data_dir/'collision-cooked',
            BOTTLE_COOKED_PACK_SHA256,384)
        original=self.reference_pack['geometry'][0]
        if any(not np.array_equal(self.bottle_collision.physical[field],original[field])
               for field in ('mass','inertia','com')):
            raise ValueError('Cooked bottle mass properties differ from the original model')
        if self.bottle_collision.source_geometry_sha256!=self.reference_pack['files'][original['file']]:
            raise ValueError('Cooked bottle geometry was derived from a different source')
        self._ring = None
        self._rings = []
        super().__init__(*args, **kwargs)

    def _load_agent(self, options):
        for name, digest in self.reference_pack['asset_sha256'].items():
            if hashlib.sha256((self.legacy_asset_dir/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Original Pour asset checksum mismatch: {name}')
        self.agent = LegacyPourPanda(self.scene, self._control_freq, self._control_mode,
            legacy_asset_dir=self.legacy_asset_dir, robot_parameters=self.reference_pack['robot_parameters'],
            initial_pose=sapien.Pose([-.55, 0., 0.]))
        self.grasp_sites = self.agent.robot.links_map['panda_hand_tcp']._objs
        self.lfingers = self.agent.robot.links_map['panda_leftfinger']._objs
        self.rfingers = self.agent.robot.links_map['panda_rightfinger']._objs
        self.grasp_site, self.lfinger, self.rfinger = self.grasp_sites[0], self.lfingers[0], self.rfingers[0]

    def _load_scene(self, options):
        self._load_ground()
        bottle = self.legacy_asset_dir/'deformable_manipulation/bottle.glb'
        beaker = self.legacy_asset_dir/'deformable_manipulation/beaker.glb'
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(str(bottle), scale=[.025]*3)
        builder.initial_pose = sapien.Pose()
        self.source_container = builder.build('bottle')
        self.source_bodies = self.source_container._bodies
        self.source_body = self.source_bodies[0]
        material = sapien.physx.PhysxMaterial(1., 1., 0.)
        self.bottle_collision.attach(self.scene.px,self.source_bodies,material,300.)
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(str(beaker), scale=[.04]*3)
        builder.add_nonconvex_collision_from_file(str(beaker), scale=[.04]*3)
        builder.initial_pose = sapien.Pose()
        self.target_beaker = builder.build_kinematic('target_beaker')
        self.beaker_bodies = self.target_beaker._bodies
        self.beaker_body = self.beaker_bodies[0]
        for bodies, record in zip((self.source_bodies, self.beaker_bodies), self.reference_pack['geometry']):
            for body in bodies:
                body.mass, body.inertia = record['mass'], record['inertia']
                body.cmass_local_pose = sapien.Pose(record['com'][:3], record['com'][3:])
        self._fill_heights = np.zeros((self.num_envs, 2), dtype=np.float64)
        self._reset_ik_attempts = np.zeros(self.num_envs, dtype=np.int64)
        self._rings = [None] * self.num_envs
        # Reward targets retain the reference's explicitly exported geometry.
        # They do not pretend to be measurements of the native cooked hulls.
        self.source_aabb = np.asarray(self.reference_pack['source_aabb'])
        self.target_aabb = np.asarray(self.reference_pack['target_aabb'])
        self.target_aabc = np.asarray(self.reference_pack['target_aabc'])
        self._target_height = self.reference_pack['target_height']
        self._target_radius = self.reference_pack['target_radius']

    @property
    def _default_sensor_configs(self):
        return [CameraConfig('base_camera', sapien.Pose([.4,0.,.3], euler2quat(0.,np.pi/10,-np.pi)),
            128,128,np.pi/2,near=.001,far=10.)]

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig('render_camera', sapien.Pose([-.05,.7,.3], euler2quat(0.,np.pi/10,-np.pi/2)),
            512,512,1.,near=.001,far=10.)

    def _determine_target_pos(self, rng, env_index=0):
        robot = self.agent.robot._objs[env_index]
        model = robot.create_pinocchio_model()
        index = robot.links.index(self.grasp_sites[env_index])
        # A bounded failure is preferable to an unobservable infinite reset.
        # Successful draws and their order are identical to the legacy recipe.
        for attempt in range(1000):
            r, angle = rng.uniform(.2,.25), rng.uniform(0.,np.pi)
            target = sapien.Pose([r*np.cos(angle),r*np.sin(angle),0.])
            r, angle = rng.uniform(.05,.1), rng.uniform(np.pi,2*np.pi)
            source = sapien.Pose([r*np.cos(angle),r*np.sin(angle),0.])
            q = qmult(axangle2quat([0,0,1], rng.uniform(-np.pi/8,np.pi/8)), [.5,-.5,-.5,-.5])
            qpos, success, _ = model.compute_inverse_kinematics(index,
                sapien.Pose([source.p[0]+.55,source.p[1],rng.uniform(.04,.06)],q),
                [-.555,.646,.181,-1.892,1.171,1.423,-1.75,.04,.04], active_qmask=[1]*7+[0]*2)
            if success:
                qpos[-2:] = .04
                self._reset_ik_attempts[env_index] = attempt+1
                if env_index == 0:
                    self.reset_ik_attempts = attempt+1
                return target, source, qpos
        raise RuntimeError('Pour reset could not find a reachable bottle grasp')

    def _initialize_episode(self, env_idx, options):
        if self._mpm_batch is not None:
            return self._initialize_batch_episode(env_idx)
        rng = np.random.RandomState(int(self._episode_seed[0]))
        target, source, qpos = self._determine_target_pos(rng)
        self.target_beaker.set_pose(target)
        self.source_container.set_pose(source)
        self.reset_rigid_velocity(self.source_body, [0.,0.,0.], [0.,0.,0.])
        self.agent.reset(torch.as_tensor(qpos,dtype=torch.float32,device=self.device)[None])
        self.agent.robot.set_pose(sapien.Pose([-.55,0.,0.]))
        builder, bodies, self.h1, self.h2 = self._pour_builder(rng, source, 0)
        self.rebuild_mpm(builder,bodies)
        self._update_ring()

    def _pour_builder(self, rng, source, index):
        builder = MPMModelBuilder()
        builder.set_mpm_domain([.8,.8,.8],grid_length=.005)
        bodies = [self.source_bodies[index],self.beaker_bodies[index]]
        for body, record, arrays in zip(bodies,self.reference_pack['geometry'],self.reference_geometry):
            # Fluid contact uses the original visual SDF, independently of the
            # open rigid collision decomposition used for PhysX contacts.
            register_reference_collision_body(builder,body,record,arrays,validate_shape_count=False)
        E, nu = 3e5, .1
        viscosity, density = rng.uniform(0.,3.), rng.uniform(.8e3,2e3)
        builder.add_mpm_cylinder(pos=(*source.p[:2],.01),vel=(0.,0.,0.),radius=.024,
            height=rng.uniform(.07,.09),dx=.0025,density=density,
            mu_lambda_ys=(E/(2*(1+nu)),E*nu/((1+nu)*(1-2*nu)),0.),
            friction_cohesion=(0.,0.,viscosity),type=2,jitter=False,random_state=rng,color=[0.,.5,.8])
        builder.mpm_particle_volume = [v*1.2 for v in builder.mpm_particle_volume]
        h1 = rng.uniform(.01,.02)
        return builder, bodies, h1, h1+.004

    def _initialize_batch_episode(self, env_idx):
        indices = self._mpm_batch.indices(env_idx)
        random_states, targets, sources, qposes = [], [], [], []
        for index in indices:
            rng = np.random.RandomState(int(self._episode_seed[index]))
            target, source, qpos = self._determine_target_pos(rng, index)
            random_states.append(rng); targets.append(target); sources.append(source); qposes.append(qpos)
        self.target_beaker.set_pose(Pose.create(targets,device=self.device))
        self.source_container.set_pose(Pose.create(sources,device=self.device))
        for index in indices:
            self.reset_rigid_velocity(self.source_bodies[index],[0.,0.,0.],[0.,0.,0.])
        self.agent.reset(torch.as_tensor(np.asarray(qposes),dtype=torch.float32,device=self.device))
        self.agent.robot.set_pose(sapien.Pose([-.55,0.,0.]))
        for index, rng, source in zip(indices,random_states,sources):
            builder, bodies, h1, h2 = self._pour_builder(rng,source,index)
            self._fill_heights[index] = [h1,h2]
            self.rebuild_mpm(builder,bodies,env_idx=index)
        for index in indices:
            self._update_ring(index)

    def _height_pair(self, index):
        return self._fill_heights[index] if self._mpm_batch is not None else (self.h1,self.h2)

    def _configure_mpm_model(self, model):
        super()._configure_mpm_model(model)
        model.adaptive_grid = False
        model.struct.body_sticky = 0
        model.particle_contact, model.grid_contact = True, False

    def _update_ring(self, index=0):
        if not self.scene.can_render():
            return
        # Camera groups retain render-object membership. Release them before
        # replacing a ring even when its particle pool has not changed size.
        if self.gpu_sim_enabled:
            self._invalidate_particle_render_groups()
        scene = self.scene.sub_scenes[index]
        if self._rings[index] is not None:
            scene.remove_entity(self._rings[index])
        h1, h2 = self._height_pair(index)
        angles = np.linspace(0.,2*np.pi,16,endpoint=False)
        radius = self._target_radius*1.02
        lower = np.c_[radius*np.cos(angles),radius*np.sin(angles),np.zeros(16)]
        vertices = np.r_[lower,lower+[0.,0.,h2-h1]].astype(np.float32)
        triangles = []
        for a in range(16):
            b,c,d = (a+1)%16,(a+1)%16+16,a+16
            triangles.extend([(a,b,c),(a,c,d),(a,c,b),(a,d,c)])
        material = sapien.render.RenderMaterial(base_color=[1.,0.,0.,1.])
        normals = np.tile(np.c_[np.cos(angles),np.sin(angles),np.zeros(16)],(2,1)).astype(np.float32)
        shape = sapien.render.RenderShapeTriangleMesh(vertices,np.asarray(triangles,dtype=np.uint32),
            normals,np.zeros((32,2),dtype=np.float32),material)
        component = sapien.render.RenderBodyComponent()
        component.attach(shape)
        ring = sapien.Entity()
        ring.name = 'pour_target_ring_visual_only'
        ring.add_component(component)
        ring.pose = sapien.Pose([*self.rigid_pose(self.beaker_bodies[index]).p[:2],h1])
        scene.add_entity(ring)
        self._rings[index] = ring
        if index == 0:
            self._ring = ring

    def _clear(self):
        self._ring = None
        self._rings = []
        super()._clear()

    def _task_counts(self, index=0):
        x = self.mpm_couplers[index].particle_state()['x']
        h1, h2 = self._height_pair(index)
        inside = (np.sum((x[:,:2]-self.rigid_pose(self.beaker_bodies[index]).p[:2])**2,axis=1)<self._target_radius**2) & (x[:,2]<self._target_height)
        return tuple(int(np.count_nonzero(v)) for v in
            (inside & (x[:,2]>h1),inside & (x[:,2]>h2),~inside & (x[:,2]<.001),inside))

    def _task_success(self, index=0):
        above_start,above_end,spill,_ = self._task_counts(index)
        qvel = self.agent.robot.get_qvel()[index].detach().cpu().numpy()
        upright = self.rigid_pose(self.source_bodies[index]).to_transformation_matrix()[2,2] >= .866
        return bool(above_start>100 and above_end<10 and spill<100 and upright and qvel.max()<.05 and qvel.min()>-.05)

    def evaluate(self):
        return {'success':torch.tensor([self._task_success(i) for i in range(self.num_envs)],device=self.device)}

    def _get_obs_extra(self, info):
        extra = super()._get_obs_extra(info)
        poses = [self.rigid_pose(body) for body in self.grasp_sites]
        tcp = torch.as_tensor(np.array([np.r_[pose.p,pose.q] for pose in poses]),device=self.device)
        if self._mpm_batch is not None:
            return {**extra,'tcp_pose':tcp,
                'particle_count':torch.tensor([[c.model.struct.n_particles] for c in self.mpm_couplers],dtype=torch.int32,device=self.device),
                'target':torch.as_tensor(self._fill_heights[:,:1].copy(),dtype=torch.float32,device=self.device)}
        n = self.mpm_coupler.model.struct.n_particles
        if n > self.observation_particle_capacity:
            raise ValueError('Pour particle count exceeds its observation capacity')
        padded = {}
        for key, value in extra['mpm'].items():
            output = torch.zeros((1,self.observation_particle_capacity,*value.shape[2:]),dtype=value.dtype,device=self.device)
            output[:,:n] = value
            padded[key] = output
        return {**extra,'mpm':padded,'particle_count':torch.tensor([[n]],dtype=torch.int32,device=self.device),
            'tcp_pose':tcp,
            'target':torch.tensor([[self.h1]],dtype=torch.float32,device=self.device)}

    def _check_grasp(self, index=0):
        flags = []
        for link, sign in [(self.agent.finger1_link,1),(self.agent.finger2_link,-1)]:
            impulse = self.scene.get_pairwise_contact_impulses(link,self.source_container)[index].cpu().numpy()
            direction = sign*self.rigid_pose(link._objs[index]).to_transformation_matrix()[:3,1]
            flags.append(np.linalg.norm(impulse)>=1e-6 and np.rad2deg(np_compute_angle_between(direction,impulse))<=85)
        return all(flags)

    def compute_dense_reward(self, obs, action, info):
        return torch.tensor([self._dense_reward_value(index=i) for i in range(self.num_envs)],dtype=torch.float32,device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs,action,info)/15.

    def get_state_dict(self):
        heights = self._fill_heights.copy() if self._mpm_batch is not None else [[self.h1,self.h2]]
        return {**super().get_state_dict(),'task':{'fill_heights':torch.tensor(heights,dtype=torch.float64,device=self.device)}}

    def set_state_dict(self, state, env_idx=None):
        if not self._mpm_reset_active:
            raise RuntimeError('Task state assignment requires reset')
        heights = torch.as_tensor(state['task']['fill_heights']).cpu().numpy()
        selected = self._mpm_batch.indices(env_idx) if self._mpm_batch is not None else [0]
        if (heights.shape!=(len(selected),2) or not np.isfinite(heights).all()
                or np.any(heights[:,0] <= 0) or np.any(heights[:,0] >= heights[:,1])
                or np.any(heights[:,1] >= self._target_height)):
            raise ValueError('Invalid Pour fill-height state')
        super().set_state_dict({k:v for k,v in state.items() if k!='task'},env_idx)
        if self._mpm_batch is not None:
            self._fill_heights[selected] = heights
        else:
            self.h1,self.h2 = heights[0]
        for index in selected:
            self._update_ring(index)

    def _dense_reward_value(self, reward_info=False, index=0):
        if self._task_success(index):
            if reward_info:
                return {"reward": 15}
            return 15
        above_start, above_end, spill, in_beaker = self._task_counts(index)

        source_mat = self.rigid_pose(self.source_bodies[index]).to_transformation_matrix()
        target_mat = self.rigid_pose(self.beaker_bodies[index]).to_transformation_matrix()

        a, b = self.source_aabb
        t = np.array([0.5, 0.5, 0.33])
        grasp_site_target = a * (1 - t) + b * t
        grasp_site_target = source_mat[:3, :3] @ grasp_site_target + source_mat[:3, 3]
        grasp_site_dist = np.linalg.norm(self.rigid_pose(self.grasp_sites[index]).p - grasp_site_target)
        reward_grasp_site = -grasp_site_dist

        if grasp_site_dist < 0.05:
            reward_grasp_site = 0
            check_grasp = self._check_grasp(index)
            reward_grasp = float(check_grasp)
        else:
            check_grasp = False
            reward_grasp = 0

        top_center = np.zeros(3)
        top_center[:2] = (a[:2] + b[:2]) * 0.5
        top_center[2] = b[2]
        top_center = source_mat[:3, :3] @ top_center + source_mat[:3, 3]

        bottom_center = np.zeros(3)
        bottom_center[:2] = (a[:2] + b[:2]) * 0.5
        bottom_center[2] = a[2]
        bottom_center = source_mat[:3, :3] @ bottom_center + source_mat[:3, 3]

        tx, ty, tr, tzmin, tzmax = self.target_aabc
        target_top_center = (
            target_mat[:3, :3] @ np.array([tx, ty, tzmax]) + target_mat[:3, 3]
        )

        dist_lf = np.linalg.norm(self.rigid_pose(self.lfingers[index]).p[:2] - target_top_center[:2])
        dist_rf = np.linalg.norm(self.rigid_pose(self.rfingers[index]).p[:2] - target_top_center[:2])
        reward_finger = 10 * (dist_rf - dist_lf)

        hdist = np.linalg.norm(target_top_center[:2] - self.rigid_pose(self.grasp_sites[index]).p[:2])
        vdist = target_top_center[2] - self.rigid_pose(self.grasp_sites[index]).p[2]

        reward_in_beaker = 0
        if above_start < 100 or (above_start > 100 and above_start - above_end < 100):
            done = False
            reward_in_beaker = in_beaker
        else:
            done = True
            reward_in_beaker = in_beaker - max(0, above_start - 500) * 2

        reward_spill = -spill

        stage = 0
        z = source_mat[:3, 2]
        edist = 0
        bot_hdist = 0
        if not check_grasp:
            # not grasping the bottle
            reward_orientation = 0
            reward_dist = 0
            stage = 0
        elif done:
            # finish pouring
            angle = np.arcsin(np.clip(np.linalg.norm(np.cross(z, [0, 0, 1])), -1, 1))
            reward_orientation = 3.5 - angle
            reward_dist = 5.5 - np.tanh(10.0 * (bottom_center[2] - top_center[2]))
            reward_finger = 1
            stage = 3
        elif vdist > -0.06:
            # looking for the right range
            angle = np.arcsin(np.clip(np.linalg.norm(np.cross(z, [0, 0, 1])), -1, 1))
            reward_orientation = 1 - angle
            reward_dist = 1 - np.tanh(10.0 * (max(0, vdist + 0.06)))
            stage = 1
        else:
            # within the right range to pour

            top_dist = np.linalg.norm(top_center[:2] - target_top_center[:2])
            bottom_dist = np.linalg.norm(bottom_center[:2] - target_top_center[:2])

            reward_dist = 2 - np.tanh(
                10.0 * np.linalg.norm(top_center - target_top_center)
            )
            if dist_rf - dist_lf > 0:
                reward_finger = max(
                    10 * (dist_rf - dist_lf), 10 * (bottom_dist - top_dist)
                )

            reward_orientation = 2 - np.tanh(
                10.0 * max(0, top_center[2] - bottom_center[2])
            )
            stage = 2

        if reward_info:
            return {
                "reward": reward_grasp_site
                + reward_grasp
                + reward_in_beaker * 0.001
                + reward_spill * 0.01
                + reward_orientation
                + reward_dist
                + reward_finger,
                "reward_grasp_site": reward_grasp_site,
                "reward_grasp": reward_grasp,
                "reward_in_beaker": reward_in_beaker,
                "reward_spill": reward_spill,
                "reward_orientation": reward_orientation,
                "reward_dist": reward_dist,
                "stage": stage,
                "hdist": hdist,
                "vdist": vdist,
                "tilt": top_center[2] - bottom_center[2],
                "edist": edist,
                "above_start": above_start,
                "in_beaker": in_beaker,
                "above_end": above_end,
                "tr": tr,
                "reward_finger": reward_finger,
            }
        return (
            reward_grasp_site
            + reward_grasp
            + reward_in_beaker * 0.001
            + reward_spill * 0.01
            + reward_orientation
            + reward_dist
            + reward_finger
        )

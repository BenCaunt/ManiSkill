"""Pour batch seed and selection contracts; native evidence is separate."""
from types import SimpleNamespace
import numpy as np
import pytest
import sapien
import torch

from mani_skill.envs.softbody.base_env import MPMBaseEnv
from mani_skill.envs.softbody.batch import MPMBatchRuntime
from mani_skill.envs.softbody.pour import PourEnv


def task(seeds):
    env=PourEnv.__new__(PourEnv);env.num_envs=len(seeds);env.device=torch.device('cpu')
    env._mpm_batch=MPMBatchRuntime(env,len(seeds),16384) if len(seeds)>1 else None
    env._mpm_reset_active=True;env._episode_seed=np.asarray(seeds)
    env._fill_heights=np.zeros((len(seeds),2));env._reset_ik_attempts=np.zeros(len(seeds),np.int64)
    env.grasp_sites=[object() for _ in seeds];env.model_calls=[]
    robots=[]
    for i,link in enumerate(env.grasp_sites):
        def model(i=i):
            calls=[];env.model_calls.append((i,calls))
            def solve(index,pose,initial,**kwargs):
                calls.append((index,pose,initial,kwargs))
                q=np.r_[pose.p,pose.q,[.04,.04]]
                return q,len(calls)>1,None
            return SimpleNamespace(compute_inverse_kinematics=solve)
        robots.append(SimpleNamespace(links=[link],create_pinocchio_model=model))
    env.source_bodies=[object() for _ in seeds];env.beaker_bodies=[object() for _ in seeds]
    env.source_body=env.source_bodies[0];env.beaker_body=env.beaker_bodies[0]
    env.source_poses=[];env.target_poses=[];env.qposes=[];env.velocity_rows=[];env.rebuilds=[];env.rings=[]
    env.source_container=SimpleNamespace(set_pose=lambda p:env.source_poses.append(p))
    env.target_beaker=SimpleNamespace(set_pose=lambda p:env.target_poses.append(p))
    env.agent=SimpleNamespace(reset=lambda q:env.qposes.append(q.clone()),robot=SimpleNamespace(_objs=robots,set_pose=lambda p:None))
    env.reset_rigid_velocity=lambda body,l,a:env.velocity_rows.append(env.source_bodies.index(body))
    env.rebuild_mpm=lambda builder,bodies,env_idx=0:env.rebuilds.append((env_idx,builder,bodies))
    env._update_ring=lambda index=0:env.rings.append(index)
    env.reference_pack={'geometry':[]};env.reference_geometry=[]
    return env


def test_each_seed_preserves_its_ik_draws_fluid_parameters_and_fill_heights():
    batch=task([101,17]);batch._initialize_episode(torch.tensor([0,1]),{})
    assert batch.velocity_rows==[0,1] and batch.rings==[0,1]
    assert [i for i,_ in batch.model_calls]==[0,1]
    for i,seed in enumerate((101,17)):
        single=task([seed]);single._initialize_episode(torch.tensor([0]),{})
        np.testing.assert_array_equal(batch.qposes[0][i],single.qposes[0][0])
        np.testing.assert_array_equal(batch._fill_heights[i],[single.h1,single.h2])
        assert batch._reset_ik_attempts[i]==single.reset_ik_attempts==2
        for group in ('source_poses','target_poses'):
            b=getattr(batch,group)[0].raw_pose[i].numpy();s=getattr(single,group)[0]
            np.testing.assert_array_equal(b,np.r_[s.p,s.q])
        b=batch.rebuilds[i][1];s=single.rebuilds[0][1]
        for field in ('mpm_particle_q','mpm_particle_qd','mpm_particle_volume','mpm_particle_mass',
                      'mpm_particle_mu_lam_ys','mpm_particle_friction_cohesion','mpm_particle_type'):
            np.testing.assert_array_equal(getattr(b,field),getattr(s,field))


def test_partial_recipe_changes_only_selected_seed_and_target():
    env=task([101,17]);env._fill_heights[0]=[.011,.015]
    env._initialize_episode(torch.tensor([1]),{})
    assert env.velocity_rows==[1] and env.rings==[1]
    assert [i for i,_,_ in env.rebuilds]==[1]
    assert [i for i,_ in env.model_calls]==[1]
    np.testing.assert_array_equal(env._fill_heights[0],[.011,.015])
    assert env.qposes[0].shape==(1,9)


def test_selected_fill_heights_restore_and_validation(monkeypatch):
    env=task([101,17]);env._target_height=.15;env._fill_heights[:]=[[.012,.016],[.018,.022]]
    calls=[];monkeypatch.setattr(MPMBaseEnv,'set_state_dict',lambda self,state,index:calls.append(index))
    state={'task':{'fill_heights':torch.tensor([[.025,.03]],dtype=torch.float64)}}
    env.set_state_dict(state,[1]);assert calls==[[1]] and env.rings==[1]
    np.testing.assert_array_equal(env._fill_heights,[[.012,.016],[.025,.03]])
    state['task']['fill_heights'][0,1]=.02
    with pytest.raises(ValueError,match='fill-height'):env.set_state_dict(state,[1])
    assert calls==[[1]]


def test_success_uses_own_beaker_heights_source_tilt_and_joint_velocity():
    env=task([101,17]);env._target_radius=.03;env._target_height=.15
    env._fill_heights[:]=[[.01,.014],[.02,.024]]
    env.source_bodies=[sapien.Pose(),sapien.Pose()];env.beaker_bodies=[sapien.Pose(),sapien.Pose([1,0,0])]
    env.rigid_pose=lambda body:body
    xs=[np.tile([0.,0.,.012],(101,1)),np.tile([1.,0.,.022],(101,1))]
    env._mpm_batch.couplers=[SimpleNamespace(particle_state=lambda x=x:{'x':x}) for x in xs]
    qvel=torch.zeros((2,9));env.agent.robot.get_qvel=lambda:qvel
    assert env.evaluate()['success'].tolist()==[True,True]
    qvel[1,0]=.1;assert env.evaluate()['success'].tolist()==[True,False]
    qvel.zero_();env.source_bodies[1]=sapien.Pose(q=[np.sqrt(.5),np.sqrt(.5),0,0])
    assert env.evaluate()['success'].tolist()==[True,False]


def test_grasp_uses_each_native_contact_row_and_finger_direction():
    env=task([101,17]);env.rigid_pose=lambda body:body
    links=[SimpleNamespace(_objs=[sapien.Pose(),sapien.Pose()]) for _ in range(2)]
    env.agent.finger1_link,env.agent.finger2_link=links
    def contacts(link,actor):
        sign=1 if link is links[0] else -1
        return torch.tensor([[0.,sign*.01,0.],[sign*.01,0.,0.]])
    env.scene=SimpleNamespace(get_pairwise_contact_impulses=contacts)
    assert env._check_grasp(0) and not env._check_grasp(1)

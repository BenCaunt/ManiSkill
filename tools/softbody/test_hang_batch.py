"""Hang seed, grasp and task contracts; native physics evidence is separate."""
from types import SimpleNamespace

import numpy as np
import pytest
import sapien
import torch

from mani_skill.envs.softbody.base_env import MPMBaseEnv
from mani_skill.envs.softbody.batch import MPMBatchRuntime
from mani_skill.envs.softbody.hang import HangEnv


def task(count=2):
    env=HangEnv.__new__(HangEnv);env.num_envs=count;env.device=torch.device('cpu')
    env._mpm_reset_active=True;env._mpm_initial_checkpoint=None
    env._mpm_batch=MPMBatchRuntime(env,count,128) if count>1 else None
    env._episode_seed=np.array([101,17][:count]);env._selected_indices=np.zeros((count,5),np.int64)
    env.rope_start_indices=np.zeros(count,np.int64);env.rod_poses=[];env.nominals=[]
    env.rod=SimpleNamespace(set_pose=lambda p:env.rod_poses.append(p))
    env.agent=SimpleNamespace(reset=lambda q:env.nominals.append(q.clone()),
        controller=SimpleNamespace(reset=lambda:None),robot=SimpleNamespace(name='panda',set_pose=lambda p:None))
    env.particles=np.column_stack([np.linspace(-.2,.2,101),np.zeros(101),np.full(101,.05)]).astype(np.float32)
    env.rope_starts={}
    for key,shape in {'x':(3,),'v':(3,),'F':(3,3),'C':(3,3),'vc':()}.items():
        env.rope_starts['mpm_'+key]=np.array([np.full((101,*shape),i+.125,np.float32) for i in range(4)])
    for key,width in [('root_pose',7),('root_vel',3),('root_qvel',3),('qpos',9),('qvel',9)]:
        env.rope_starts['robot_'+key]=np.array([np.full(width,i+.5,np.float32) for i in range(4)])
    def coupler():
        return SimpleNamespace(particle_state=lambda:{'x':env.particles.copy()},
                               model=SimpleNamespace(struct=SimpleNamespace(n_particles=101)))
    if count>1:env._mpm_batch.couplers=[coupler() for _ in range(count)]
    else:env.mpm_coupler=coupler()
    env.draws=[];env.rebuilds=[]
    def build(rng,index):
        env.draws.append(rng.rand(17));return object(),[index]
    env._hang_builder=build
    env.rebuild_mpm=lambda builder,bodies,env_idx=0:env.rebuilds.append(env_idx)
    def checkpoint():
        capacity=128 if count>1 else 101
        state={'mpm':{key:torch.zeros((count,capacity,*shape)) for key,shape in
                       {'x':(3,),'v':(3,),'F':(3,3),'C':(3,3),'vc':()}.items()},
               'articulations':{'panda':torch.zeros((count,31))},
               'task':{'selected_indices':torch.as_tensor(env._selected_indices.copy() if count>1 else env.selected_indices[None])}}
        if count>1:state['mpm_meta']={'count':torch.full((count,1),101),'mask':torch.arange(capacity)[None].repeat(count,1)<101}
        return state
    env.get_state_dict=checkpoint
    return env


def test_batch_seed_draws_rod_grasp_and_particle_selection_match_independent_n1():
    batch=task();batch._initialize_episode(torch.tensor([0,1]),{})
    saved=batch._mpm_initial_checkpoint
    for row,seed in enumerate((101,17)):
        single=task(1);single._episode_seed=np.array([seed]);single._initialize_episode(torch.tensor([0]),{})
        assert np.array_equal(batch.draws[row],single.draws[0])
        assert batch.rope_start_indices[row]==single.rope_start_index
        assert np.array_equal(batch._selected_indices[row],single.selected_indices)
        assert torch.equal(batch.nominals[0][row],single.nominals[0][0])
        assert np.array_equal(batch.rod_poses[0].raw_pose[row].numpy(),np.r_[single.rod_poses[0].p,single.rod_poses[0].q])
        for key,value in single._mpm_initial_checkpoint['mpm'].items():
            assert torch.equal(saved['mpm'][key][row,:101],value[0])
            assert not torch.count_nonzero(saved['mpm'][key][row,101:])
        assert torch.equal(saved['articulations']['panda'][row],single._mpm_initial_checkpoint['articulations']['panda'][0])
    assert batch.rebuilds==[0,1]


def test_partial_initialization_defers_only_selected_grasp():
    env=task();env._selected_indices[0]=[1,2,3,4,5];env.rope_start_indices[0]=3
    env._initialize_episode(torch.tensor([1]),{})
    assert env.rebuilds==[1] and env.rope_start_indices[0]==3
    assert env._selected_indices[0].tolist()==[1,2,3,4,5]
    state=env._mpm_initial_checkpoint
    assert state['mpm']['x'].shape==(1,128,3) and state['articulations']['panda'].shape==(1,31)
    assert state['task']['selected_indices'].tolist()==env._selected_indices[[1]].tolist()


def test_selected_task_restore_validates_indices_against_saved_particle_count(monkeypatch):
    env=task();env._selected_indices[0]=[1,2,3,4,5];calls=[]
    monkeypatch.setattr(MPMBaseEnv,'set_state_dict',lambda self,state,indices:calls.append(indices))
    state={'task':{'selected_indices':torch.tensor([[0,1,2,3,4]])},'mpm_meta':{'count':torch.tensor([[5]])}}
    env.set_state_dict(state,[1]);assert calls==[[1]] and env._selected_indices[1].tolist()==[0,1,2,3,4]
    state['task']['selected_indices'][0,-1]=5
    with pytest.raises(ValueError,match='evaluation particle indices'):env.set_state_dict(state,[1])
    assert calls==[[1]] and env._selected_indices[0].tolist()==[1,2,3,4,5]


def test_each_environment_uses_own_rope_rod_fingers_and_release_joint_state():
    env=task();env._selected_indices[:]=np.arange(5)
    x=np.array([[0.,-.06,.15],[0.,-.03,.19],[0.,0.,.23],[0.,.03,.19],[0.,.06,.15]],np.float32)
    env._mpm_batch.couplers=[SimpleNamespace(particle_state=lambda x=x+np.array([i,0,0]):{'x':x,'v':np.zeros_like(x)}) for i in (0,1)]
    env.hands=[sapien.Pose([i,0.,.23]) for i in (0,1)]
    env.rod_bodies=[sapien.Pose([i,0.,.2]) for i in (0,1)]
    env.leftfingers=[sapien.Pose([-.04,0.,.23]),sapien.Pose([.99,0.,.23])]
    env.rightfingers=[sapien.Pose([.04,0.,.23]),sapien.Pose([1.01,0.,.23])]
    env.rigid_pose=lambda body:body
    env.agent.robot._objs=[SimpleNamespace(get_qlimits=lambda:np.array([[-.04,.04]]*9)) for _ in (0,1)]
    q=torch.zeros((2,9));q[0,-2:]=.04;q[1,-2:]=.01;env.agent.robot.get_qpos=lambda:q
    assert env.evaluate()['success'].tolist()==[True,False]
    np.testing.assert_allclose(env.compute_dense_reward(None,None,None).numpy(),[6.,4.45],atol=1e-6)

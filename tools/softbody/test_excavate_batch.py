"""Per-environment task contracts; native terrain/physics evidence is separate."""
from types import SimpleNamespace

import numpy as np
import pytest
import sapien
import torch

from mani_skill.envs.softbody.base_env import MPMBaseEnv
from mani_skill.envs.softbody.batch import MPMBatchRuntime
from mani_skill.envs.softbody.excavate import ExcavateEnv


def task(count=2):
    env = ExcavateEnv.__new__(ExcavateEnv)
    env.num_envs = count; env.device = torch.device('cpu'); env._mpm_reset_active = True
    env._mpm_batch = MPMBatchRuntime(env,count,32768) if count > 1 else None
    env._target_nums = np.array([700]*count,dtype=np.int64)
    env._episode_seed = np.array([101,17][:count]); env.target_height = .2
    env.collision_geometry = [['old-'+str(i)] for i in range(count)]
    env.calls = []; env.qposes = []
    env.agent = SimpleNamespace(reset=lambda q: env.qposes.append(q.clone()), robot=SimpleNamespace(set_pose=lambda pose: None))
    def build(rng, index):
        env.calls.append((index,rng.rand(4).copy()))
        return object(), [index], ['new-'+str(index)]
    env._excavate_builder = build
    env.rebuilds = []
    env.rebuild_mpm = lambda builder,bodies,env_idx=0: env.rebuilds.append((env_idx,bodies))
    return env


def test_each_seed_preserves_single_environment_robot_and_rng_recipe():
    batch = task(); batch._initialize_episode(torch.tensor([0,1]),{})
    for index, seed in enumerate((101,17)):
        single = task(1); single._episode_seed = np.array([seed])
        single._initialize_episode(torch.tensor([0]),{})
        assert torch.equal(batch.qposes[0][index],single.qposes[0][0])
        assert np.array_equal(batch.calls[index][1],single.calls[0][1])
        assert batch._target_nums[index] == single.target_num
    assert batch.rebuilds == [(0,[0]),(1,[1])]


@pytest.mark.parametrize('override',[345,[345]])
def test_partial_initialization_preserves_untouched_target_and_geometry(override):
    env = task(); env._target_nums[:] = [1001,502]
    env._initialize_episode(torch.tensor([1]),{'target_num':override})
    assert env._target_nums.tolist() == [1001,345]
    assert env.rebuilds == [(1,[1])]
    assert env.collision_geometry == [['old-0'],['new-1']]


@pytest.mark.parametrize('value',[0,-1,.5,[1,2],float('nan')])
def test_invalid_selected_target_override_does_not_build_particles(value):
    env = task()
    with pytest.raises(ValueError,match='positive integer targets'):
        env._initialize_episode(torch.tensor([1]),{'target_num':value})
    assert not env.rebuilds


def test_task_counts_use_each_environments_particles_and_target():
    env = task(); env._target_nums[:] = [199,250]
    arrays = [np.tile([0.,0.,.25],(100,1)),np.r_[np.tile([0.,0.,.05],(250,1)),np.tile([.2,0.,.05],(30,1))]]
    env._mpm_batch.couplers = [SimpleNamespace(model=SimpleNamespace(struct=SimpleNamespace(n_particles=len(a))),
        particle_state=lambda a=a: {'x':a,'v':np.zeros_like(a)}) for a in arrays]
    assert env.n_particles.tolist() == [100,280]
    assert env._task_counts(0) == (100,0,True)
    assert env._task_counts(1) == (0,30,False)
    result = env.evaluate()
    assert result['success'].tolist() == [True,False]
    assert result['spilled_particles'].tolist() == [0,30]


def test_bucket_membership_uses_own_rigid_pose_and_collision_hull():
    env = task(); env.buckets = [object(),object()]
    env.rigid_pose = lambda body: sapien.Pose([0.,0.,0.]) if body is env.buckets[0] else sapien.Pose([1.,0.,0.])
    points = np.stack(np.meshgrid(np.linspace(-.04,.04,5),np.linspace(-.025,.025,5),np.linspace(.01,.09,5)),axis=-1).reshape(-1,3)
    env._mpm_batch.couplers = [SimpleNamespace(particle_state=lambda: {'x':points}),
                               SimpleNamespace(particle_state=lambda: {'x':points+[1.,0.,0.]})]
    hull = np.stack(np.meshgrid([-.05,.05],[-.04,.04],[0.,.1]),axis=-1).reshape(-1,3)
    narrow = hull.copy(); narrow[:,0] *= .5
    env.vertices_mats = [np.column_stack((v,np.ones(len(v)))) for v in (hull,narrow)]
    a,b = env.particles_inside_bucket(0),env.particles_inside_bucket(1)
    assert len(a) > len(b) > 0
    assert np.all(np.abs(b[:,0]-1.) <= .025)
    assert np.all(np.abs(a[:,0]) <= .05)


def test_checkpoint_restores_only_selected_task_target(monkeypatch):
    env = task(); env._target_nums[:] = [1001,502]
    forwarded = []
    monkeypatch.setattr(MPMBaseEnv,'set_state_dict',lambda self,state,env_idx: forwarded.append((state,env_idx)))
    env.set_state_dict({'task':{'target_num':torch.tensor([[777.]])},'physical':{}},[1])
    assert env._target_nums.tolist() == [1001,777]
    assert forwarded == [({'physical':{}},[1])]
    with pytest.raises(ValueError,match='Invalid target particle count'):
        env.set_state_dict({'task':{'target_num':torch.tensor([[1.5]])}},[0])
    assert env._target_nums.tolist() == [1001,777] and len(forwarded) == 1

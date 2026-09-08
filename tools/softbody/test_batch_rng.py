"""Use the real BaseEnv reset/RNG dispatch without a physics or rendering world."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.softbody.base_env import MPMBaseEnv
from mani_skill.envs.softbody.batch import MPMBatchRuntime


def rng_state(rng):
    kind,words,cursor,has_gauss,gaussian=rng.get_state()
    return kind,words.tobytes(),cursor,has_gauss,gaussian


def env_with_rng(count=3):
    env=MPMBaseEnv.__new__(MPMBaseEnv)
    env.num_envs=count;env.device=torch.device('cpu');env._mpm_reset_active=False
    env._mpm_batch=MPMBatchRuntime(env,count,4);env.mpm_gpu_world=None
    env._main_seed=None;env._episode_seed=np.zeros(count,dtype=np.int64)
    env._batched_episode_rng=None;env._enhanced_determinism=True
    env._batched_rng_backend='numpy:random_state'
    env._set_main_rng(np.arange(count)+101)
    env._set_episode_rng(np.arange(count)+101,torch.arange(count))
    env.reconfiguration_freq=0;env._reconfig_counter=1
    env._elapsed_steps=torch.zeros(count,dtype=torch.int32)
    env.scene=SimpleNamespace(_reset_mask=torch.ones(count,dtype=torch.bool),gpu_sim_enabled=False)
    env._sim_device=SimpleNamespace(is_cuda=lambda:False)
    env.agent=None;env._clear_sim_state=lambda:None
    env.get_info=lambda: {};env.get_obs=lambda info: env._episode_seed.copy()
    env.initialized=[]
    env._initialize_episode=lambda indices,options:env.initialized.append((indices.clone(),env._episode_seed.copy()))
    return env


@pytest.mark.parametrize('selected,seeds', [([1],[17]),([2,0],[71,19]),([0],91),([2,1],123)])
def test_partial_seed_assignment_preserves_unselected_draw_positions(selected,seeds):
    env=env_with_rng()
    for i in range(3):
        env._batched_main_rng[i].rand(7+i);env._batched_episode_rng[i].randn(3+i)
    env._main_rng.rand(4)
    before={name:[rng_state(getattr(env,name)[i]) for i in range(3)] for name in ('_batched_main_rng','_batched_episode_rng')}
    old_main_seed=env._main_seed.copy();old_episode_seed=env._episode_seed.copy();old_legacy=rng_state(env._main_rng)
    values=MPMBatchRuntime.seed_values(seeds,len(selected))
    env.reset(seed=seeds,options={'env_idx':torch.tensor(selected)})
    for i in range(3):
        if i in selected:
            seed=values[selected.index(i)]
            assert env._main_seed[i]==env._episode_seed[i]==seed
            assert rng_state(env._batched_main_rng[i])==rng_state(np.random.RandomState(seed))
            assert rng_state(env._batched_episode_rng[i])==rng_state(np.random.RandomState(seed))
        else:
            assert env._main_seed[i]==old_main_seed[i] and env._episode_seed[i]==old_episode_seed[i]
            for name in before:assert rng_state(getattr(env,name)[i])==before[name][i]
    if 0 not in selected:assert rng_state(env._main_rng)==old_legacy
    assert env._episode_rng is env._batched_episode_rng[0]
    assert env.initialized[-1][0].tolist()==selected


@pytest.mark.parametrize('enhanced',[False,True])
def test_unseeded_partial_reset_advances_only_selected_streams(enhanced):
    env=env_with_rng();env._enhanced_determinism=enhanced
    for i in range(3):env._batched_main_rng[i].rand(5);env._batched_episode_rng[i].randn(3)
    main=[copy.deepcopy(env._batched_main_rng[i]) for i in range(3)]
    episode=[rng_state(env._batched_episode_rng[i]) for i in range(3)]
    seeds=env._episode_seed.copy()
    expected_seed=main[1].randint(2**31) if enhanced else seeds[1]
    env.reset(options={'env_idx':torch.tensor([1])})
    assert env._episode_seed.tolist()==[seeds[0],expected_seed,seeds[2]]
    for i in range(3):
        assert rng_state(env._batched_main_rng[i])==rng_state(main[i])
        if i!=1 or not enhanced:assert rng_state(env._batched_episode_rng[i])==episode[i]


@pytest.mark.parametrize('seed',[17,[17],[101,17,1]])
def test_full_reset_retains_upstream_seed_and_draw_contract(seed):
    actual=env_with_rng();reference=env_with_rng()
    BaseEnv._set_main_rng(reference,seed);BaseEnv._set_episode_rng(reference,seed,torch.arange(3))
    actual.reset(seed=seed)
    assert np.array_equal(actual._main_seed,reference._main_seed)
    assert np.array_equal(actual._episode_seed,reference._episode_seed)
    for i in range(3):
        assert rng_state(actual._batched_main_rng[i])==rng_state(reference._batched_main_rng[i])
        assert rng_state(actual._batched_episode_rng[i])==rng_state(reference._batched_episode_rng[i])


def test_reversed_full_seed_list_follows_requested_environment_order():
    env=env_with_rng();env.reset(seed=[17,19,23],options={'env_idx':torch.tensor([2,0,1])})
    assert env._episode_seed.tolist()==env._main_seed.tolist()==[19,23,17]


@pytest.mark.parametrize('seed',[True,-1,2**32,1.5,[],[[1]],[1,2,3],['17']])
def test_invalid_partial_seeds_fail_before_rng_or_world_mutation(seed):
    env=env_with_rng();before=[rng_state(env._batched_main_rng[i]) for i in range(3)]
    with pytest.raises(ValueError,match='seed'):env.reset(seed=seed,options={'env_idx':torch.tensor([1])})
    assert not env.initialized and env._mpm_batch.reset_indices is None
    assert [rng_state(env._batched_main_rng[i]) for i in range(3)]==before


def test_original_partial_reseed_bug_is_exercised():
    env=env_with_rng()
    BaseEnv._set_main_rng(env,[17]);BaseEnv._set_episode_rng(env,[17],torch.tensor([1]))
    assert env._episode_seed.tolist()!=[101,17,103]
    assert env._episode_seed[1]!=17

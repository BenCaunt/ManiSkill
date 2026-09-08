"""CPU contract tests; genuine shared-world execution is checked on GPU."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mani_skill.envs.softbody.batch import MPMBatchRuntime
from mani_skill.envs.softbody.batch import BaseEnv


@pytest.fixture
def runtime():
    env = SimpleNamespace(device=torch.device('cpu'), _mpm_reset_active=False, mpm_gpu_world=None,
                          _reconfig_counter=1, reconfiguration_freq=0)
    return MPMBatchRuntime(env, 2, 4)


@pytest.mark.parametrize('indices', [[], [0, 0], [-1], [2], [0.5], [[0]]])
def test_reset_indices_reject_invalid_selection(runtime, indices):
    with pytest.raises(ValueError, match='indices'):
        runtime.indices(indices)


def test_partial_reset_does_not_revive_failed_neighbour(runtime):
    world = SimpleNamespace(pending_step=False, _active=False, failed=True)
    runtime.env.mpm_gpu_world = world
    with pytest.raises(RuntimeError, match='requires a full reset'):
        runtime.reset([17], {'env_idx': [0]})
    assert world.failed and runtime.reset_indices is None and not runtime.env._mpm_reset_active


def test_reset_cannot_interrupt_shared_physics_step(runtime):
    world = SimpleNamespace(pending_step=True, _active=False, failed=False)
    runtime.env.mpm_gpu_world = world
    with pytest.raises(RuntimeError, match='pending shared physics step'):
        runtime.reset([17, 18], {})
    assert world.pending_step and not world.failed


@pytest.mark.parametrize('automatic', [False, True])
def test_invalid_partial_reconfigure_does_not_poison_live_world(runtime, automatic):
    world = SimpleNamespace(pending_step=False, _active=False, failed=False)
    runtime.env.mpm_gpu_world = world
    if automatic:
        runtime.env._reconfig_counter = 0
        runtime.env.reconfiguration_freq = 1
    with pytest.raises(RuntimeError, match='reconfigure only part'):
        runtime.reset([17], {'env_idx': [0], 'reconfigure': not automatic})
    assert not world.failed and runtime.reset_indices is None and not runtime.env._mpm_reset_active


def test_clear_during_reconfiguration_preserves_selected_reset_indices(runtime):
    runtime.reset_indices = [0, 1]
    runtime.clear()
    assert runtime.reset_indices == [0, 1] and runtime.couplers == [None, None]


def test_particle_padding_exposes_real_counts_without_aliasing(runtime):
    arrays = [np.full((2, 3), 1., np.float32), np.full((3, 3), 2., np.float32)]
    runtime.couplers = [SimpleNamespace(model=SimpleNamespace(struct=SimpleNamespace(n_particles=len(a))),
                                      particle_state=lambda a=a: {'x': a}) for a in arrays]
    data = runtime.padded()['x']; meta = runtime.metadata()
    assert data.shape == (2, 4, 3) and meta['count'].tolist() == [[2], [3]]
    assert meta['mask'].tolist() == [[True, True, False, False], [True, True, True, False]]
    assert torch.equal(data[0, :2], torch.ones((2, 3))) and torch.equal(data[1, :3], torch.full((3, 3), 2.))
    assert not torch.count_nonzero(data[0, 2:]) and not torch.count_nonzero(data[1, 3:])
    data.fill_(99.)
    assert np.all(arrays[0] == 1.) and np.all(arrays[1] == 2.)


def test_invalid_count_is_rejected_before_building_a_model(runtime):
    runtime.env._mpm_reset_active = True; runtime.reset_indices = [0]
    with pytest.raises(ValueError, match='declared batch capacity'):
        runtime.rebuild(SimpleNamespace(mpm_particle_q=[0]*5), [], 0)


def test_checkpoint_cannot_target_a_nonselected_environment(runtime):
    runtime.env._mpm_reset_active = True; runtime.reset_indices = [0]
    with pytest.raises(ValueError, match='among the environments being reset'):
        runtime.restore({}, [1])


def test_checkpoint_mask_must_match_declared_live_count(runtime):
    runtime.env._mpm_reset_active = True; runtime.reset_indices = [0]
    state = {'mpm_meta': {'count': [[2]], 'mask': [[True, False, False, False]]}}
    with pytest.raises(ValueError, match='live prefix'):
        runtime.restore(state, [0])


@pytest.mark.parametrize('explicit', [False, True])
def test_deferred_grasp_follows_controller_reset_and_explicit_checkpoint_wins(runtime, monkeypatch, explicit):
    env = runtime.env; events = []
    pending, requested = {'grasp':torch.ones((1,3))}, {'saved':torch.zeros((1,3))}
    env._mpm_initial_checkpoint = {'stale':True}
    def base_reset(self, *, seed, options):
        assert self._mpm_initial_checkpoint is None and self._mpm_reset_active
        events.append('controller reset'); self._mpm_initial_checkpoint = pending
        return 'fresh', {'reconfigure':False}
    monkeypatch.setattr(BaseEnv,'reset',base_reset)
    env.set_state_dict = lambda state, selected: events.append((state, selected))
    env.get_info = lambda: {'success':False}
    env.get_obs = lambda info: 'restored'
    options={'env_idx':[1]}
    if explicit: options['reset_to_env_states']={'env_states':requested}
    obs,info = runtime.reset([17],options)
    assert events[0]=='controller reset' and events[1][0] is (requested if explicit else pending)
    assert events[1][1]==[1] and obs==env._last_obs=='restored'
    assert not env._mpm_reset_active and env._mpm_initial_checkpoint is None and runtime.reset_indices is None


def test_failed_deferred_restore_clears_pending_grasp_and_poisoned_world(runtime,monkeypatch):
    env=runtime.env; coupler=SimpleNamespace(failed=False)
    env.mpm_gpu_world=SimpleNamespace(pending_step=False,_active=False,failed=False,couplers=[coupler])
    def base_reset(self,**kwargs):
        self._mpm_initial_checkpoint={'grasp':True}; return None,{'reconfigure':False}
    monkeypatch.setattr(BaseEnv,'reset',base_reset)
    def reject(*args):raise ValueError('bad grasp')
    env.set_state_dict=reject
    with pytest.raises(ValueError,match='bad grasp'):runtime.reset([101,17],{})
    assert env.mpm_gpu_world.failed and coupler.failed
    assert env._mpm_initial_checkpoint is None and not env._mpm_reset_active and runtime.reset_indices is None

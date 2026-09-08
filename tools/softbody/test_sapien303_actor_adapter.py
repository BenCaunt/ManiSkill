"""Python selection plumbing; native type/index/isolation evidence is separate."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from mani_skill.utils import sapien303 as adapter


def test_native_selection_uses_gpu_indices_and_requested_order(monkeypatch):
    bodies=[SimpleNamespace(gpu_index=i) for i in (4,1,3,0,2)]
    source=torch.arange(65,dtype=torch.float32).reshape(5,13)
    system=SimpleNamespace(rigid_dynamic_components=bodies,cuda_rigid_dynamic_data=source)
    calls=[]
    monkeypatch.setattr(adapter,'version',lambda _: '3.0.3')
    monkeypatch.setattr(adapter,'import_module',lambda _:SimpleNamespace(apply_actors=lambda *args:calls.append(args)))
    adapter.apply_selected_actor_data(system,torch.tensor([3,1],dtype=torch.int32))
    px,selected,rows=calls.pop()
    assert px is system and selected==[bodies[2],bodies[1]]
    np.testing.assert_array_equal(rows,source[[3,1]].numpy())
    rows[:]=0
    assert source[3,0]==39


def test_missing_native_module_fails_before_assignment(monkeypatch):
    monkeypatch.setattr(adapter,'version',lambda _: '3.0.3')
    def absent(name):raise ModuleNotFoundError(name=name)
    monkeypatch.setattr(adapter,'import_module',absent)
    with pytest.raises(RuntimeError,match='native actor compatibility module'):
        adapter.apply_selected_actor_data(object(),torch.tensor([1],dtype=torch.int32))


@pytest.mark.parametrize('indices',[[1,1],[7]])
def test_invalid_selection_never_reaches_native_apply(monkeypatch,indices):
    monkeypatch.setattr(adapter,'version',lambda _: '3.0.3')
    monkeypatch.setattr(adapter,'import_module',lambda _:object())
    system=SimpleNamespace(rigid_dynamic_components=[SimpleNamespace(gpu_index=1)])
    with pytest.raises(ValueError,match='indices'):
        adapter.apply_selected_actor_data(system,torch.tensor(indices,dtype=torch.int32))

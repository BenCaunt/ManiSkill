"""Selected rigid-actor updates for the pinned SAPIEN 3.0.3 GPU backend."""
from importlib import import_module
from importlib.metadata import version

import torch


def apply_selected_actor_data(system, indices):
    """Apply only selected fetched rows, with native compact source indices.

    SAPIEN 3.0.3's indexed apply retains full-buffer source indices after
    compacting its data. Its full apply also rounds untouched nonidentity-COM
    actors. The small native compatibility module avoids both problems.
    This function synchronizes the selected CPU staging copy and native apply;
    it is used for reset/checkpoint assignment, not simulation stepping.
    """
    if version('sapien') != '3.0.3':
        # Keep the existing path for backends not covered by this native fix.
        system.gpu_apply_rigid_dynamic_data()
        return
    try:
        bridge = import_module('sapien303_actor_bridge')
    except ModuleNotFoundError as exc:
        if exc.name != 'sapien303_actor_bridge':
            raise
        raise RuntimeError(
            'Partial actor reset on SAPIEN 3.0.3 requires the native actor '
            'compatibility module; see tools/softbody/native/README.md.'
        ) from exc
    selected = indices.detach().cpu().tolist()
    by_index = {body.gpu_index: body for body in system.rigid_dynamic_components}
    if len(selected) != len(set(selected)) or any(i not in by_index for i in selected):
        raise ValueError('Expected unique initialized native rigid-actor indices')
    actors = [by_index[i] for i in selected]
    data = torch.as_tensor(system.cuda_rigid_dynamic_data, device=indices.device)
    rows = data[indices.long()].detach().cpu().numpy().copy()
    bridge.apply_actors(system, actors, rows)


def apply_selected_articulation_data(system, indices, *, targets=False):
    """Apply selected native joint rows without dirtying neighbouring link state.

    Called only by the reset/checkpoint path. PhysX updates link kinematics for
    articulations marked dirty, so resending unchanged joint positions still
    recomputes untouched link transforms and velocities.
    """
    selected = indices.detach().cpu().tolist()
    if not selected:
        return
    if version('sapien') != '3.0.3':
        raise RuntimeError('Selected articulation reset is verified only for SAPIEN 3.0.3')
    try:
        bridge = import_module('sapien303_actor_bridge')
    except ModuleNotFoundError as exc:
        if exc.name != 'sapien303_actor_bridge':
            raise
        raise RuntimeError('Selected articulation reset requires the native compatibility module') from exc
    if not hasattr(bridge, 'apply_articulation_data'):
        raise RuntimeError('Rebuild the native compatibility module with selected articulation support')
    bridge.apply_articulation_data(system, selected, targets)

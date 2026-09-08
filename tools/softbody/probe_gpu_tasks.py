"""Exercise actual legacy task APIs, physical stepping, and checkpoint replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import sapien


def run(args):
    if args.backend == 'gpu':
        sapien.physx.enable_gpu()
    sys.path.insert(0, str(args.source))
    from mani_skill.envs.softbody.capture import CaptureAdapter
    kwargs = dict(sim_backend='physx_cuda' if args.backend == 'gpu' else 'physx_cpu', render_backend='gpu')
    options = {}
    if args.task in ('Write', 'Pinch'):
        kwargs['level_dir'] = '/levels'
        options['level_file'] = 'line.h5' if args.task == 'Write' else 'squeeze_y.h5'
    adapter = CaptureAdapter(args.task + '-v0', control_mode='pd_joint_target_delta_pos', env_kwargs=kwargs)
    env = adapter.env
    try:
        initial = adapter.reset(seed=101, reset_kwargs=dict(options=options))
        checkpoint = env.get_state_dict()
        random = np.random.default_rng(1043)
        actions = random.uniform(-.025, .025, (3, *env.single_action_space.shape)).astype(np.float32)
        trajectories = []
        infos = []
        for trial in range(2):
            if trial:
                env.reset(seed=101, options={**options, 'reset_to_env_states': dict(env_states=checkpoint)})
                initial = adapter.snapshot()
            frames = [initial]; outcomes = []
            for action in actions:
                outcomes.append(adapter.step(action))
                frames.append(adapter.snapshot())
            arrays = {key: np.stack([frame[key] for frame in frames]) for key in initial if key != 'sim_state'}
            if any(not np.isfinite(value).all() for value in arrays.values()):
                raise RuntimeError('Nonfinite task trajectory')
            name = f'trial-{trial}.npz'
            np.savez_compressed(args.output / name, **arrays)
            trajectories.append(dict(path=name, sha256=hashlib.sha256((args.output/name).read_bytes()).hexdigest()))
            infos.append(outcomes)
        try:
            env.set_state_dict(checkpoint)
        except RuntimeError:
            protected = True
        else:
            raise AssertionError('Checkpoint assignment outside reset was accepted')
        report = dict(task=args.task, backend=args.backend, physics_system=type(env.scene.px).__name__,
            control_mode=adapter.control_mode, actions=actions.tolist(), trajectories=trajectories, outcomes=infos,
            particle_count=int(env.mpm_coupler.model.struct.n_particles), mpm_substeps=env.mpm_coupler.substeps,
            rigid_dt=float(env.scene.px.timestep), control_dt=float(env.control_timestep),
            world_steps=env.mpm_gpu_world.steps if env.gpu_sim_enabled else None,
            reset_guard=protected, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope='Native task API, physical force stepping and same-backend checkpoint replay. Three actions; not full-task success, benchmark parity or batching.')
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k: v for k,v in report.items() if k not in ('actions','outcomes')}, indent=2))
    finally:
        adapter.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['cpu','gpu'], required=True)
    parser.add_argument('--task', choices=['Fill','Excavate','Hang','Pour','Write','Pinch'], required=True)
    run(parser.parse_args())

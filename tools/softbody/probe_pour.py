"""Exercise native Pour fluid state, open cooked cavity and checkpoint replay."""
import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
import torch
from scipy.spatial import ConvexHull

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from mani_skill.envs.softbody.capture import CaptureAdapter
from mani_skill.envs.softbody.mpm import wp

def run(args):
    wp.config.kernel_cache_dir = str(args.output.parent/'warp-cache')
    adapter = CaptureAdapter('Pour-v0',control_mode='pd_joint_delta_pos',env_kwargs={'render_backend':'gpu'})
    env = adapter.env
    try:
        initial = adapter.reset(seed=args.seed,reset_kwargs={})
        np.savez_compressed(args.output/'000000.npz',**initial)
        initial_ik_attempts = env.reset_ik_attempts
        n = len(initial['x'])
        assert 'vol' in initial and 'vc' in initial
        assert np.all(initial['vol']>0)
        assert n==8534 if args.seed==101 else n>0
        assert abs(float(initial['mass'].sum())-.2456702888)<1e-7 if args.seed==101 else True
        z = np.linspace(.012,.17,200)
        angles = np.linspace(0,2*np.pi,16,endpoint=False)
        points = np.concatenate([np.c_[np.full_like(z,r*np.cos(a)),np.full_like(z,r*np.sin(a)),z]
            for r in (0.,.003,.006) for a in angles])
        for shape in env.source_body.get_collision_shapes():
            local = shape.local_pose.to_transformation_matrix()
            vertices = shape.vertices*shape.scale @ local[:3,:3].T+local[:3,3]
            eq = ConvexHull(vertices).equations
            assert not np.any(np.all(points@eq[:,:3].T+eq[:,3]<=1e-8,axis=1)), 'Native cooked hull obstructs the cavity'
        def picture(name):
            value = env.render_rgb_array().cpu().numpy()
            imageio.imwrite(args.output/name,value[0] if value.ndim==4 else value)
        picture('initial.png')
        records = []
        for i in range(args.steps):
            metrics = adapter.step(np.zeros(8,dtype=np.float32))
            state = adapter.snapshot()
            assert all(np.isfinite(v).all() for v in state.values())
            assert len(state['x'])==n and np.array_equal(state['mass'],initial['mass'])
            np.savez_compressed(args.output/f'{i+1:06}.npz',**state)
            records.append(dict(step=i+1,reward=metrics['reward'],success=bool(metrics['success'])))
        picture('final.png')
        checkpoint,flat = env.get_state_dict(),env.get_state().clone()
        current = checkpoint['mpm']['vol']
        rest = checkpoint['mpm_material']['particle_vol']
        difference = float(torch.max(torch.abs(current-rest)))
        assert difference>1e-12, 'Checkpoint must exercise compressed or expanded fluid'
        action = np.array([.01,-.02,.01,.02,0.,-.01,0.,-.5],dtype=np.float32)
        env.step(action)
        expected = adapter.snapshot()
        trials = []
        other_seed = 1 if args.seed==101 else 101
        env.reset(seed=other_seed)
        other_count = env.mpm_coupler.model.struct.n_particles
        assert other_count!=n, 'Checkpoint probe must exercise a changed particle count'
        for kind,saved in [('dict',checkpoint),('flat',flat),('reconfigure',checkpoint)]:
            env.reset(seed=other_seed,options={'reconfigure':kind=='reconfigure','reset_to_env_states':{'env_states':saved}})
            restored = env.get_state_dict()
            for key in ('mpm','mpm_material','mpm_drives','task'):
                assert all(torch.equal(restored[key][k],v) for k,v in checkpoint[key].items()), f'{kind}: {key}'
            env.step(action)
            actual = adapter.snapshot()
            trials.append(dict(kind=kind,reset_seed=other_seed,reset_particle_count=other_count,
                restored_particle_count=len(actual['x']),particle_replay_max_abs_m=float(np.max(abs(actual['x']-expected['x']))),
                current_volume_replay_max_abs_m3=float(np.max(abs(actual['vol']-expected['vol']))),
                qpos_replay_max_abs=float(np.max(abs(actual['qpos']-expected['qpos'])))))
            assert trials[-1]['qpos_replay_max_abs']<1e-4, f'{kind}: unstable joint replay'
        invalid = env.get_state_dict()
        invalid['mpm']['vol'][0,0] = -1.
        try:
            env.reset(seed=args.seed,options={'reset_to_env_states':{'env_states':invalid}})
        except ValueError as exc:
            assert 'volum' in str(exc)
        else:
            raise AssertionError('Negative fluid volume was accepted')
        return dict(completed=True,parity_validated=False,env_id='Pour-v0',seed=args.seed,steps=args.steps,
            particle_count=n,particle_mass_kg=float(initial['mass'].sum()),initial_ik_attempts=initial_ik_attempts,
            initial_qpos=initial['qpos'].tolist(),initial_task_state=initial['task_state'].tolist(),
            cooked_cavity_clear=True,cavity_probe_radius_m=.006,rigid_collision_hulls=len(env.source_body.get_collision_shapes()),
            current_vs_rest_volume_max_abs_m3=difference,negative_volume_rejected=True,checkpoint_trials=trials,records=records,
            collision_scope='Open rigid cavity replaces the legacy closed hull; exact original fluid SDF and inertial inputs retained')
    finally:
        adapter.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=101)
    parser.add_argument('--steps',type=int,default=20)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report={'completed':False,'parity_validated':False}
    try:
        report=run(args)
    except BaseException as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))

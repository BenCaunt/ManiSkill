"""Verify count-changing reset checkpoints against pressureless free fall."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from probe_environment import BallisticEnv, wp


def run(device, output):
    wp.config.kernel_cache_dir = str(output.parent/'warp-cache')
    env = BallisticEnv(mpm_device=device, robot_uids='none', obs_mode='none',
                       reward_mode='none', render_backend='none')
    results=[]
    try:
        for saved_dimension, reset_dimension in ((2, 3), (3, 2)):
            env.diagnostic_dimension=saved_dimension
            env.reset(seed=42)
            env.step(None)
            checkpoint, flat = env.get_state_dict(), env.get_state().clone()
            env.step(None)
            expected=env.get_state_dict()['mpm']
            for kind, saved, reconfigure in (('dict', checkpoint, False), ('flat', flat, False),
                                              ('reconfigure', checkpoint, True)):
                env.diagnostic_dimension=reset_dimension
                env.reset(seed=17)
                assert env.mpm_coupler.model.struct.n_particles==(reset_dimension+1)**3
                env.reset(seed=17, options={'reconfigure':reconfigure, 'reset_to_env_states':{'env_states':saved}})
                actual=env.get_state_dict()
                for group in ('mpm', 'mpm_material'):
                    assert all(torch.equal(v, actual[group][key]) for key,v in checkpoint[group].items())
                env.step(None)
                difference=float(torch.max(abs(env.get_state_dict()['mpm']['x']-expected['x'])))
                assert difference < 3e-6
                results.append(dict(kind=kind, saved_particles=(saved_dimension+1)**3,
                    reset_particles=(reset_dimension+1)**3, replay_error_m=difference))
        try:
            env.reset(options={'reset_to_env_states':{'env_states':flat[:,:-1]}})
        except ValueError as exc:
            assert 'layout' in str(exc)
        else:
            raise AssertionError('Misaligned flat particle state accepted')
        bad=env.get_state_dict()
        bad['mpm_material']['particle_type'][0,0]=3
        try:
            env.reset(options={'reset_to_env_states':{'env_states':bad}})
        except ValueError as exc:
            assert 'types' in str(exc)
        else:
            raise AssertionError('Unknown constitutive material type accepted')
        return dict(passed=True, device=device, trials=results, invalid_layout_rejected=True,
                    invalid_material_type_rejected=True,
                    scope='Uniform-color particles; fixed rigid/task topology, same material state layout')
    finally:
        env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu','cuda'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report={'passed':False}
    try:
        report=run(args.device, args.output)
    except BaseException as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))

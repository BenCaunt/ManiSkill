"""Check fixed observation spaces across different physical fluid counts."""
import argparse
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from mani_skill.envs.softbody.pour import PourEnv
from mani_skill.envs.softbody.mpm import wp
from mani_skill.utils.common import to_numpy

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
wp.config.kernel_cache_dir=str(args.output.parent/'warp-cache')
report={'completed':False,'trials':[]}
try:
    for mode in ('state_dict','state'):
        env=PourEnv(obs_mode=mode,control_mode='pd_joint_delta_pos',render_backend='gpu')
        try:
            counts=[]
            for seed in (1,101):
                obs,_=env.reset(seed=seed)
                n=env.mpm_coupler.model.struct.n_particles
                counts.append(n)
                arrays=to_numpy(obs)
                def mismatches(space,value,path=''):
                    if isinstance(value,dict):
                        return [item for k,v in value.items() for item in mismatches(space[k],v,path+'.'+k)]
                    return [] if space.contains(value) else [dict(path=path,shape=list(value.shape),dtype=str(value.dtype),space=str(space))]
                failures=mismatches(env.observation_space,arrays)
                assert not failures,f'{mode} seed {seed}: {failures}'
                assert env.get_state_dict()['mpm']['x'].shape==(1,n,3)
                if mode=='state_dict':
                    assert int(obs['extra']['particle_count'][0,0])==n
                    for key,value in obs['extra']['mpm'].items():
                        assert value.shape[1]==env.observation_particle_capacity
                        assert torch.all(value[:,n:]==0)
                report['trials'].append(dict(mode=mode,seed=seed,live_particles=n,space_contains_observation=True))
            assert counts[0]!=counts[1], 'Test must exercise a change in live particle count'
        finally:
            env.close()
    report['completed']=True
finally:
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

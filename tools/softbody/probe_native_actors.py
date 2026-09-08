"""Zero-step native actor-reset diagnostic, not a manipulation/parity claim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
import numpy as np
import sapien
import torch

def run(args):
    out=args.output; out.mkdir(exist_ok=True)
    report=dict(case=args.case,complete=False,scope='Native actor reset only; no task parity claim',files={})
    env=None
    try:
        sapien.physx.enable_gpu()
        sys.path.insert(0,str(args.source));sys.path.insert(0,str(args.extension))
        from mani_skill.envs.softbody.pour import PourEnv
        import sapien303_actor_bridge as bridge
        class DiagnosticPour(PourEnv):
            def _setup_scene(self):
                super()._setup_scene()
                if args.case.endswith('zero-origin'):
                    for scene in self.scene.sub_scenes:
                        self.scene.px.set_scene_offset(scene,[0.,0.,0.])
            def _load_scene(self,options):
                super()._load_scene(options)
                if args.case.startswith('identity-com'):
                    for body in self.source_bodies:
                        body.cmass_local_pose=sapien.Pose()
        env=DiagnosticPour(num_envs=2,sim_backend='physx_cuda',obs_mode='state_dict',
            control_mode='pd_ee_delta_pose_align',mpm_batch_particle_capacity=16384)
        env.reset(seed=[101,17])
        px=env.scene.px
        actors=[env.source_bodies[0],env.source_bodies[1],env.beaker_bodies[0],env.beaker_bodies[1]]
        indices=[body.gpu_index for body in actors]
        report['metadata']=bridge.describe(px,actors)
        report['python_gpu_indices']=indices
        def fetched():
            px.gpu_fetch_rigid_dynamic_data(); torch.cuda.synchronize()
            value=torch.as_tensor(px.cuda_rigid_dynamic_data,device='cuda')
            return value[indices].cpu().numpy().copy()
        arrays={}
        def save(label,value): arrays[label]=np.asarray(value).copy()
        save('initial_public',fetched());save('initial_bridge',bridge.read_actors(px,actors))
        with np.load(args.poses) as z:
            wanted=z['state/actors/bottle'].astype(np.float32)
        base=fetched();base[:2]=wanted
        save('initial_requested',base)
        bridge.apply_actors(px,actors,np.ascontiguousarray(base))
        save('assigned',fetched())
        # A selection has different order/layout than the native full buffer.
        selection=[3,1]
        start=fetched();requested=start[selection].copy()
        requested[:,0]+=[.0125,-.0175]
        requested[:,7:10]=[[.13,.24,.35],[-.16,-.27,-.38]]
        save('selection',selection);save('selected_before',start);save('selected_requested',requested)
        bridge.apply_actors(px,[actors[i] for i in selection],requested)
        save('selected_after',fetched());save('selected_bridge',bridge.read_actors(px,actors))
        # Repeated zero-step read -> apply exposes native float round trips.
        repeated=[fetched()]
        for _ in range(20):
            value=fetched()[[1]].copy()
            bridge.apply_actors(px,[actors[1]],value)
            repeated.append(fetched())
        save('selected_roundtrips',repeated)
        # The built-in full apply resends every actor's current global pose.
        full=[fetched()]
        for _ in range(20):
            px.gpu_apply_rigid_dynamic_data();torch.cuda.synchronize()
            full.append(fetched())
        save('full_roundtrips',full)
        before=fetched();bad=[]
        invalid_calls=[('wrong_system',lambda:bridge.describe(object(),actors)),
            ('wrong_actor',lambda:bridge.describe(px,[object()])),
            ('duplicate',lambda:bridge.describe(px,[actors[0],actors[0]])),
            ('wrong_shape',lambda:bridge.apply_actors(px,[actors[0]],np.zeros((1,12),np.float32))),
            ('wrong_dtype',lambda:bridge.apply_actors(px,[actors[0]],before[[0]].astype(np.float64))),
            ('nonfinite',lambda:bridge.apply_actors(px,[actors[0]],np.full((1,13),np.nan,np.float32))),
            ('nonunit',lambda:bridge.apply_actors(px,[actors[0]],np.zeros((1,13),np.float32)))]
        for name,call in invalid_calls:
            try: call()
            except Exception as exc: bad.append(dict(name=name,rejected=True,error=type(exc).__name__))
            else: bad.append(dict(name=name,rejected=False))
        bridge.apply_actors(px,[],np.empty((0,13),np.float32))
        save('invalid_before',before);save('invalid_after',fetched());report['invalid_calls']=bad
        report['explicit_steps_after_reset']=0
        np.savez_compressed(out/'actors.npz',**arrays)
        report['files']['actors.npz']=hashlib.sha256((out/'actors.npz').read_bytes()).hexdigest()
        report['complete']=True
    except Exception:
        report['error']=traceback.format_exc();raise
    finally:
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if env is not None: env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--extension',type=Path,required=True);p.add_argument('--poses',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=['native-com','identity-com','identity-com-zero-origin'],required=True)
    run(p.parse_args())

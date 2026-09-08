"""Exercise native Pinch inputs, physical steps, camera goals and checkpoints."""
import argparse
import copy
import json
from pathlib import Path
import sys
import imageio.v2 as imageio
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from mani_skill.envs.softbody.pinch import PinchEnv
from mani_skill.envs.softbody.mpm import wp


def run(output):
    wp.config.kernel_cache_dir=str(output.parent/'warp-cache')
    env=PinchEnv(level_dir='/levels',control_mode='pd_joint_target_delta_pos',obs_mode='state_dict',render_backend='gpu')
    try:
        obs,info=env.reset(seed=101,options={'level_file':'squeeze_y.h5'})
        source=np.load('/levels/squeeze_y.npz',allow_pickle=False)
        def snapshot():
            state=env.mpm_coupler.particle_state()
            state.update(qpos=env.agent.robot._objs[0].qpos.copy(),mass=env.mpm_coupler.model.struct.particle_mass.numpy().copy(),
                goal=env.goal_particle_points.copy(),goal_points_observation=env.goal_points.copy(),
                chamfer=np.asarray(env._compute_chamfer()),progress=np.asarray(env.evaluate()['progress'].cpu()))
            return state
        initial=snapshot();n=len(initial['x'])
        for key in ('x','v','F','C','vc','qpos','goal'):
            assert np.array_equal(initial[key],source[key]),key
        assert np.array_equal(initial['mass'],source['particle_mass'])
        projection_error=float(np.max(abs(initial['goal_points_observation']-source['goal_points_observation'])))
        assert projection_error<1e-7,projection_error
        assert abs(float(info['progress'][0]))<1e-12
        assert np.array_equal(obs['extra']['target_depth'][0].cpu(),source['goal_depths'])
        assert np.array_equal(obs['extra']['target_rgb'][0].cpu(),source['goal_rgbs'])
        np.savez_compressed(output/'initial.npz',**initial)
        def picture(name):
            value=env.render_rgb_array().cpu().numpy();imageio.imwrite(output/name,value[0] if value.ndim==4 else value)
        picture('initial.png')
        camera=env.scene.human_render_cameras['render_camera']
        mask=np.isin(camera.get_obs()['segmentation'].cpu().numpy(),[e.per_scene_id for e in env._particle_entities])
        assert np.count_nonzero(mask)>100
        records=[]
        action=np.array([.01,-.02,.01,.02,-.01,.01,.02,1.],dtype=np.float32)
        for step in range(20):
            _,reward,_,_,info=env.step(action if step<4 else np.r_[np.zeros(7),1.].astype(np.float32))
            state=snapshot()
            assert all(np.isfinite(v).all() for v in state.values())
            assert np.array_equal(state['mass'],initial['mass'])
            records.append(dict(step=step+1,reward=float(reward[0]),progress=float(info['progress'][0]),success=bool(info['success'][0])))
        np.savez_compressed(output/'final.npz',**state);picture('final.png')
        checkpoint,flat=env.get_state_dict(),env.get_state().clone()
        follow=action.copy();follow[:7]*=-1
        env.step(follow);expected=snapshot()
        trials=[]
        for kind,saved in [('dict',checkpoint),('flat',flat),('reconfigure',checkpoint)]:
            env.reset(seed=17,options={'level_file':'shear_x.h5','reconfigure':kind=='reconfigure'})
            assert not np.array_equal(env.goal_particle_points,initial['goal'])
            env.reset(seed=17,options={'level_file':'shear_x.h5','reset_to_env_states':{'env_states':saved}})
            restored=env.get_state_dict()
            for group in ('mpm','mpm_material','mpm_drives','task','task_particles'):
                assert all(torch.equal(v,restored[group][k]) for k,v in checkpoint[group].items()),group
            assert torch.equal(checkpoint['controller']['arm']['target_qpos'],restored['controller']['arm']['target_qpos'])
            env.step(follow);actual=snapshot()
            particle_error=float(np.max(abs(actual['x']-expected['x'])));joint_error=float(np.max(abs(actual['qpos']-expected['qpos'])))
            assert particle_error<1e-5 and joint_error<1e-4,(particle_error,joint_error)
            trials.append(dict(kind=kind,particle_error_m=particle_error,joint_error_rad=joint_error))
        # Isolated serialization test, not a task recipe or physical success run.
        # Restore a smaller saved particle/goal topology, then restore the full
        # checkpoint from that different topology via both supported formats.
        reduced=copy.deepcopy(checkpoint)
        for group in env.PARTICLE_CHECKPOINT_GROUPS:
            reduced[group]={k:v[:,::2].clone() for k,v in reduced[group].items()}
        for kind,saved in [('dict',checkpoint),('flat',flat)]:
            env.reset(seed=101,options={'level_file':'squeeze_y.h5','reset_to_env_states':{'env_states':reduced}})
            assert len(env.goal_particle_points)==n//2 and env.mpm_coupler.model.struct.n_particles==n//2
            smallflat=env.get_state().clone()
            env.reset(seed=101,options={'level_file':'squeeze_y.h5','reset_to_env_states':{'env_states':smallflat}})
            assert len(env.goal_particle_points)==n//2
            env.reset(seed=101,options={'level_file':'squeeze_y.h5','reset_to_env_states':{'env_states':saved}})
            restored=env.get_state_dict()
            for group in env.PARTICLE_CHECKPOINT_GROUPS:
                assert all(torch.equal(v,restored[group][k]) for k,v in checkpoint[group].items()),group
        return dict(completed=True,parity_validated=False,official_levels_verified=False,env_id='Pinch-v0',seed=101,
            particle_count=n,mass_kg=float(initial['mass'].sum()),initial_inputs_exact=True,projection_error_m=projection_error,
            particle_pixels=int(np.count_nonzero(mask)),records=records,checkpoint_trials=trials,count_changing_checkpoint=True)
    finally:env.close()


def observations():
    reports=[]
    for mode,level,seed in [('state','shear_x.h5',17),('state_dict','squeeze_x.h5',1)]:
        env=PinchEnv(level_dir='/levels',control_mode='pd_ee_target_delta_pose',obs_mode=mode,render_backend='gpu')
        try:
            obs,_=env.reset(seed=seed,options={'level_file':level})
            for _ in range(2):
                def numpy_tree(v):
                    if isinstance(v,dict):return {k:numpy_tree(x) for k,x in v.items()}
                    data=v.detach().cpu().numpy();assert np.isfinite(data).all();return data
                assert env.observation_space.contains(numpy_tree(obs)),(mode,level,env.observation_space)
                obs,*_=env.step(np.array([.001,-.001,-.001,0,0,0,1.],dtype=np.float32))
            reports.append(dict(mode=mode,level=level,seed=seed,passed=True))
        finally:env.close()
    return reports

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    report={'completed':False}
    try:
        report=run(a.output);report['observation_modes']=observations()
    except BaseException as e:
        report['error']=f'{type(e).__name__}: {e}';raise
    finally:
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

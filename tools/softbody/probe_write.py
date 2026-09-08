"""Exercise Write physics, goal rasterization, camera pixels and reset checkpoints."""
import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from mani_skill.envs.softbody.write import WriteEnv
from mani_skill.envs.softbody.mpm import wp


def run(output):
    wp.config.kernel_cache_dir=str(output.parent/'warp-cache')
    env=WriteEnv(level_dir='/levels',control_mode='pd_joint_target_delta_pos',obs_mode='state_dict',render_backend='gpu')
    try:
        obs,_=env.reset(seed=101,options={'level_file':'line.h5'})
        def snapshot():
            model=env.mpm_coupler.model.struct
            return {**env.mpm_coupler.particle_state(), 'qpos':env.agent.robot._objs[0].qpos.copy(),
                'mass':model.particle_mass.numpy()[:model.n_particles].copy(),
                'goal':env.goal_points.copy(),'goal_image':env.goal_image.numpy(),
                'current_image':env.current_image.numpy()}
        initial=snapshot();n=len(initial['x'])
        assert n==19404 and abs(float(initial['mass'].sum())-7.2764997482299805)<1e-6
        assert obs['extra']['goal'].shape==(1,64,64)
        assert torch.equal(obs['extra']['goal'],torch.as_tensor(np.clip(initial['goal_image'][:,::-1],0,255).copy(),dtype=torch.uint8)[None])
        np.savez_compressed(output/'initial.npz',**initial)
        def picture(name):
            value=env.render_rgb_array().cpu().numpy()
            imageio.imwrite(output/name,value[0] if value.ndim==4 else value)
        picture('initial.png')
        camera=env.scene.human_render_cameras['render_camera']
        mask=np.isin(camera.get_obs()['segmentation'].cpu().numpy(),[e.per_scene_id for e in env._particle_entities])
        assert np.count_nonzero(mask)>100
        records=[]
        for step in range(20):
            _,reward,_,_,info=env.step(np.zeros(7,dtype=np.float32))
            state=snapshot()
            assert all(np.isfinite(v).all() for v in state.values())
            assert np.array_equal(state['mass'],initial['mass'])
            records.append(dict(step=step+1,reward=float(reward[0]),iou=float(info['iou'][0]),success=bool(info['success'][0])))
        np.savez_compressed(output/'final.npz',**state)
        picture('final.png')
        action=np.array([.01,-.02,.01,.02,-.01,.01,.02],dtype=np.float32)
        env.step(action)
        checkpoint,flat=env.get_state_dict(),env.get_state().clone()
        saved_image=env.goal_image.numpy().copy()
        env.step(-action)
        expected=snapshot()
        trials=[]
        for kind,saved in [('dict',checkpoint),('flat',flat),('reconfigure',checkpoint)]:
            env.reset(seed=17,options={'level_file':'arc.h5','reconfigure':kind=='reconfigure'})
            assert not np.array_equal(saved_image,env.goal_image.numpy()),'Goal change was not exercised'
            obs,_=env.reset(seed=17,options={'level_file':'arc.h5','reset_to_env_states':{'env_states':saved}})
            restored=env.get_state_dict()
            for group in ('mpm','mpm_material','mpm_drives','task'):
                assert all(torch.equal(v,restored[group][k]) for k,v in checkpoint[group].items()),group
            assert torch.equal(checkpoint['controller']['arm']['target_qpos'],restored['controller']['arm']['target_qpos'])
            assert np.array_equal(saved_image,env.goal_image.numpy())
            env.step(-action);actual=snapshot()
            error=float(np.max(abs(actual['qpos']-expected['qpos'])))
            assert error<1e-4,f'Joint checkpoint replay differs: {error}'
            trials.append(dict(kind=kind,joint_replay_error_rad=error,particle_replay_error_m=float(np.max(abs(actual['x']-expected['x'])))))
        return dict(completed=True,parity_validated=False,env_id='Write-v0',seed=101,steps=20,
            goal_scope='Catalog-authored diagnostic targets; official levels not verified',
            particle_count=n,mass_kg=float(initial['mass'].sum()),particle_pixels=int(np.count_nonzero(mask)),
            records=records,checkpoint_trials=trials)
    finally:
        env.close()


def verify_observations():
    reports=[]
    for mode in ('state','state_dict'):
        env=WriteEnv(level_dir='/levels',control_mode='pd_ee_delta_pose_demo',obs_mode=mode,render_backend='gpu')
        try:
            obs,_=env.reset(seed=17,options={'level_file':'elbow.h5'})
            shape=None
            for step in range(2):
                def numpy_tree(value):
                    if isinstance(value,dict):return {k:numpy_tree(v) for k,v in value.items()}
                    data=value.detach().cpu().numpy()
                    assert np.isfinite(data).all()
                    return data
                assert env.observation_space.contains(numpy_tree(obs))
                unflattened=env.get_obs(unflattened=True)
                expected=np.clip(env.goal_image.numpy()[:,::-1],0,255).astype(np.uint8)
                assert np.array_equal(unflattened['extra']['goal'][0].cpu().numpy(),expected)
                if mode=='state':
                    if shape is not None:assert obs.shape==shape
                    shape=obs.shape
                obs,_,_,_,_=env.step(np.array([.001,-.001,-.001,0,0,0],dtype=np.float32))
            reports.append(dict(mode=mode,goal='elbow.h5',seed=17,valid=True))
        finally:env.close()
    return reports


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report={'completed':False}
    try:
        report=run(args.output)
        report['observation_modes']=verify_observations()
    except BaseException as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))

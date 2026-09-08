"""Exercise legacy controls, native IK/drives and controller checkpoint memory."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from mani_skill.envs.softbody.capture import CaptureAdapter
from mani_skill.envs.softbody.controllers import legacy_arm_configs
from mani_skill.envs.softbody.mpm import wp


def action_for(env, mode):
    arm=env.agent.controller.controllers['arm']
    joints=env.agent.robot._objs[0].qpos[:7]
    if mode=='pd_joint_pos':
        action=joints+np.array([.002,-.003,.001,-.002,.001,.002,-.001])
    elif mode=='pd_joint_pos_vel':
        action=np.r_[joints+.002,np.full(7,.02)]
    elif mode=='pd_joint_delta_pos_vel':
        action=np.r_[np.full(7,.01),np.full(7,.02)]
    elif mode=='pd_ee_pose':
        pose=arm.ee_pose_at_base.sp
        action=np.r_[pose.p+[.001,-.002,.001],Rotation.from_quat(pose.q[[1,2,3,0]]).as_rotvec()+[.002,-.001,.003]]
    elif 'ee' in mode:
        action=np.array([.01,-.02,.01,.03,-.02,.01])[:arm.single_action_space.shape[0]]
    else:
        action=np.full(7,.01)
    if 'gripper' in env.agent.controller.controllers:
        action=np.r_[action,0.]
    return action.astype(np.float32)


def run(output):
    wp.config.kernel_cache_dir=str(output.parent/'warp-cache')
    trials=[]
    tasks=[('Fill-v0',list(legacy_arm_configs([], 'bucket'))),
           ('Hang-v0',['pd_ee_delta_pose_align','pd_ee_target_delta_pose']),
           ('Pour-v0',['pd_ee_pose']),
           *[(task,[None]) for task in ('Fill-v0','Hang-v0','Pour-v0')]]
    for task,modes in tasks:
        for mode in modes:
            # Legacy SDF preparation reads visual meshes, so retain the renderer
            # even though this probe never captures camera images.
            kwargs={}
            if task=='Hang-v0':kwargs['legacy_mpm_data_dir']='/legacy-data/hang'
            if task=='Pour-v0':kwargs['legacy_mpm_data_dir']='/legacy-data/pour'
            adapter=CaptureAdapter(task,control_mode=mode,env_kwargs=kwargs)
            env=adapter.env
            try:
                adapter.reset(seed=101,reset_kwargs={})
                default_requested=mode is None
                if default_requested:
                    mode=env.agent.control_mode
                    assert mode=='pd_joint_delta_pos', 'Legacy default control mode changed'
                arm=env.agent.controller.controllers['arm']
                action=action_for(env,mode)
                ik=[]
                for _ in range(2):
                    adapter.step(action)
                    if 'ee' in mode:ik.append(arm.last_ik_success)
                checkpoint,flat=env.get_state_dict(),env.get_state().clone()
                adapter.step(action)
                expected=adapter.snapshot()
                errors=[]
                for kind,saved in [('dict',checkpoint),('flat',flat)]:
                    env.reset(seed=101,options={'reset_to_env_states':{'env_states':saved}})
                    restored=env.get_state_dict()
                    for group in ('mpm','mpm_material','mpm_drives'):
                        assert all(torch.equal(v,restored[group][k]) for k,v in checkpoint[group].items()),f'{task}/{mode}/{group}'
                    def same(a,b):
                        return all(same(a[k],b[k]) for k in a) if isinstance(a,dict) else torch.equal(a,b)
                    assert same(checkpoint.get('controller',{}),restored.get('controller',{})), 'Controller memory restore differs'
                    adapter.step(action)
                    actual=adapter.snapshot()
                    assert all(np.isfinite(v).all() for v in actual.values())
                    error=float(np.max(abs(actual['qpos']-expected['qpos'])))
                    assert error<1e-3, f'Unstable controller replay: {task}/{mode}/{kind}: {error}'
                    errors.append(dict(kind=kind,joint_replay_error_rad=error))
                if ik:assert all(ik), f'Near-current-pose IK failed: {task}/{mode}'
                trial=dict(env_id=task,control_mode=mode,default_requested=default_requested,action_dimension=len(action),ik_success=ik,
                    controller_memory_fields=list(checkpoint.get('controller',{}).get('arm',{})),trials=errors)
                trials.append(trial)
                print(json.dumps(trial),flush=True)
            finally:
                adapter.close()
    return dict(passed=True,scope='Native controller/IK/checkpoint lifecycle; no full task or reference dynamics parity claim',trials=trials)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report={'passed':False}
    try:report=run(args.output)
    except BaseException as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')

"""Pinch goal/material/controller row contracts without native GPU integration."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import sapien
import torch

from mani_skill.envs.softbody import pinch
from mani_skill.envs.softbody.base_env import MPMBaseEnv
from mani_skill.envs.softbody.batch import MPMBatchRuntime
from mani_skill.envs.softbody.mpm import wp


@pytest.fixture(scope='module',autouse=True)
def native_arrays(tmp_path_factory):
    wp.config.kernel_cache_dir=str(tmp_path_factory.mktemp('pinch-warp'));wp.init()


def level(n,offset=0):
    rng=np.random.RandomState(n)
    x=(rng.uniform(-.03,.03,(n,3))+[offset,0,.05]).astype(np.float32)
    data={'x':x,'v':np.full((n,3),.002,np.float32),'F':np.repeat(np.eye(3,dtype=np.float32)[None],n,axis=0),
          'C':np.full((n,3,3),.003,np.float32),'vc':np.ones(n,np.float32),
          'root_pose':np.array([-.56,0,0,1,0,0,0],np.float32),'root_velocity':np.zeros(6,np.float32),
          'qpos':np.arange(9,dtype=np.float32)*.1,'qvel':np.zeros(9,np.float32),
          'ground_pose':np.array([0,0,0,1,0,0,0],np.float32),
          'particle_mass':np.full(n,.003,np.float32),'particle_vol':np.full(n,1e-6,np.float32),
          'particle_mu_lam_ys':np.full((n,3),1000,np.float32),
          'particle_friction_cohesion':np.zeros((n,3),np.float32),'particle_type':np.zeros(n,np.int32),
          'goal':x+np.array([0,offset*.3,0],np.float32),'deformed_distance':np.array([.02,.03],np.float64),
          'goal_depths':np.zeros((4,128,128),np.float32),'goal_rgbs':np.full((4,128,128,3),int(n),np.uint8),
          'goal_cam_pos':np.array([[offset+i*.1,0,.2] for i in range(4)],np.float32),
          'goal_cam_rot':np.repeat(np.array([[1,0,0,0]],np.float32),4,axis=0),
          'goal_cam_intrinsic':np.array([[100,0,64],[0,100,64],[0,0,1]],np.float32)}
    data['goal_depths'][:,64,64]=.25
    return data


@pytest.fixture
def levels(tmp_path):
    records={};arrays={}
    for name,n,offset in [('a.h5',32,0),('b.h5',48,.08)]:
        data=level(n,offset);path=tmp_path/(name+'.npz');np.savez_compressed(path,**data)
        records[name]={'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_sha256':str(n)}
        arrays[name]=data
    return tmp_path,records,arrays


def base_state(env):
    batch=env._mpm_batch is not None;count=env.num_envs
    ns=[len(env.builders[i].mpm_particle_q) for i in range(count)]
    width=env._mpm_batch.capacity if batch else ns[0]
    data={'mpm':{k:torch.zeros((count,width,*shape)) for k,shape in
                 [('x',(3,)),('v',(3,)),('F',(3,3)),('C',(3,3)),('vc',())]},
          'articulations':{'panda':torch.zeros((count,31))},'controller':{'arm':{'target_qpos':env.nominal_state[:,:7].clone()}},
          'mpm_drives':{'position':torch.zeros((count,9)),'velocity':torch.zeros((count,9))}}
    if batch:data['mpm_meta']={'count':torch.tensor(ns)[:,None],'mask':torch.arange(width)[None]<torch.tensor(ns)[:,None]}
    return data


def task(levels,monkeypatch,count=2):
    directory,records,_=levels;env=pinch.PinchEnv.__new__(pinch.PinchEnv)
    env.num_envs=count;env.device=torch.device('cpu');env.mpm_device='cpu';env._mpm_reset_active=True
    env._mpm_batch=MPMBatchRuntime(env,count,64) if count>1 else None
    env._episode_seed=np.array([101,17][:count]);env._pinch_goals=[None]*count
    env.level_files=[None]*count;env.level_sha256s=[None]*count;env.level_dir=directory;env.levels=records
    env.ground=SimpleNamespace(_bodies=[object() for _ in range(count)])
    env.rigid_pose=lambda body:sapien.Pose()
    env.reference_pack={'geometry':[{'name':n} for n in ('hand','left','right')]};env.reference_geometry=[{}]*3
    env.grasp_sites=[object() for _ in range(count)];env.qposes=[];env.builders={};env.rebuilds=[]
    env.nominal_state=torch.zeros((count,9));env.active_indices=list(range(count))
    def reset(q):
        env.qposes.append(q.clone());env.nominal_state[sorted(env.active_indices)]=q
    env.agent=SimpleNamespace(reset=reset,controller=SimpleNamespace(reset=lambda:None),
        robot=SimpleNamespace(name='panda',set_pose=lambda pose:None,
            links_map={name:SimpleNamespace(_objs=[object() for _ in range(count)]) for name in ('hand','left','right')}))
    env.collisions=[]
    monkeypatch.setattr(pinch,'register_reference_collision_body',lambda builder,body,record,arrays:env.collisions.append(body))
    def rebuild(builder,bodies,env_idx=0):env.builders[env_idx]=builder;env.rebuilds.append((env_idx,bodies))
    env.rebuild_mpm=rebuild
    monkeypatch.setattr(MPMBaseEnv,'get_state_dict',base_state)
    return env


def attach_particles(env,arrays):
    couplers=[SimpleNamespace(model=SimpleNamespace(struct=SimpleNamespace(n_particles=len(x))),
                particle_state=lambda x=x:{'x':x.copy()}) for x in arrays]
    if env._mpm_batch is None:env.mpm_coupler=couplers[0]
    else:env._mpm_batch.couplers=couplers


def test_heterogeneous_levels_defer_physics_and_keep_nominal_targets(levels,monkeypatch):
    env=task(levels,monkeypatch);env._initialize_episode(torch.tensor([0,1]),{'level_file':['a.h5','b.h5']})
    checkpoint=env._mpm_initial_checkpoint
    assert checkpoint['mpm_meta']['count'].flatten().tolist()==[32,48]
    for i,(filename,seed) in enumerate(zip(('a.h5','b.h5'),(101,17))):
        data=levels[2][filename];n=len(data['x'])
        single=task(levels,monkeypatch,1);single._episode_seed=np.array([seed])
        single._initialize_episode(torch.tensor([0]),{'level_file':filename})
        for key,value in single._mpm_initial_checkpoint['mpm'].items():
            assert torch.equal(checkpoint['mpm'][key][i,:n],value[0])
            assert not torch.count_nonzero(checkpoint['mpm'][key][i,n:])
        assert torch.equal(checkpoint['controller']['arm']['target_qpos'][i],single._mpm_initial_checkpoint['controller']['arm']['target_qpos'][0])
        assert np.array_equal(checkpoint['articulations']['panda'][i],np.r_[data['root_pose'],data['root_velocity'],data['qpos'],data['qvel']])
        assert np.array_equal(checkpoint['task_particles']['goal'][i,:n],data['goal'])
        assert not torch.count_nonzero(checkpoint['task_particles']['goal'][i,n:])
        for field in ('mpm_particle_q','mpm_particle_mass','mpm_particle_mu_lam_ys'):
            assert np.array_equal(getattr(env.builders[i],field),getattr(single.builders[0],field))
        assert env.rebuilds[i][1]==[env.agent.robot.links_map[name]._objs[i] for name in ('hand','left','right')]


def test_partial_initialization_preserves_unselected_goal_and_targets(levels,monkeypatch):
    env=task(levels,monkeypatch);env._initialize_episode(torch.tensor([0,1]),{'level_file':['a.h5','b.h5']})
    untouched=env._pinch_goals[0];untouched.chamfer=[.1,.2];target=env.nominal_state[0].clone()
    env.active_indices=[1];env._initialize_episode(torch.tensor([1]),{'level_file':'a.h5'})
    assert env._pinch_goals[0] is untouched and untouched.chamfer==[.1,.2]
    assert torch.equal(env.nominal_state[0],target)
    assert env._mpm_initial_checkpoint['mpm_meta']['count'].shape==(1,1)
    assert env._mpm_initial_checkpoint['mpm_meta']['count'][0,0]==32
    assert env.level_files==['a.h5','a.h5'] and [i for i,_ in env.rebuilds]==[0,1,1]


def test_seed_choice_and_reversed_nominal_rows_use_original_recipe(levels,monkeypatch):
    env=task(levels,monkeypatch);env._initialize_episode(torch.tensor([1,0]),{})
    for i,seed in enumerate((101,17)):
        rng=np.random.RandomState(seed);noise=rng.uniform([-.1]*7+[0,0],[.1]*7+[0,0])
        assert np.array_equal(env.nominal_state[i],(np.array([0,.01,0,-1.96,0,1.98,0,.06,.06])+noise).astype(np.float32))
        assert env.level_files[i]==rng.choice(sorted(env.levels))


def test_each_reward_goal_projection_and_observation_uses_own_row(levels,monkeypatch):
    env=task(levels,monkeypatch);env._initialize_episode(torch.tensor([0,1]),{'level_file':['a.h5','b.h5']})
    arrays=[levels[2][name]['x'] for name in ('a.h5','b.h5')];attach_particles(env,arrays)
    poses=[sapien.Pose([0,0,.08]),sapien.Pose([.07,0,.11])]
    env.rigid_pose=lambda body:poses[env.grasp_sites.index(body)]
    monkeypatch.setattr(MPMBaseEnv,'_get_obs_extra',lambda self,info:{})
    info=env.evaluate();rewards=env.compute_dense_reward(None,None,info);obs=env._get_obs_extra(info)
    assert obs['target_points'].shape==(2,65536,4) and obs['target_rgb'].dtype==torch.uint8
    assert info['success'].tolist()==[True,False]
    for i,name in enumerate(('a.h5','b.h5')):
        single=task(levels,monkeypatch,1);single._initialize_episode(torch.tensor([0]),{'level_file':name})
        attach_particles(single,[arrays[i]]);single.rigid_pose=lambda body,i=i:poses[i]
        assert rewards[i]==single.compute_dense_reward(None,None,{})[0]
        assert info['progress'][i]==single.evaluate()['progress'][0]
        assert torch.equal(obs['target_points'][i],single._get_obs_extra({})['target_points'][0])
        projected=env._pinch_goals[i].projection
        expected=levels[2][name]['goal_cam_pos']+np.array([.25,-.00125,-.00125])
        assert np.allclose(projected[:4,:3],expected,rtol=0,atol=1e-8)
        assert np.all(projected[:4,3]==1) and not np.any(projected[4:])


@pytest.mark.parametrize('flat',[False,True])
def test_reduced_goal_restore_changes_only_selected_row(levels,monkeypatch,flat):
    env=task(levels,monkeypatch);env._initialize_episode(torch.tensor([0,1]),{'level_file':['a.h5','b.h5']})
    untouched=env._pinch_goals[0];untouched.chamfer=[0.,0.]
    state=env.get_state_dict()
    def select(value):return {k:select(v) for k,v in value.items()} if isinstance(value,dict) else value[1:2].clone()
    selected=select(state);selected['mpm_meta']['count'][:]=24;selected['mpm_meta']['mask'][:]=torch.arange(64)<24
    selected['task_particles']['goal'].zero_();selected['task_particles']['goal'][0,:24]=torch.tensor(levels[2]['b.h5']['goal'][::2])
    forwarded=[];monkeypatch.setattr(MPMBaseEnv,'set_state_dict',lambda self,state,indices:forwarded.append((state,indices)))
    if flat:
        def leaves(tree):return [leaf for value in tree.values() for leaf in (leaves(value) if isinstance(value,dict) else [value])]
        env.set_state(torch.cat([v.reshape(1,-1) for v in leaves(selected)],dim=1),[1])
    else:env.set_state_dict(selected,[1])
    assert env._pinch_goals[0] is untouched and untouched.chamfer==[0.,0.]
    assert len(env._pinch_goals[1].particles)==24 and env._pinch_goals[1].chamfer is None
    assert set(forwarded[0][0]).isdisjoint({'task','task_particles'}) and forwarded[0][1]==[1]


@pytest.mark.parametrize('bad',['count','padding','rgb','distance','goal','outside_reset'])
def test_invalid_checkpoint_fails_before_physical_assignment(levels,monkeypatch,bad):
    env=task(levels,monkeypatch);env._initialize_episode(torch.tensor([0,1]),{'level_file':['a.h5','b.h5']})
    state=env.get_state_dict();old=list(env._pinch_goals)
    if bad=='count':state['mpm_meta']['count'][0]=0
    if bad=='padding':state['task_particles']['goal'][0,-1]=1
    if bad=='rgb':state['task']['goal_rgbs']=state['task']['goal_rgbs'].float();state['task']['goal_rgbs'][0,0,0,0,0]=.5
    if bad=='distance':state['task']['deformed_distance'][0]=0
    if bad=='goal':state['task_particles']['goal'][0,0,0]=float('nan')
    if bad=='outside_reset':env._mpm_reset_active=False
    calls=[];monkeypatch.setattr(MPMBaseEnv,'set_state_dict',lambda *args:calls.append(args))
    with pytest.raises((ValueError,RuntimeError)):env.set_state_dict(state,[0,1])
    assert not calls and all(a is b for a,b in zip(env._pinch_goals,old))

"""Write row isolation using real CPU Warp rasters and unfinalized MPM builders.

These tests do not replace shared-world GPU dynamics and rendering evidence.
"""
import hashlib
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import sapien
import torch

from mani_skill.envs.softbody import write
from mani_skill.envs.softbody.base_env import MPMBaseEnv
from mani_skill.envs.softbody.batch import MPMBatchRuntime
from mani_skill.envs.softbody.mpm import wp


@pytest.fixture(scope='module', autouse=True)
def native_cpu_rasters(tmp_path_factory):
    wp.config.kernel_cache_dir=str(tmp_path_factory.mktemp('write-warp'))
    wp.init()


def points(axis=0):
    xy=np.stack(np.meshgrid(np.linspace(-.105,.102,64),np.linspace(-.105,.102,64)),axis=-1).reshape(-1,2)
    z=np.where(np.abs(xy[:,axis])<.018,.025,.055)
    return np.resize(np.column_stack((xy,z)),(19404,3)).astype(np.float32)


@pytest.fixture
def levels(tmp_path):
    for name,axis in [('a.h5',0),('b.h5',1)]:
        with h5py.File(tmp_path/name,'w') as f:
            f['goal']=points(axis)
    return tmp_path


def task(levels,count=2):
    env=write.WriteEnv.__new__(write.WriteEnv)
    env.num_envs=count;env.device=torch.device('cpu');env.mpm_device='cpu'
    env._mpm_reset_active=True
    env._mpm_batch=MPMBatchRuntime(env,count,32768) if count>1 else None
    env._episode_seed=np.array([101,17][:count])
    env._write_goals=[None]*count;env.level_files=[None]*count;env.level_sha256s=[None]*count
    env.level_dir=levels;env.all_filepaths=sorted(levels.glob('*.h5'))
    env.hands=[object() for _ in range(count)];env.end_effectors=[object() for _ in range(count)]
    env.walls=[SimpleNamespace(_bodies=[object() for _ in range(count)]) for _ in range(4)]
    env.reference_pack={'geometry':list(range(5))};env.reference_geometry=list(range(5))
    env.qposes=[];env.rebuilds=[]
    env.agent=SimpleNamespace(reset=lambda q:env.qposes.append(q.clone()),robot=SimpleNamespace(set_pose=lambda p:None))
    env.rebuild_mpm=lambda builder,bodies,env_idx=0:env.rebuilds.append((env_idx,builder,bodies))
    return env


@pytest.fixture
def collision_calls(monkeypatch):
    calls=[]
    monkeypatch.setattr(write,'register_reference_collision_body',lambda builder,body,record,arrays:calls.append((body,record,arrays)))
    return calls


def couplers(env,arrays,padding=0):
    result=[]
    for array in arrays:
        padded=np.r_[array,np.tile([0.,0.,1.],(padding,1))].astype(np.float32)
        result.append(SimpleNamespace(model=SimpleNamespace(struct=SimpleNamespace(n_particles=len(array))),
            states=[SimpleNamespace(struct=SimpleNamespace(particle_q=wp.array(padded,dtype=wp.vec3,device='cpu')))],
            particle_state=lambda array=array:{'x':array.copy()}))
    if env._mpm_batch is not None:
        env._mpm_batch.couplers=result
    else:
        env.mpm_coupler=result[0]


def test_seeds_preserve_reference_robot_level_choice_and_particle_recipe(levels,collision_calls):
    batch=task(levels);batch._initialize_episode(torch.tensor([0,1]),{})
    assert [x[0] for x in batch.rebuilds]==[0,1]
    for index,seed in enumerate((101,17)):
        single=task(levels,1);single._episode_seed=np.array([seed])
        single._initialize_episode(torch.tensor([0]),{})
        rng=np.random.RandomState(seed)
        nominal=np.array([-.029177314,.10816099,.03054934,-2.1639752,-.0013982388,2.2785723,.79039097])
        expected=(nominal+rng.uniform([-.1]*7,[.1]*7)).astype(np.float32)
        assert np.array_equal(batch.qposes[0][index].numpy(),expected)
        assert torch.equal(batch.qposes[0][index],single.qposes[0][0])
        assert batch.level_files[index]==rng.choice(batch.all_filepaths).name==single.level_file
        assert batch.level_sha256s[index]==hashlib.sha256((levels/batch.level_files[index]).read_bytes()).hexdigest()
        a,b=batch.rebuilds[index][1],single.rebuilds[0][1]
        for field in ('mpm_particle_q','mpm_particle_qd','mpm_particle_mass','mpm_particle_volume',
                      'mpm_particle_mu_lam_ys','mpm_particle_friction_cohesion','mpm_particle_type','mpm_particle_colors'):
            assert np.array_equal(getattr(a,field),getattr(b,field)),field
        assert len(a.mpm_particle_q)==19404
        assert np.isclose(np.asarray(a.mpm_particle_mass,dtype=np.float32).sum(),7.2764997482299805,rtol=0,atol=1e-6)
        assert batch.rebuilds[index][2]==[batch.hands[index],*[w._bodies[index] for w in batch.walls]]
    assert [call[0] for call in collision_calls[:10]]==[b for row in batch.rebuilds for b in row[2]]


def test_partial_reversed_goal_overrides_preserve_untouched_buffers(levels,collision_calls):
    env=task(levels);env._initialize_episode(torch.tensor([1,0]),{'level_file':['b.h5','a.h5']})
    assert env.level_files==['a.h5','b.h5']
    for i,seed in enumerate((101,17)):
        nominal=np.array([-.029177314,.10816099,.03054934,-2.1639752,-.0013982388,2.2785723,.79039097])
        expected=(nominal+np.random.RandomState(seed).uniform([-.1]*7,[.1]*7)).astype(np.float32)
        assert np.array_equal(env.qposes[0][i].numpy(),expected)
    couplers(env,[points(),points(1)])
    env.evaluate()
    untouched=env._write_goals[0]
    before=(untouched.points.copy(),untouched.image.numpy(),untouched.current.numpy(),untouched.iou)
    env._initialize_episode(torch.tensor([1]),{'level_file':'a.h5'})
    assert env._write_goals[0] is untouched and env.level_files==['a.h5','a.h5']
    assert np.array_equal(before[0],untouched.points)
    assert np.array_equal(before[1],untouched.image.numpy())
    assert np.array_equal(before[2],untouched.current.numpy()) and before[3]==untouched.iou
    assert env._write_goals[1].iou is None
    assert [r[0] for r in env.rebuilds]==[1,0,1]


@pytest.mark.parametrize('override',[['a.h5'],['a.h5','../bad.h5'],['a.h5',3],False,''])
def test_invalid_overrides_do_not_mutate_robot_or_material(levels,override,collision_calls):
    env=task(levels)
    with pytest.raises(ValueError):env._initialize_episode(torch.tensor([0,1]),{'level_file':override})
    assert not env.qposes and not env.rebuilds and env._write_goals==[None,None]


@pytest.mark.parametrize('bad',['shape','nonfinite','softlink','external','symlink'])
def test_invalid_second_goal_is_rejected_before_any_reset(levels,bad,collision_calls):
    path=levels/'invalid.h5'
    if bad=='symlink':
        path.symlink_to(levels/'a.h5')
    else:
        with h5py.File(path,'w') as f:
            if bad=='shape':f['goal']=np.zeros((3,3))
            if bad=='nonfinite':f['goal']=np.full((19404,3),np.nan)
            if bad=='softlink':f['actual']=points();f['goal']=h5py.SoftLink('/actual')
            if bad=='external':f['goal']=h5py.ExternalLink('a.h5','/goal')
    env=task(levels)
    with pytest.raises(ValueError):env._initialize_episode(torch.tensor([0,1]),{'level_file':['a.h5','invalid.h5']})
    assert not env.qposes and not env.rebuilds


def test_rasters_iou_reward_and_observation_use_own_row_and_live_count(levels,monkeypatch):
    env=task(levels)
    arrays=[points(),points()[:4096:3]]
    for i in range(2):env._set_goal(points(i),i)
    couplers(env,arrays,padding=500)
    poses=[sapien.Pose([0.,0.,.04]),sapien.Pose([.08,.01,.13],[.9238795,0.,.3826834,0.])]
    env.rigid_pose=lambda body:poses[env.end_effectors.index(body)]
    monkeypatch.setattr(MPMBaseEnv,'_get_obs_extra',lambda self,info:{})
    report=env.evaluate();reward=env.compute_dense_reward(None,None,report);obs=env._get_obs_extra(report)
    assert report['success'].tolist()==[True,False]
    assert obs['goal'].shape==(2,64,64) and obs['tcp_pose'].shape==(2,7)
    for i in range(2):
        single=task(levels,1);single._set_goal(points(i));couplers(single,[arrays[i]])
        single.rigid_pose=lambda body,i=i:poses[i]
        assert report['iou'][i]==single.evaluate()['iou'][0]
        assert reward[i]==single.compute_dense_reward(None,None,{})[0]
        assert np.array_equal(env._write_goals[i].current.numpy(),single.current_image.numpy())
        assert np.array_equal(obs['goal'][i].numpy(),single.goal_image_display)
        assert np.array_equal(obs['tcp_pose'][i].numpy(),np.r_[poses[i].p,poses[i].q])
        g,c=env._write_goals[i].image.numpy(),env._write_goals[i].current.numpy()
        ratio=np.count_nonzero(((g<50)&(c<40))|((g<40)&(c<50)))/np.count_nonzero((g<40)|(c<40))
        assert env._compute_iou(i)==ratio
    assert not np.array_equal(obs['goal'][0],obs['goal'][1])
    assert env.goal_points is env._write_goals[0].points
    assert env.goal_image is env._write_goals[0].image and env.iou_buffer is env._write_goals[0].buffer
    monkeypatch.setattr(MPMBaseEnv,'_after_control_step',lambda self:None)
    env._after_control_step()
    assert all(g.iou is None for g in env._write_goals)


@pytest.mark.parametrize('flat',[False,True])
def test_partial_checkpoint_restores_selected_goal_and_keeps_unselected_cache(levels,monkeypatch,flat):
    env=task(levels)
    for i in range(2):env._set_goal(points(i),i)
    couplers(env,[points(),points(1)]);env.evaluate()
    untouched=env._write_goals[0]
    forwarded=[]
    monkeypatch.setattr(MPMBaseEnv,'get_state_dict',lambda self:{'physical':{'value':torch.zeros((2,1))}})
    monkeypatch.setattr(MPMBaseEnv,'set_state_dict',lambda self,state,indices:forwarded.append((state,indices)))
    saved=env.get_state_dict()
    selected={'physical':{'value':torch.ones((1,1))},'task':{'goal_points':saved['task']['goal_points'][:1].clone()}}
    if flat:
        env.set_state(torch.cat([torch.ones((1,1)),selected['task']['goal_points'].reshape(1,-1)],dim=1),[1])
    else:env.set_state_dict(selected,[1])
    assert env._write_goals[0] is untouched and untouched.iou==1.
    assert env._write_goals[1].iou is None
    assert np.array_equal(env._write_goals[1].points,points())
    assert np.array_equal(env._write_goals[1].image.numpy(),untouched.image.numpy())
    assert forwarded[0][1]==[1] and set(forwarded[0][0])=={'physical'}
    saved['task']['goal_points'].zero_()
    assert np.array_equal(env.goal_points,points())


@pytest.mark.parametrize('bad',['size','nan','overflow','complex','outside_reset'])
def test_bad_checkpoint_cannot_change_physical_or_task_state(levels,monkeypatch,bad):
    env=task(levels);calls=[]
    goal=torch.tensor(points())[None]
    if bad=='size':goal=goal[:,:100]
    if bad=='nan':goal[0,0,0]=float('nan')
    if bad=='overflow':goal=goal.double();goal[0,0,0]=1e100
    if bad=='complex':goal=goal.to(torch.complex64)+1j
    if bad=='outside_reset':env._mpm_reset_active=False
    monkeypatch.setattr(MPMBaseEnv,'set_state_dict',lambda *a:calls.append(a))
    with pytest.raises((ValueError,RuntimeError)):env.set_state_dict({'task':{'goal_points':goal}},[1])
    assert not calls and env._write_goals==[None,None]


def test_reversed_checkpoint_rows_match_native_mask_order(levels,monkeypatch):
    env=task(levels);calls=[]
    monkeypatch.setattr(MPMBaseEnv,'set_state_dict',lambda self,state,indices:calls.append((state,indices)))
    env.set_state_dict({'task':{'goal_points':torch.tensor(np.stack([points(1),points()]))},
                        'physical':{'rows':torch.tensor([[17],[101]])}},[1,0])
    assert calls[0][1]==[0,1] and calls[0][0]['physical']['rows'].tolist()==[[101],[17]]
    assert np.array_equal(env._write_goals[0].points,points())
    assert np.array_equal(env._write_goals[1].points,points(1))

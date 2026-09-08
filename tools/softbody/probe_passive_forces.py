"""Independent native-CPU and Pinocchio force/online-compensation experiment."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import sapien
import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def values(pose):
    return np.r_[pose.p, pose.q].tolist()


def panda(scene, index, urdf, parameters, output):
    # Keep the existing robot's kinematic/inertial recipe. Collision and visual
    # elements are unnecessary for a force diagnostic; no mesh is redistributed.
    xml = ET.parse(urdf)
    for link in xml.getroot().findall('link'):
        for child in list(link):
            if child.tag in ('visual', 'collision'):
                link.remove(child)
    stripped = output / f'panda-{index}.urdf'
    xml.write(stripped)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    articulation_builders, actor_builders, _ = loader.parse(str(stripped))
    if len(articulation_builders) != 1 or actor_builders:
        raise ValueError('Expected one native Panda articulation')
    builder = articulation_builders[0]
    angle = .31 * index
    axis = np.array([1., 2., 1.]); axis /= np.linalg.norm(axis)
    builder.set_initial_pose(sapien.Pose([.1, -.2, .3], np.r_[np.cos(angle/2), axis*np.sin(angle/2)]))
    robot = builder.build(fix_root_link=True, build_mimic_joints=False)
    links = {b.name: b for b in robot.links}
    joints = {j.name: j for j in robot.joints}
    if set(links) != {r['name'] for r in parameters['links']}:
        raise ValueError('Panda link recipe differs from reference export')
    for record in parameters['links']:
        body = links[record['name']]
        body.mass = record['mass']; body.inertia = record['inertia']
        pose = record['com']; body.cmass_local_pose = sapien.Pose(pose[:3], pose[3:])
    for record in parameters['joints']:
        joint = joints[record['name']]
        parent, child = record['parent_pose'], record['child_pose']
        joint.pose_in_parent = sapien.Pose(parent[:3], parent[3:])
        joint.pose_in_child = sapien.Pose(child[:3], child[3:])
    q = dict(zip([f'panda_joint{i}' for i in range(1, 8)], [0., .4, 0., -1.8, 0., 2., .7]))
    v = dict(zip([f'panda_joint{i}' for i in range(1, 8)], [.15, -.12, .07, -.08, .05, .1, -.04]))
    q.update(panda_finger_joint1=.02, panda_finger_joint2=.02)
    v.update(panda_finger_joint1=.002, panda_finger_joint2=-.002)
    return robot, np.array([q[j.name] for j in robot.active_joints], np.float32), np.array([v[j.name] for j in robot.active_joints], np.float32)


def run(mode, input_root, urdf, bucket_urdf, panda_parameters, output):
    gpu = mode == 'gpu-model'
    if gpu:
        sapien.physx.enable_gpu()
    gravity = [0.4, -1.2, -9.81]
    sapien.physx.set_scene_config(gravity=gravity, enable_pcm=gpu, enable_tgs=False)
    sapien.physx.set_body_config(solver_position_iterations=25, solver_velocity_iterations=1, sleep_threshold=0.)
    system = sapien.physx.PhysxGpuSystem() if gpu else sapien.physx.PhysxCpuSystem()
    system.timestep = .002
    if gpu:
        system.gpu_set_cuda_stream(torch.cuda.current_stream().cuda_stream)
    branched = load('branched_fixture', input_root / 'branched.py')
    adapter = load('passive_candidate', input_root / 'passive_forces.py') if mode != 'cpu-native' else None
    parameters = json.loads((input_root / 'robot_parameters.json').read_text())['robot_parameters']
    gripper_parameters = json.loads(panda_parameters.read_text())['robot_parameters']
    scenes, robots, starts, models, recipes = [], [], [], [], []
    for index in range(7):
        scene = sapien.Scene([system]); scenes.append(scene)
        if gpu:
            system.set_scene_offset(scene, [4. * index, 0., 0.])
        if index < 3:
            robot, q, v = branched.build(scene, index, gpu)
        else:
            robot, q, v = panda(scene, index, urdf if index < 5 else bucket_urdf,
                gripper_parameters if index < 5 else parameters, output)
        robot.sleep_threshold = 0.
        for link in robot.links:
            link.linear_damping = 0.; link.angular_damping = 0.
            if link.collision_shapes:
                raise ValueError('Force diagnostic must be collision-free')
        for joint in robot.active_joints:
            joint.friction = 0.; joint.armature = [0.]; joint.set_drive_properties(0., 0.)
        robots.append(robot); starts.append((q, v))
        root_q = robot.root.entity_pose.q.copy()
        if adapter:
            models.append(adapter.FixedBasePassiveForces(robot, gravity_world=gravity, root_quaternion_wxyz=root_q))
        recipes.append(dict(kind='branched' if index < 3 else ('panda' if index < 5 else 'bucket'), root=values(robot.root.entity_pose),
            joint_names=[j.name for j in robot.active_joints],
            links=[dict(name=b.name, mass=float(b.mass), inertia=b.inertia.tolist(), com=values(b.cmass_local_pose),
                parent=b.parent.name if b.parent else None, type=b.joint.type,
                parent_pose=values(b.joint.pose_in_parent), child_pose=values(b.joint.pose_in_child)) for b in robot.links]))
    # Static evaluations are separate reset states, not a manipulation rollout.
    random = np.random.default_rng(5031)
    samples = []
    for index, robot in enumerate(robots):
        cases = []
        for case in range(25):
            limits = np.array([j.limit[0] for j in robot.active_joints])
            low, high = np.maximum(limits[:, 0], -2.), np.minimum(limits[:, 1], 2.)
            q = random.uniform(low + .1*(high-low), high - .1*(high-low)).astype(np.float32)
            v = random.uniform(-3., 3., robot.dof).astype(np.float32) if case else np.zeros(robot.dof, np.float32)
            if mode == 'cpu-native':
                robot.set_qpos(q); robot.set_qvel(v)
                compute = lambda g, c: robot.compute_passive_force(gravity=g, coriolis_and_centrifugal=c)
            else:
                compute = lambda g, c: models[index].compute(q, v, gravity=g, coriolis_and_centrifugal=c)
            cases.append(dict(qpos=q.tolist(), qvel=v.tolist(),
                gravity=compute(True, False).tolist(), coriolis=compute(False, True).tolist(), total=compute(True, True).tolist()))
        samples.append(cases)
    if gpu:
        system.gpu_init()
        for robot, (q, v) in zip(robots, starts):
            row, count = robot.gpu_index, robot.dof
            system.cuda_articulation_qpos.torch()[row, :count] = torch.tensor(q, device='cuda')
            system.cuda_articulation_qvel.torch()[row, :count] = torch.tensor(v, device='cuda')
        system.cuda_articulation_qf.torch().zero_()
        system.gpu_apply_articulation_qpos(); system.gpu_apply_articulation_qvel(); system.gpu_apply_articulation_qf()
        system.gpu_update_articulation_kinematics()
        gpu_passive = adapter.GPUPassiveForces(system, robots, gravity_world=gravity)
    else:
        for robot, (q, v) in zip(robots, starts):
            robot.set_qpos(q); robot.set_qvel(v); robot.set_qf(np.zeros(robot.dof))
    frames, applied = [], []
    for step in range(61):
        if gpu:
            system.gpu_fetch_articulation_qpos(); system.gpu_fetch_articulation_qvel()
            qdata = system.cuda_articulation_qpos.torch().cpu().numpy()
            vdata = system.cuda_articulation_qvel.torch().cpu().numpy()
            observed = [(qdata[r.gpu_index, :r.dof].copy(), vdata[r.gpu_index, :r.dof].copy()) for r in robots]
        else:
            observed = [(r.qpos.copy(), r.qvel.copy()) for r in robots]
        frames.append([dict(qpos=q.tolist(), qvel=v.tolist()) for q, v in observed])
        if step == 60:
            break
        forces = [r.compute_passive_force() for r in robots] if mode == 'cpu-native' else [m.compute(q, v) for m, (q, v) in zip(models, observed)]
        if gpu:
            target = system.cuda_articulation_qf.torch()
            computed = gpu_passive.compute()
            forces = [computed[r.gpu_index, :r.dof].cpu().numpy().copy() for r in robots]
            target.copy_(computed)
            system.gpu_apply_articulation_qf()
        else:
            for r, f in zip(robots, forces):
                r.set_qf(f)
        applied.append([np.asarray(f).tolist() for f in forces])
        system.step()
    report = dict(mode=mode, physics_system=type(system).__name__, gravity_world=gravity, recipes=recipes,
        static_samples=samples, frames=frames, applied=applied, dt=.002,
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
        bucket_urdf_sha256=hashlib.sha256(bucket_urdf.read_bytes()).hexdigest(),
        panda_parameters_sha256=hashlib.sha256(panda_parameters.read_bytes()).hexdigest(),
        online_force_path='GPUPassiveForces live native GPU reads' if gpu else mode,
        scope='Native-model force evaluation and force-only constant-velocity trajectory; no MPM or collision/manipulation claim')
    (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(mode=mode, static_states=175, physics_steps=60)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['cpu-native', 'cpu-model', 'gpu-model'], required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--bucket-urdf', type=Path, required=True)
    parser.add_argument('--panda-parameters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.mode, args.input, args.urdf, args.bucket_urdf, args.panda_parameters, args.output)

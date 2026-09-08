"""Real SAPIEN 3 + legacy MPM contact probe with independent impulse accounting.

An initially moving deformable strikes a free rigid plate in zero gravity.
Only contact forces move the plate. No controller is used. All geometry and
material values are authored diagnostic assumptions, not calibrated materials.
"""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mani_skill.envs.softbody.mpm import MPMCoupler, MPMModelBuilder, register_collision_body, wp
import sapien


def run(device, output, *, rotated=False, steps=100, articulation=None):
    wp.config.kernel_cache_dir = str((output.parent / "warp-cache").resolve())
    wp.init()
    sapien.physx.set_scene_config(gravity=[0., 0., 0.], enable_enhanced_determinism=True)
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    scene.set_timestep(.002)
    half = np.array([.006, .04, .04], dtype=np.float32)
    inertia = .05 / 3 * np.array([half[1]**2 + half[2]**2,
                                           half[0]**2 + half[2]**2,
                                           half[0]**2 + half[1]**2])
    material = sapien.physx.PhysxMaterial(0., 0., 0.)
    pivot = np.array([0., 0., .1])
    lever = .025 if articulation == "hinge" else 0.
    if articulation:
        if rotated:
            raise ValueError("Rotated free-body and articulation variants are separate probes")
        ab = scene.create_articulation_builder()
        root = ab.create_link_builder()
        root.set_name("fixed_base")
        link = ab.create_link_builder(root)
        link.set_name("contact_plate")
        link.add_box_collision(half_size=half, material=material)
        link.set_mass_and_inertia(.05, sapien.Pose(), inertia)
        # SAPIEN joints use their local X axis. Turn it onto world Z for the hinge.
        joint_q = [np.sqrt(.5), 0., -np.sqrt(.5), 0.] if articulation == "hinge" else [1., 0., 0., 0.]
        link.set_joint_properties("revolute_unwrapped" if articulation == "hinge" else "prismatic",
                                  limits=[[-10., 10.]],
                                  pose_in_parent=sapien.Pose(q=joint_q),
                                  pose_in_child=sapien.Pose([0., -lever, 0.], joint_q),
                                  friction=0., damping=0.)
        ab.set_initial_pose(sapien.Pose(pivot))
        robot = ab.build(fix_root_link=True)
        joint = robot.active_joints[0]
        # SAPIEN 3.0.3's builder stores friction but does not copy it to the joint.
        joint.friction = 0.
        joint.armature = [0.]
        joint.set_drive_properties(0., 0.)
        robot.set_qpos([0.])
        robot.set_qvel([0.])
        body = robot.get_links()[1]
        entity = body.entity
        effective_inertia = float(inertia[2] + body.mass * lever**2)
    else:
        entity = sapien.Entity()
        body = sapien.physx.PhysxRigidDynamicComponent()
        body.attach(sapien.physx.PhysxCollisionShapeBox(half, material))
        body.mass = .05
        body.inertia = inertia
        entity.add_component(body)
        angle = np.pi / 6 if rotated else 0.
        entity.set_pose(sapien.Pose(pivot, [np.cos(angle/2), 0., 0., np.sin(angle/2)]))
        scene.add_entity(entity)
    body.linear_damping = 0.
    body.angular_damping = 0.
    if articulation:
        robot.sleep_threshold = 0.
    else:
        body.sleep_threshold = 0.
    builder = MPMModelBuilder()
    # Leave enough travel for the rebound over the complete 0.2 s horizon.
    # Grid boundary contact would add an external impulse to this closed-system test.
    builder.set_mpm_domain([.6, .2, .2], grid_length=.005)
    register_collision_body(builder, body)
    builder.add_mpm_grid(pos=(-.035, lever, .1), vel=(1., 0., 0.),
                         dim_x=3, dim_y=3, dim_z=3, cell_x=.003, cell_y=.003, cell_z=.003,
                         density=1000., mu_lambda_ys=(1000., 1000., 10000.),
                         friction_cohesion=(0., 0., 0.), type=0, jitter=False)
    model = builder.finalize(device)
    model.gravity = np.zeros(3, dtype=np.float32)
    model.struct.ground_normal = wp.vec3(0., 0., 1.)
    model.struct.particle_radius = .0015
    model.struct.body_sticky = 1
    model.struct.body_mu = 0.
    model.grid_contact = True
    model.particle_contact = True
    model.adaptive_grid = False
    states = [model.state() for _ in range(5)]
    builder.init_model_state(model, states)
    coupler = MPMCoupler(scene, model, states, [body], mpm_dt=.0005)
    mass = model.struct.particle_mass.numpy().astype(np.float64)
    initial = coupler.particle_state()
    p0 = np.sum(initial["v"].astype(np.float64) * mass[:, None], axis=0)
    frames = {"particle_x": [initial["x"]], "particle_v": [initial["v"]],
              "rigid_pose": [np.r_[entity.pose.p, entity.pose.q]],
              "rigid_v": [body.linear_velocity.copy()], "wrenches": [],
              "total_momentum": [p0]}
    impulse = np.zeros(3)
    joint_impulse = 0.
    joint_errors = []
    for _ in range(steps):
        com = (entity.pose * body.cmass_local_pose).p.copy()
        detail = coupler.step()
        state = coupler.particle_state()
        impulse += detail.mean_wrench_torque_force[0, 3:].astype(float) * detail.rigid_dt
        if articulation:
            wrench = detail.mean_wrench_torque_force[0].astype(float)
            if articulation == "hinge":
                joint_impulse += (wrench[:3] + np.cross(com - pivot, wrench[3:]))[2] * detail.rigid_dt
                joint_errors.append(abs(effective_inertia * robot.get_qvel()[0] - joint_impulse))
            else:
                joint_impulse += wrench[3] * detail.rigid_dt
                joint_errors.append(abs(body.mass * robot.get_qvel()[0] - joint_impulse))
            frames.setdefault("joint_qpos", []).append(robot.get_qpos().copy())
            frames.setdefault("joint_qvel", []).append(robot.get_qvel().copy())
        p = np.sum(state["v"].astype(float) * mass[:, None], axis=0) + body.mass * body.linear_velocity.astype(float)
        frames["particle_x"].append(state["x"])
        frames["particle_v"].append(state["v"])
        frames["rigid_pose"].append(np.r_[entity.pose.p, entity.pose.q])
        frames["rigid_v"].append(body.linear_velocity.copy())
        frames["wrenches"].append(detail.substep_wrench_torque_force)
        frames["total_momentum"].append(p)
    frames = {k: np.asarray(v) for k, v in frames.items()}
    frames["particle_mass"] = mass
    momentum_error = float(np.max(np.linalg.norm(frames["total_momentum"] - p0, axis=1)))
    rigid_impulse_error = float(np.linalg.norm(body.mass * body.linear_velocity.astype(float) - impulse))
    displacement = float(np.linalg.norm(entity.pose.p - frames["rigid_pose"][0, :3]))
    peak_force = float(np.linalg.norm(frames["wrenches"][..., 3:], axis=-1).max())
    grid_clearance = float(.3 - np.abs(frames["particle_x"][..., 0]).max())
    finite = all(np.isfinite(v).all() for v in frames.values())
    if articulation == "hinge":
        impulse_pass = max(joint_errors) <= 2e-8
    elif articulation == "slider":
        impulse_pass = max(joint_errors) <= 2e-7
    else:
        impulse_pass = momentum_error <= 2e-6 and rigid_impulse_error <= 2e-7
    report = {"probe": "mpm-sapien3-free-body-contact", "device": device,
              "rotated_plate": rotated, "steps": steps, "rigid_dt_s": .002, "mpm_dt_s": .0005,
              "articulation": articulation,
              "max_joint_impulse_error": float(max(joint_errors)) if joint_errors else None,
              "joint_impulse_units": "kg m^2 / s" if articulation == "hinge" else "kg m / s",
              "particles": len(mass), "particle_mass_kg": float(mass.sum()), "rigid_mass_kg": float(body.mass),
              "peak_contact_force_n": peak_force, "rigid_displacement_m": displacement,
              "minimum_x_grid_clearance_m": grid_clearance,
              "max_momentum_error_kg_m_s": momentum_error, "rigid_impulse_error_kg_m_s": rigid_impulse_error,
              "limits": {"momentum_error_kg_m_s": 2e-6, "rigid_impulse_error_kg_m_s": 2e-7,
                         "minimum_rigid_displacement_m": 1e-5,
                         "joint_impulse_error": 2e-8 if articulation == "hinge" else 2e-7},
              "impulse_gates_applied": ["joint_impulse_error"] if articulation else ["momentum_error_kg_m_s", "rigid_impulse_error_kg_m_s"],
              "passed": bool(finite and impulse_pass and displacement > 1e-5 and peak_force > 0 and grid_clearance > .02),
              "scope": ("Single CPU PhysX scene. Joint variants compare generalized impulse; the fixed base exchanges linear momentum. "
                        "Free-body variants compare total linear momentum. No task/batching/soft-robot claim."),
              "platform": platform.platform(), "sapien_version": sapien.__version__,
              "fixture": {"source": "repo://ManiSkill/tools/softbody/probe_coupling.py", "creator": "sim-infra port work",
                          "license": "Apache-2.0 for authored probe; bundled MPM retains its own license",
                          "units": "m, kg, s", "frame": "Z-up, SAPIEN wxyz, world-frame wrenches about COM",
                          "rigid_dimensions_m": (half * 2).tolist(), "collision": "box",
                          "inertia_basis": "Uniform solid box estimate", "friction": 0., "restitution": 0.,
                          "joint": articulation, "joint_pivot_world_m": pivot.tolist() if articulation else None,
                          "hinge_com_offset_m": lever, "joint_damping": 0., "joint_friction": 0.,
                          "actual_joint_friction": float(joint.friction) if articulation else None,
                          "actual_joint_armature": joint.armature.tolist() if articulation else None,
                          "actual_inertia_kg_m2": body.inertia.tolist(),
                          "mpm_domain_m": [.6, .2, .2], "grid_length_m": .005,
                          "mpm_grid_contact": True, "mpm_particle_contact": True,
                          "material_basis": "Uncalibrated diagnostic assumptions", "modifications": "none",
                          "file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    np.savez_compressed(output / "states.npz", **frames)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rotated", action="store_true")
    parser.add_argument("--articulation", choices=["slider", "hinge"])
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        report = run(args.device, args.output, rotated=args.rotated, steps=args.steps, articulation=args.articulation)
    except BaseException as exc:
        report = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)

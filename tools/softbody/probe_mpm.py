"""Small real-kernel probes for the experimental legacy MPM runtime.

These analytical diagnostics do not establish ManiSkill 2 CUDA task parity.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "warp_maniskill"))

import warp as wp
from mpm.mpm_model import MPMModelBuilder
from mpm.mpm_simulator import Simulator


def run(device, cache, steps=40, dt=0.0005):
    wp.config.kernel_cache_dir = str(cache.resolve())
    wp.init()
    if device == "cuda" and not wp.is_cuda_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    builder = MPMModelBuilder()
    builder.set_mpm_domain([0.2, 0.2, 0.2], grid_length=0.01)
    builder.add_mpm_grid(
        pos=(0., 0., 0.1), vel=(0., 0., 0.), dim_x=2, dim_y=2, dim_z=2,
        cell_x=.005, cell_y=.005, cell_z=.005, density=1000.,
        mu_lambda_ys=(1000., 1000., 10000.), friction_cohesion=(0., 0., 0.),
        type=0, jitter=False,
    )
    model = builder.finalize(device)
    model.gravity = np.array([0., 0., -9.81], dtype=np.float32)
    model.struct.ground_normal = wp.vec3(0., 0., 1.)
    model.struct.particle_radius = .0025
    states = [model.state(), model.state()]
    builder.init_model_state(model, states)
    simulator = Simulator(device=device)
    mass = model.struct.particle_mass.numpy()
    initial = states[0].struct.particle_q.numpy().copy()
    initial_com = np.average(initial, axis=0, weights=mass)
    positions, velocities = [initial], [states[0].struct.particle_qd.numpy().copy()]
    started = time.monotonic()
    for _ in range(steps):
        simulator.simulate(model, states[0], states[1], dt)
        states.reverse()
        x = states[0].struct.particle_q.numpy().copy()
        v = states[0].struct.particle_qd.numpy().copy()
        if not np.isfinite(x).all() or not np.isfinite(v).all():
            raise RuntimeError("Non-finite MPM state")
        if states[0].struct.error.numpy()[0] != 0:
            raise RuntimeError("MPM kernel reported an error")
        positions.append(x)
        velocities.append(v)
    # Gravity is integrated before position update (semi-implicit Euler).
    expected_com = initial_com + model.gravity * dt * dt * steps * (steps + 1) / 2
    actual_com = np.average(positions[-1], axis=0, weights=mass)
    actual_v = np.average(velocities[-1], axis=0, weights=mass)
    expected_v = model.gravity * dt * steps
    position_error = float(np.linalg.norm(actual_com - expected_com))
    velocity_error = float(np.linalg.norm(actual_v - expected_v))
    result = {
        "probe": "ballistic-mpm", "device": device, "platform": platform.platform(),
        "steps": steps, "dt_s": dt, "particles": len(initial),
        "mass_kg": float(mass.sum()), "position_error_m": position_error,
        "velocity_error_m_s": velocity_error,
        "limits": {"position_error_m": 3e-6, "velocity_error_m_s": 3e-5},
        "passed": position_error <= 3e-6 and velocity_error <= 3e-5,
        "wall_time_s": time.monotonic() - started,
        "initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
        "scope": "Analytical free-flight diagnostic; no task, contact or CUDA parity claim",
        "material_basis": "Uncalibrated diagnostic material; original MPM constitutive parameters",
    }
    return result, {"x": np.array(positions), "v": np.array(velocities), "mass": mass}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result, states = run(args.device, args.output.parent / "warp-cache")
        np.savez_compressed(args.output / "states.npz", **states)
    except Exception as exc:
        result = {"passed": False, "probe": "ballistic-mpm", "error": f"{type(exc).__name__}: {exc}"}
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)

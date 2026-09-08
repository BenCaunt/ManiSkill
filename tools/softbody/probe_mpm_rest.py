"""Measure the original MPM solver at rest without SAPIEN or a coupling adapter."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def run(source, output):
    sys.path.insert(0, str(source / 'warp_maniskill'))
    import warp as wp
    from mpm.mpm_model import MPMModelBuilder
    from mpm.mpm_simulator import Simulator
    wp.config.kernel_cache_dir = '/cache/warp'
    wp.init()

    @wp.kernel
    def identity_stress(result: wp.array(dtype=wp.mat33), singular: wp.array(dtype=wp.vec3)):
        identity = wp.mat33(1., 0., 0., 0., 1., 0., 0., 0., 1.)
        U, s, V = wp.svd3(identity)
        rotation = U * wp.transpose(V)
        # Type-0 elastic branch of the unchanged legacy kernel, F=I, J=1.
        stress = (identity - rotation) * wp.transpose(identity) * 2000.
        result[0] = U
        result[1] = V
        result[2] = rotation
        result[3] = stress
        singular[0] = s

    matrices = wp.zeros(4, dtype=wp.mat33, device='cuda')
    singular = wp.zeros(1, dtype=wp.vec3, device='cuda')
    wp.launch(identity_stress, dim=1, inputs=[matrices, singular], device='cuda')
    builder = MPMModelBuilder()
    builder.set_mpm_domain([.6, .2, .2], grid_length=.005)
    builder.add_mpm_grid(pos=(-.035, 0., .1), vel=(0., 0., 0.),
        dim_x=3, dim_y=3, dim_z=3, cell_x=.003, cell_y=.003, cell_z=.003,
        density=1000., mu_lambda_ys=(1000., 1000., 10000.),
        friction_cohesion=(0., 0., 0.), type=0, jitter=False)
    model = builder.finalize('cuda')
    model.gravity = np.zeros(3, np.float32)
    model.struct.ground_normal = wp.vec3(0., 0., 1.)
    model.struct.particle_radius = .0015
    model.struct.body_sticky = 1
    model.struct.body_mu = 0.
    model.grid_contact = True
    model.particle_contact = True
    model.adaptive_grid = False
    states = [model.state() for _ in range(5)]
    builder.init_model_state(model, states)
    simulator = Simulator(device='cuda')
    fields = dict(x='particle_q', v='particle_qd', F='particle_F', C='particle_C')
    snapshots = {k: [] for k in fields}
    errors = []
    for step in range(101):
        for key, field in fields.items():
            snapshots[key].append(getattr(states[0].struct, field).numpy().copy())
        errors.append(states[0].struct.error.numpy().copy())
        if step < 100:
            for i in range(4):
                simulator.simulate(model, states[i], states[i + 1], .0005)
            states = [states[-1], *states[:-1]]
    arrays = {k: np.array(v) for k, v in snapshots.items()}
    arrays.update(error=np.array(errors), svd_matrices=matrices.numpy(), singular_values=singular.numpy(),
        mass=model.struct.particle_mass.numpy())
    np.savez_compressed(output / 'trace.npz', **arrays)
    paths = ['mpm/mpm_integrator.py', 'mpm/mpm_model.py', 'mpm/mpm_simulator.py', 'warp/bin/warp.so']
    report = dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files={p: hashlib.sha256((source / 'warp_maniskill' / p).read_bytes()).hexdigest() for p in paths},
        arrays_sha256=hashlib.sha256((output / 'trace.npz').read_bytes()).hexdigest(),
        steps=400, dt=.0005, particles=int(model.struct.n_particles), body_count=int(model.body_count),
        scope='Stationary authored diagnostic, no SAPIEN, no collision shapes or GPU coupling adapter; no threshold calibration or parity claim')
    (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)

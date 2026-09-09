"""Native-entity fallback and partial visual-pool regressions (no simulation)."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import sapien
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mani_skill.envs.softbody.particle_visuals import ParticleVisualPool


class ParticleVisualTests(unittest.TestCase):
    def test_fallback_replaces_stale_poses_without_touching_inactive_rows(self):
        positions = np.array([[.1, .2, .3], [.4, .5, .6], [1., 2., 3.]], dtype=np.float32)
        particle_q = SimpleNamespace(device='cpu', numpy=lambda: positions.copy())
        coupler = SimpleNamespace(states=[SimpleNamespace(struct=SimpleNamespace(particle_q=particle_q))])
        for buffer, update_entities in ((False, True), (True, True), (True, False)):
            with self.subTest(buffer=buffer, update_entities=update_entities):
                pool = ParticleVisualPool(None)
                pool.entities = [sapien.Entity() for _ in range(3)]
                for entity in pool.entities:
                    entity.pose = sapien.Pose([-8., -8., -8.])
                pool.active = pool.entities[:2]
                poses = torch.full((3, 7), -5.) if buffer else None
                saved = positions.copy()
                pool.update(coupler, poses, update_entities=update_entities)
                np.testing.assert_array_equal(positions, saved)
                for index, entity in enumerate(pool.entities):
                    expected = positions[index] if update_entities and index < 2 else [-8., -8., -8.]
                    np.testing.assert_array_equal(entity.pose.p, expected)
                if poses is not None:
                    np.testing.assert_array_equal(poses[:2, :3].numpy(), positions[:2])
                    np.testing.assert_array_equal(poses[:, 3:].numpy(), np.full((3, 4), -5.))
                    np.testing.assert_array_equal(poses[2].numpy(), np.full(7, -5.))


if __name__ == '__main__':
    unittest.main()

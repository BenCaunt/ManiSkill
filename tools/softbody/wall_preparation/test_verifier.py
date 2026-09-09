"""Negative controls for coverage and unsampled geometry failures."""
import copy
import itertools
import json
from pathlib import Path
import unittest
import numpy as np
from .verify import (cavity_clearance,load,verify_inventory,
                            verify_native_piece,verify_profile_partition)

ROOT=Path(__file__).parent
PROTOCOL=dict(json.loads((ROOT/'limits.json').read_text()),manifest_sha256='a'*64)

class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.profile=np.array([[0,0],[2,0],[2,1],[1,1],[1,2],[0,2]],float)
        self.cells=[[0,1,2,3],[0,3,4,5]]
    def test_concave_material_with_convex_cover(self):
        _,difference,overlap=verify_profile_partition(self.profile,self.cells,PROTOCOL)
        self.assertEqual(difference,0);self.assertEqual(overlap,0)
    def test_missing_region(self):
        with self.assertRaisesRegex(ValueError,'gaps or overlaps'):
            verify_profile_partition(self.profile,self.cells[:1],PROTOCOL)
    def test_duplicate_region(self):
        with self.assertRaisesRegex(ValueError,'gaps or overlaps'):
            verify_profile_partition(self.profile,self.cells+self.cells[:1],PROTOCOL)
    def test_concave_cell(self):
        with self.assertRaisesRegex(ValueError,'Nonconvex'):
            verify_profile_partition(self.profile,[list(range(6))],PROTOCOL)
    def test_reversed_cell(self):
        with self.assertRaisesRegex(ValueError,'reversed'):
            verify_profile_partition(self.profile,[self.cells[0][::-1],self.cells[1]],PROTOCOL)
    def test_inward_wall_obstruction_between_height_samples(self):
        # A 2 micrometre slab intersects the cavity, including at the axis.
        slab=np.array(list(itertools.product([-.01,.01],[-.01,.01],[.100001,.100003])))
        self.assertEqual(cavity_clearance(slab,.012,.17),0.)
    def test_clipped_slanted_wall(self):
        # Closest radial point occurs on the clipping plane, not a source vertex.
        vertices=np.array([[x,y,z] for z in [0.,1.] for x in [z,z+.1] for y in [-.1,.1]])
        self.assertAlmostEqual(cavity_clearance(vertices,.25,.75),.25)
    def test_material_outside_cavity_height(self):
        vertices=np.array(list(itertools.product([0.,1.],[0.,1.],[0.,.01])))
        self.assertIsNone(cavity_clearance(vertices,.012,.17))

class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.manifest=dict(sectors=32,cells=[[] for _ in range(12)],pieces=[dict(name=f's{s:02d}_c{c:02d}',sector=s,cell=c) for s in range(32) for c in range(12)])
        self.execution=dict(input_manifest_sha256='a'*64,cases=[dict(case=c) for c in ['cpu','gpu-data']])
    def test_complete(self):verify_inventory(self.manifest,self.execution,PROTOCOL)
    def test_missing_sector(self):
        self.manifest['pieces']=self.manifest['pieces'][12:]
        with self.assertRaisesRegex(ValueError,'coverage'):verify_inventory(self.manifest,self.execution,PROTOCOL)
    def test_duplicate_piece(self):
        self.manifest['pieces'].append(self.manifest['pieces'][0])
        with self.assertRaisesRegex(ValueError,'coverage'):verify_inventory(self.manifest,self.execution,PROTOCOL)
    def test_missing_cpu_case(self):
        self.execution['cases']=self.execution['cases'][1:]
        with self.assertRaisesRegex(ValueError,'native cases'):verify_inventory(self.manifest,self.execution,PROTOCOL)
    def test_duplicate_gpu_case(self):
        self.execution['cases'].append(self.execution['cases'][1])
        with self.assertRaisesRegex(ValueError,'native cases'):verify_inventory(self.manifest,self.execution,PROTOCOL)
    def test_renamed_mesh(self):
        self.manifest['pieces'][0]['name']='s00_c01'
        with self.assertRaisesRegex(ValueError,'piece name'):verify_inventory(self.manifest,self.execution,PROTOCOL)

class NativeGeometryTests(unittest.TestCase):
    def setUp(self):
        vertices=np.array(list(itertools.product([.02,.03],[-.005,.005],[.02,.03])),dtype=np.float32)
        polygons=[[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]]
        planes=np.array([[-1,0,0,.02],[1,0,0,-.03],[0,-1,0,-.005],[0,1,0,-.005],[0,0,-1,.02],[0,0,1,-.03]],dtype=np.float32)
        self.actual=dict(vertices=vertices,planes=planes,polygon_indices=np.array(polygons,dtype=np.uint32).ravel(),polygon_offsets=np.arange(0,25,4,dtype=np.uint32))
        self.ideal=vertices.astype(float)
    def evaluate(self,gpu=True):return verify_native_piece(self.actual,self.ideal,2.774447118400538e-8,PROTOCOL,gpu)
    def test_unchanged_native(self):self.assertEqual(self.evaluate()[1],[])
    def test_native_hull_translation(self):
        translation=np.array([0.,0.,.001])
        self.actual['vertices']+=translation
        self.actual['planes'][:,3]-=self.actual['planes'][:,:3]@translation
        self.assertIn('Solid union geometry bound exceeded',self.evaluate()[1])
    def test_native_planes_only_changed(self):
        self.actual['planes'][:,3]-=.0001
        failures=self.evaluate()[1]
        self.assertIn('Native plane/vertex disagreement',failures)
        self.assertIn('Solid union geometry bound exceeded',failures)
    def test_gpu_incompatible(self):self.assertIn('GPU incompatible',self.evaluate(False)[1])
    def test_nonfinite_native(self):
        self.actual['vertices'][0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'Nonfinite'):self.evaluate()

if __name__=='__main__':unittest.main()

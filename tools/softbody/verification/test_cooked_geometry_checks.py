import copy

import numpy as np
import pytest

from tools.softbody.verification.cooked_geometry_checks import check_model


def fixture():
    vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
    triangles=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.uint32)
    ref=dict(vertices=vertices,planes=np.array([[0,0,-1,0],[0,-1,0,0],[-1,0,0,0],[1,1,1,-1]],np.float32),
             polygon_indices=triangles.ravel(),polygon_offsets=np.arange(0,13,3,dtype=np.uint32))
    model={'source_shape_count':np.array([1,1])}
    for i in range(2):
        prefix=f'cooked/{i}/0/'
        model.update({prefix+k:v.copy() for k,v in ref.items()})
        model.update({prefix+'cached_vertices':vertices.copy(),prefix+'cached_triangles':triangles.copy(),
                      prefix+'flags':np.array([True,True]),prefix+'properties':np.array([300,1,1,0]),
                      prefix+'scale':np.ones(3),prefix+'pose':np.array([0,0,0,1,0,0,0]),
                      prefix+'mesh_aabb':np.array([[0,0,0],[1,1,1]])})
    return model,{'leaves':['piece']},{'piece':ref}


def test_each_native_and_cached_shape_matches_cooking():
    result=check_model(*fixture(),2,1e-6)
    assert not result['failures'] and result['pieces']==2 and result['max_bounds_error_m']==0


@pytest.mark.parametrize('field',['vertices','planes','polygon_indices','polygon_offsets','cached_vertices',
    'cached_triangles','flags','properties','scale','pose','mesh_aabb'])
def test_changed_second_row_native_fields_cannot_pass(field):
    model,pack,refs=fixture();key='cooked/1/0/'+field
    if field=='flags':model[key][0]=False
    else:model[key].flat[0]+=1
    assert check_model(model,pack,refs,2,1e-6)['failures']


def test_changed_count_fails():
    model,pack,refs=fixture();model['source_shape_count'][1]=2
    assert check_model(model,pack,refs,2,1e-6)['failures']

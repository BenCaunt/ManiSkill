"""Synthetic fault controls; these do not substitute for native camera evidence."""
from copy import deepcopy

import numpy as np
import pytest

from tools.softbody.verification.pour_visibility_checks import compare_views
from tools.softbody.verification.test_gpu_rendering_checks import frame


@pytest.fixture
def views(frame):
    hidden, _, limits = frame
    limits.update(minimum_explained_pixel_fraction=.95,require_native_buffer_checks=True)
    hidden.update(rigid_visual_ids=np.array([10]),rigid_visual_visibility=np.array([0.]),
        before_native_rigid=np.zeros((2,13)),after_native_rigid=np.zeros((2,13)))
    default=deepcopy(hidden)
    default['rigid_visual_visibility'][:]=1.
    default['segmentation'][:]=10
    default['depth'][:]=90
    return default,hidden,deepcopy(default),limits


def test_correct_occluder_and_physical_spheres(views):
    result=compare_views(*views)
    assert not result['failures']
    assert result['occluded_particle_pixels']==16


@pytest.mark.parametrize('key', ['particle_x','before/mpm/x','before_native_rigid','cam2world_gl'])
def test_inspection_cannot_change_physics_or_camera(views,key):
    default,hidden,restored,limits=views
    hidden[key].flat[0]+=.01
    if key == 'particle_x':
        hidden['before/mpm/x']=hidden[key][None].copy()
        hidden['after/mpm/x']=hidden[key][None].copy()
    result=compare_views(default,hidden,restored,limits)
    assert any('physical state or camera changed' in f for f in result['failures'])


@pytest.mark.parametrize('key', ['rgb','depth','position','segmentation','rigid_visual_visibility'])
def test_default_view_is_restored_exactly(views,key):
    views[2][key].flat[0]+=1
    assert any('Restoring visibility changed' in f for f in compare_views(*views)['failures'])


def test_wrong_occluder_is_not_an_explanation(views):
    views[0]['segmentation'][:]=9
    views[2]['segmentation'][:]=9
    assert any('not explained' in f for f in compare_views(*views)['failures'])


def test_occluder_behind_the_particle_is_not_an_explanation(views):
    views[0]['depth'][:]=150
    views[2]['depth'][:]=150
    assert any('not explained' in f for f in compare_views(*views)['failures'])


def test_missing_particle_pixels_still_fail(views):
    views[1]['segmentation'][:]=9
    assert 'Too few actual particle pixels' in compare_views(*views)['failures']


def test_hidden_body_pixels_are_rejected(views):
    views[1]['segmentation'][0,0,0]=10
    assert 'Hidden rigid visuals remain in the camera image' in compare_views(*views)['failures']

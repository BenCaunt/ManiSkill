import copy

import numpy as np
import pytest

from .gpu_rng_checks import check_reset,validate_snapshot


def snapshot(seeds):
    data={'rng/main_seed':np.array(seeds),'rng/episode_seed':np.array(seeds)}
    for name,values in [('main',seeds),('episode',seeds),('legacy_main',seeds[:1])]:
        states=[np.random.RandomState(seed).get_state() for seed in values]
        for field,column in [('words',1),('cursor',2),('has_gauss',3),('gaussian',4)]:
            data[f'rng/{name}/{field}']=np.array([v[column] for v in states])
    return data


def test_partial_reset_checks_selected_seed_and_untouched_stream():
    before=snapshot([101,17,1]);after=snapshot([101,71,1])
    assert not check_reset(before,after,[1],[71],3)


def test_original_global_reseed_is_rejected():
    before=snapshot([101,17,1]);seed=71
    after=snapshot([seed,*np.random.RandomState(seed).randint(2**31,size=2)])
    failures=check_reset(before,after,[1],[seed],3)
    assert any('requested' in f for f in failures)
    assert any('unselected' in f for f in failures)


@pytest.mark.parametrize('key', ['rng/main_seed','rng/episode_seed','rng/main/words','rng/episode/cursor',
                                'rng/main/has_gauss','rng/episode/gaussian','rng/legacy_main/words'])
def test_unchosen_stream_mutation_is_detected_even_without_seed_change(key):
    before=snapshot([101,17]);after=snapshot([101,71])
    # Keep valid types/ranges while perturbing a single unchosen field.
    after[key].flat[0]=after[key].flat[0]-1 if key.endswith('/cursor') else after[key].flat[0]+1
    assert check_reset(before,after,[1],[71],2)


def test_selected_draw_position_is_checked():
    after=snapshot([101,71]);after['rng/episode/cursor'][1]-=1
    assert any('requested seed' in f for f in check_reset(None,after,[1],[71],2))


def test_reversed_selection_maps_seeds_to_correct_environment():
    after=snapshot([19,23,17])
    assert not check_reset(None,after,[2,0,1],[17,19,23],3)
    assert check_reset(None,after,[0,1,2],[17,19,23],3)


@pytest.mark.parametrize('key,value',[('rng/main/words',-1),('rng/episode/cursor',625),('rng/main/has_gauss',2),
                                    ('rng/episode/gaussian',float('nan'))])
def test_malformed_rng_records_fail_closed(key,value):
    data=snapshot([101,17]);data[key]=data[key].astype(float);data[key].flat[0]=value
    with pytest.raises(ValueError):validate_snapshot(data,2)

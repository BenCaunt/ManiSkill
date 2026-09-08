"""Independent checks of declared seeds and exact RNG stream isolation."""
import numpy as np


FIELDS=('words','cursor','has_gauss','gaussian')


def expected_state(seed):
    _,words,cursor,has_gauss,gaussian=np.random.RandomState(seed).get_state()
    return dict(words=words,cursor=np.asarray(cursor),has_gauss=np.asarray(has_gauss),gaussian=np.asarray(gaussian))


def validate_snapshot(data,count):
    for name in ('main_seed','episode_seed'):
        value=data['rng/'+name]
        if value.shape!=(count,) or value.dtype.kind not in 'iu' or np.any((value<0)|(value>=2**32)):
            raise ValueError('Invalid exposed RNG seed array')
    for name,size in [('main',count),('episode',count),('legacy_main',1)]:
        for field in FIELDS:
            value=data[f'rng/{name}/{field}'];shape=(size,624) if field=='words' else (size,)
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError('Invalid RNG stream layout')
            if field=='words' and (value.dtype.kind not in 'iu' or np.any((value<0)|(value>=2**32))):
                raise ValueError('Invalid MT19937 words')
            if field=='cursor' and (value.dtype.kind not in 'iu' or np.any((value<0)|(value>624))):
                raise ValueError('Invalid MT19937 cursor')
            if field=='has_gauss' and not np.isin(value,[0,1]).all():raise ValueError('Invalid RNG Gaussian cache flag')


def check_reset(before,after,selected,seeds,count):
    if (len(selected)!=len(seeds) or len(set(selected))!=len(selected)
            or any(i<0 or i>=count for i in selected)):
        raise ValueError('Invalid RNG reset selection')
    validate_snapshot(after,count)
    failures=[]
    for index,seed in zip(selected,seeds):
        expected=expected_state(seed)
        for name in ('main_seed','episode_seed'):
            if after['rng/'+name][index]!=seed:failures.append(f'env{index}: reset did not use its requested {name}')
        for name in ('main','episode'):
            for field,value in expected.items():
                if not np.array_equal(after[f'rng/{name}/{field}'][index],value):
                    failures.append(f'env{index}: {name} stream differs from its requested seed: {field}')
        if index==0:
            for field,value in expected.items():
                if not np.array_equal(after[f'rng/legacy_main/{field}'][0],value):
                    failures.append('Legacy main RNG differs from environment zero seed: '+field)
    if before is not None:
        validate_snapshot(before,count)
        untouched=[i for i in range(count) if i not in selected]
        for key in before:
            if not key.startswith('rng/'):continue
            rows=([0] if 0 in untouched else []) if key.startswith('rng/legacy_main/') else untouched
            if not np.array_equal(before[key][rows],after[key][rows]):
                failures.append('Partial reset changed an unselected RNG stream: '+key)
    return failures


def check_sequence(case,data,steps):
    count=len(case['seeds']);failures=[]
    for label in ('initial','reconfigured'):
        failures.extend(label+': '+f for f in check_reset(None,data[label],list(range(count)),case['seeds'],count))
    predecessors={'warmup-1':'initial','warmup-2':'warmup-1','expected':'warmup-2','continued':'expected',
                  'partial-stepped':'partial-restored','count-stepped':'partial-count','flat-stepped':'flat-restored'}
    for step in steps:
        label=step['label'];before=data[predecessors[label]];after=data[label]
        validate_snapshot(after,count)
        for key in before:
            if key.startswith('rng/') and not np.array_equal(before[key],after[key]):
                failures.append(label+': deterministic control changed RNG state: '+key)
    if count>1:
        for before,after,index,seed in [('expected','partial-restored',0,33),
            ('partial-stepped','partial-fresh',1,17),('partial-fresh','partial-count',0,34)]:
            failures.extend(after+': '+f for f in check_reset(data[before],data[after],[index],[seed],count))
        selected=list(reversed(range(count))) if case.get('reverse_flat_indices') else list(range(count))
        failures.extend('flat-restored: '+f for f in check_reset(None,data['flat-restored'],selected,list(range(31,31+count)),count))
    return failures

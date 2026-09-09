"""Aggregation must preserve late-row failures and reject changed study inputs."""
from copy import deepcopy
import pytest
from analyze_gpu_scaling import check_request, summarize_verdict


def example():
    case = dict(env_id='Fill-v0', num_envs=32, seeds=list(range(32)), timed_controls=5)
    protocol = dict(image='sha256:'+'a'*64, job_timeout_s=2400,
        candidate_commit='b'*40, native_actor_extension={'sha256': 'c'*64},
        runtime_source_files={'scene.py': 'd'*64}, harness_files={'scaling_checks.py': 'e'*64})
    request = dict(role='scaling', scaling_case=case, env_id='Fill-v0', candidate_sim_backend='physx_cuda',
        image=protocol['image'], timeout_s=2400, native_actor_extension=protocol['native_actor_extension'],
        source_checkout_commit=protocol['candidate_commit'], file_sha256=dict(
            source=protocol['runtime_source_files'], harness={'softbody_lab/scaling_checks.py': 'e'*64}))
    return deepcopy(request), protocol, case


@pytest.mark.parametrize('change', ['source', 'harness', 'case', 'image', 'native'])
def test_changed_study_input_rejected(change):
    request, protocol, case = example()
    check_request(request, protocol, case)
    if change in ('source', 'harness'): request['file_sha256'][change] = {}
    elif change == 'case': request['scaling_case']['seeds'][-1] = 999
    elif change == 'image': request['image'] = 'sha256:'+'f'*64
    else: request['native_actor_extension']['sha256'] = 'f'*64
    with pytest.raises(ValueError): check_request(request, protocol, case)


def test_late_unselected_row_and_nonreset_failure_are_retained():
    verdict = dict(passed=False, failures=['partial-fresh/env31: changed native/31/rigid', 'measured: invalid mass'],
        exact_resets={'partial-fresh': {'31': dict(selected=False, fields=['native/31/rigid'],
            max_abs_errors={'native/31/rigid': .01})}})
    result = summarize_verdict(verdict)
    assert result['unselected_changed_rows'] == 1
    assert result['unselected_changes'][0]['env_index'] == 31
    assert result['other_failures'] == ['measured: invalid mass']
    assert result['failures'] == 2 and result['strict_passed'] is False


def test_selected_precision_difference_still_fails_when_untouched_rows_pass():
    verdict = dict(passed=False, failures=['flat-restored/env31: changed native/31/rigid'],
        exact_resets={'flat-restored': {'31': dict(selected=True, fields=['native/31/rigid'],
            max_abs_errors={'native/31/rigid': 1e-7})}})
    result = summarize_verdict(verdict)
    assert result['unselected_changed_rows'] == 0
    assert result['selected_changed_rows'] == result['failures'] == 1
    assert result['max_selected_abs_error'] == 1e-7
    assert result['strict_passed'] is False

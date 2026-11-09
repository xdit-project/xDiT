"""Group A: process A and process B make one result per case; each test compares the results of one case."""

import pytest

import harness

CASES = [1, 2]


@pytest.fixture(scope="module")
def group_data(request):
    """Run the expensive processes once for all selected cases of this group: {case: (result A, result B)}."""
    cases = harness.selected_cases(request)
    results_a = harness.run_a(request, cases)
    results_b = harness.run_b(request, cases)
    return dict(zip(cases, zip(results_a, results_b, strict=True), strict=True))


@pytest.mark.parametrize("case", CASES, ids=lambda case: f"case{case}")
def test_compare(group_data, case):
    result_a, result_b = group_data[case]
    print(result_a)
    print(result_b)
    assert result_a == f"mytest_A{case}"
    assert result_b == f"mytest_B{case}"

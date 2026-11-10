# Checks in the test function

This document describes what the test function of one case checks, in which order, and how it reports the results.

## 1. Inputs of the test function

1. The case fixture gives the test function the case state ([files.md](files.md), section 4). The state holds the
   statuses of A and B for each call and rank, the warnings, and the paths of the data files.
2. The test function reads the files from these paths. It does not compute paths itself.

## 2. Immediate outcomes

The test function first handles the statuses that make the data unusable. It does this before it computes any
metric.

1. **Statuses without data.** The case gets the outcome from [statuses.md](statuses.md) at once. Examples:
   UNSUPPORTED, FAILED_TO_START, CRASHED_A, HUNG_B, ERROR_A.
2. **NON_DETERMINISTIC_A or NON_DETERMINISTIC_B.**
   1. Without the determinism override of that instance, the test fails at once.
   2. With the override, the test prints a warning and continues with section 4. The warning reminds the user that
      the override is active. It also lists the calls whose fidelity is unknown.
3. **REPLAY_MISMATCH_A or REPLAY_MISMATCH_B.** The test fails at once, with or without the override.
4. **NOT_REACHED_B.** The test fails at once.
5. **Message.** The instance knows these statuses before the test function starts. The failure message therefore
   lists every call and rank that has one of these statuses.
6. **Metrics file.** A case that stops here writes one line with its outcome and no metric values
   ([files.md](files.md), section 9).

## 3. The determinism override

1. Each instance has its own override: `expect_deterministic_a` and `expect_deterministic_b`
   ([test_definition.md](test_definition.md), section 2.2).
2. The bitwise checks always run, also with the override.
3. The override changes only one result: nondeterminism gives a warning instead of a failure.
4. REPLAY_MISMATCH always fails.

## 4. Deferred checks

### 4.1 Principle

1. The test function runs all checks that the data permits, before it fails. A failure of one check does not stop
   the other checks.
2. The test function uses the `pytest-check` package for this. A failed `check` records the failure and continues.
   pytest reports the test as failed at the end and lists all failures.
3. The default limits of `pytest-check` report all failures: `--check-max-report` and `--check-max-fail` have no
   default value
   ([plugin.py:212-223](../../../../../venv/lib/python3.12/site-packages/pytest_check/plugin.py#L212-L223),
   `pytest-check` 3.0.3).

### 4.2 The checks

For each selected call of the case:

1. **The call occurred.** A selected call that did not occur in A is a failure.
2. **Output names.** The names of the tensor outputs must be equal to the names in the tolerances
   ([test_definition.md](test_definition.md), section 5.1). Each missing or extra name is a failure. The outputs
   with matching names are still checked.
3. **Non-tensor outputs.** Each non-tensor leaf must be exactly equal in A and B.
4. **Shapes.** Each tensor output must have the same shape in A and B. For an output with different shapes, the
   test function skips the metrics of that output. The other outputs are still checked.
5. **NaN and Inf.**
   1. NaN or Inf in an output of A is a failure.
   2. NaN or Inf in an output of B is a failure with the text "reference problem".
6. **Tolerances.** For each tensor output and each rank, the test function checks:
   1. `max_abs <= tolerance.max_abs`;
   2. `rel_l2 <= tolerance.rel_l2`.

   Each rank is checked separately against the same tolerance. A failure names the call address, the output, the
   metric, the rank, the value, and the tolerance.

## 5. Metric definitions

1. The test function computes the metrics on the CPU, in float64, for each rank and each call.
2. `a` is the captured output of A. `b` is the output of the first replay of B. Both are converted to float64.
3. `max_abs` is the maximum of `|a - b|` over all elements.
4. `rel_l2` is `||a - b||_2 / ||b||_2`. If `||b||_2` is 0, `rel_l2` is 0 when `||a - b||_2` is 0, and infinity in
   the other cases.
5. `cosine` is `<a, b> / (||a||_2 * ||b||_2)`. If a norm is 0, `cosine` is NaN. The test function reports `cosine`
   but does not check it.

## 6. Reports

1. **`record_property`.** For each call, output and metric, the test function adds one property with the values of
   all ranks. JUnit XML then contains the metrics. The case status is also a property.
2. **Metrics file.** The test function appends the metrics to the metrics file ([files.md](files.md), section 9).
3. **Warnings.** The test function prints the warnings of the dtype audit of B and the reminder of an active
   determinism override.

## 7. End of the test function

1. The test function asks `pytest-check` whether any check failed (`any_failures()`).
2. If no check failed, the test function deletes the data files of the case and writes the state PASSED.
3. If a check failed, the test function keeps the data files and writes the state FAILED.
4. The failure itself comes from `pytest-check` after the test function returns.

# Changes to the pytest_integration prototype

The prototype in [_prototypes/pytest_integration/](../_prototypes/pytest_integration/) shows how a group of tests
can share two expensive processes, select single cases, show live output, and leave no processes behind. The
harness takes the parts that fit and changes them as this document lists. The prototype is an implementation hint,
not a binding rule. The section numbers refer to its [README.md](../_prototypes/pytest_integration/README.md).

## 1. Parts that the harness keeps

1. `selected_cases()` ([harness.py:27-39](../_prototypes/pytest_integration/harness.py#L27-L39)): the instances get
   only the selected cases.
2. `console_writer()` ([harness.py:42-52](../_prototypes/pytest_integration/harness.py#L42-L52)): live output in
   every capture mode.
3. `run_in_pty()` and the subreaper supervisor in [terminal.py](../_prototypes/pytest_integration/terminal.py): one
   ordered output stream and no leftover processes. The tests in `terminal_tests/` go with it.
4. The module-scoped group fixture that runs the expensive processes once for each group (README section 3.2).
5. Groups run one after another, and A and B of one group run one after another (README section 2.1, item 3).

## 2. Required changes

1. **A launcher for each instance.**
   1. Today the fixture runs the process directly, and a nonzero exit raises `subprocess.CalledProcessError`
      ([harness.py:101-103](../_prototypes/pytest_integration/harness.py#L101-L103)). README section 3.2, item 6
      says that every test of the group then reports ERROR.
   2. torchrun exits with a nonzero status when a worker crashes. The harness needs an outcome for each case, also
      after a crash.
   3. Change: the fixture runs a launcher, and the launcher runs torchrun. The launcher survives crashes and hangs
      of the workers and exits with 0 when it wrote its records. Only a nonzero exit of the launcher raises in the
      fixture ([processes.md](processes.md), section 2).
2. **Data through files.**
   1. Today each process writes one JSON array of result strings into a shared file in the prototype directory
      (`a_result.json`, `b_result.json`), and the harness adds the prefix `mytest_` (README section 2.2 and 2.3).
   2. Change: progress records, case states and tensors go into the configuration directory
      ([files.md](files.md)). The JSON result strings, the prefix and the shared result files go away. This also
      removes the conflict of shared result files in README section 5, item 9.
3. **Deadlines in the launcher.**
   1. README section 5, item 2 says: "There is no overall time limit."
   2. Change: the launcher enforces the deadlines, because it is the parent of torchrun
      ([processes.md](processes.md), section 4). The supervisor still kills all leftover processes. Thus
      `run_in_pty()` needs no stop handle.
4. **B only after a clean A.**
   1. Today the fixture runs both processes without conditions
      ([test_group_a.py:13-16](../_prototypes/pytest_integration/test_group_a.py#L13-L16)).
   2. Change: the fixture starts B only if the instance status of A is CLEAN.
5. **Cases from the configuration call.**
   1. Today each group file has a list `CASES` of integers, and the case IDs are `case1`, `case2` and so on (README
      section 6.1, item 5).
   2. Change: the cases come from the configuration call, and the case IDs are the component paths
      ([component_paths.md](component_paths.md), section 6).
6. **Shared fixtures.**
   1. Today each group file holds its own fixture (README section 2.6, item 1).
   2. Change: the group fixture and the case fixture live in `tests/component/conftest.py`. The test module defines
      only the configuration and the test function. The configuration call is the place for customization of a
      group.
7. **Case fixture and outcomes.**
   1. Today a test compares two strings with `assert`.
   2. Change: a function-scoped case fixture maps the case state to the outcomes "error" and "skipped", and the test
      function does the checks of [checks.md](checks.md) with `pytest-check`.
8. **The configuration directory replaces the old result files.**
   1. Today the harness deletes the old result file before each run (README section 3.2, item 4.1).
   2. Change: the group fixture handles the configuration directory with the rules of fresh runs, reuse runs, the
      marker file and the lock ([files.md](files.md), sections 5 and 6).
9. **Grace period.**
   1. Today the default is 5 seconds (`terminal.DEFAULT_GRACE_PERIOD`, README section 6.1, item 3).
   2. Change: GPU processes need a longer grace period. The harness uses 30 seconds (Not confirmed).
10. **Header line.** The header line before each process names the configuration, the instance and the selected
    cases (README section 2.3, item 5).
11. **Mock processes and switches go away.** `process_a.py`, `process_b.py`, `TEST_CASE` and `FAIL_PROCESS`
    (README section 2.2) are only for the demonstration.
12. **No `pytest.ini` of its own.** README section 2.6, item 4 says "No `pytest.ini`". The xDiT root has a
    [pytest.ini](../../../pytest.ini) whose `testpaths` do not include `tests/component`. Run the component tests
    with explicit paths ([pytest_integration.md](pytest_integration.md), section 1).
13. **Module name `harness`.** The prototype module `harness.py` and the harness package `harness/` have the same
    name. One pytest session must not collect both ([pytest_integration.md](pytest_integration.md), section 1,
    item 5).

## 3. Limitations of the prototype that stay

1. The internal pytest API `global_and_fixture_disabled()` (README section 5, item 3).
2. The program always sees a terminal (README section 5, item 4). With eight ranks, progress bars can mix their
   `\r` updates.
3. If the supervisor gets SIGKILL, descendants that left the process group of pytest survive (README section 5,
   item 1).
4. Linux only (README section 5, item 7).

# pytest integration

This document describes how the harness uses pytest. The prototype in
[_prototypes/pytest_integration/](../_prototypes/pytest_integration/) is an implementation hint, not a binding
rule. [prototype_changes.md](prototype_changes.md) lists the changes that the harness needs.

## 1. Files

1. Test modules: `xDiT/tests/component/test_*.py`, one module for each configuration.
2. The harness package: `xDiT/tests/component/harness/`.
3. The shared fixtures: `xDiT/tests/component/conftest.py`. A test module only defines the configuration and the
   test function ([test_definition.md](test_definition.md), section 2.1).
4. [pytest.ini](../../../pytest.ini) does not collect `tests/component`. Run the component tests with explicit
   paths, for example `python -m pytest tests/component/test_wan22_i2v.py`.
5. The prototype also has a top-level module named `harness`
   ([harness.py](../_prototypes/pytest_integration/harness.py)). With the default import mode, pytest puts the
   directory of each test module on `sys.path`. If one session collects the prototype tests and the component
   tests, the second `import harness` gets the module of the first. Thus one pytest session must not collect both.
   `tests/component/conftest.py` must exclude `_prototypes` from collection, for example with
   `collect_ignore = ["_prototypes"]`.

## 2. Groups and cases

1. A group is one configuration, that is, one test module.
2. A case is one listed component path. The harness parametrizes the test function with the case IDs
   ([component_paths.md](component_paths.md), section 6).
3. The groups run one after another. A and B of one group run one after another.

## 3. Group fixture

1. The group fixture has module scope. The first selected test of the module starts it.
2. It gets the selected cases of the module from `request.session.items`, as `selected_cases()` of the prototype
   does ([harness.py:27-39](../_prototypes/pytest_integration/harness.py#L27-L39)). Thus the instances capture
   and replay only the selected cases. The pipeline runs are complete in all cases.
3. It handles the data directory ([files.md](files.md), section 5): a fresh run or a reuse run.
4. For a fresh run:
   1. It writes the state ABOUT_TO_START for each selected case.
   2. It runs the launcher of A.
   3. If the instance status of A is CLEAN, it runs the launcher of B.
5. It raises an exception only if a launcher exits with a nonzero status ([processes.md](processes.md),
   section 2). All cases of the group then get the outcome "error".
6. Before each launcher, it writes a header line with the configuration name, the instance and the selected cases.

## 4. Case fixture

1. The case fixture has function scope. It reads the state of one case.
2. It raises an exception for the statuses with the outcome "error" ([statuses.md](statuses.md), section 4).
   pytest reports an exception in a fixture as "error".
3. It calls `pytest.skip()` for the statuses with the outcome "skipped".
4. In the other cases, it gives the state to the test function.

## 5. Test function

1. The test function does the checks of [checks.md](checks.md).
2. It uses `pytest-check` for the deferred checks.
3. It adds the metrics and the status with `record_property`.

## 6. Live output

1. The group fixture runs each launcher through `run_in_pty()`. The output of the workers goes live to the console
   in every capture mode, through `console_writer()` of the prototype
   ([harness.py:42-52](../_prototypes/pytest_integration/harness.py#L42-L52)).
2. `console_writer()` uses the internal pytest API `CaptureManager.global_and_fixture_disabled()`. A future pytest
   version can change it. The prototype was tested with pytest 9.0.3.
3. All ranks share one pseudo-terminal. Progress bars that detect a terminal can mix their `\r` updates. This is
   only cosmetic.

## 7. Selection

1. `-k`, node IDs, `-m`, `--deselect` and `--lf` work, because `selected_cases()` reads the final list of tests.
2. `-k` matches substrings ([component_paths.md](component_paths.md), section 6).
3. Selection by a node ID that contains nested brackets, for example `test_component[pipe.transformer.blocks[0]]`,
   is not verified.

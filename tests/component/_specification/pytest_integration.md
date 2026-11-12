# pytest integration

This document describes how the harness uses pytest. The prototype in
[_prototypes/pytest_integration/](../_prototypes/pytest_integration/) is an implementation hint, not a binding
rule. [prototype_changes.md](prototype_changes.md) lists the changes that the harness needs.

## 1. Files

1. Test modules: `xDiT/tests/component/test_*.py`. By convention, one module for each xDiT model, with many or all
   configurations of that model ([test_definition.md](test_definition.md), section 1).
2. The harness package: `xDiT/tests/component/harness/`.
3. The shared fixtures: `xDiT/tests/component/conftest.py`. A test module only defines its configurations and
   calls `make_test` one time ([test_definition.md](test_definition.md), section 2.1).
4. [pytest.ini](../../../pytest.ini) does not collect `tests/component`. Run the component tests with explicit
   paths, for example `python -m pytest tests/component/test_wan22_i2v.py`.
5. The prototype also has a top-level module named `harness`
   ([harness.py](../_prototypes/pytest_integration/harness.py)). With the default import mode, pytest puts the
   directory of each test module on `sys.path`. If one session collects the prototype tests and the component
   tests, the second `import harness` gets the module of the first. Thus one pytest session must not collect both.
   `tests/component/conftest.py` must exclude `_prototypes` from collection, for example with
   `collect_ignore = ["_prototypes"]`.

## 2. Configurations and cases

1. A test module holds one or more configurations.
2. A case is one listed component path of one configuration. `make_test` parametrizes the test function with one
   pair (configuration, case) for each case. The case ID holds the configuration name and the path
   ([component_paths.md](component_paths.md), section 6).
3. The configurations run one after another. A and B of one configuration run one after another.
4. The tests of a configuration run after its instances and before the next configuration starts its instances.
   Thus the disk holds the dumps of at most one configuration, plus the files that failing tests keep
   ([files.md](files.md), section 8).

## 3. Configuration fixture

1. **One instance for each configuration.**
   1. `make_test` gives the configuration to the fixture as an indirect parameter with module scope
      (`indirect=["configuration"]` and `scope="module"` in `pytest.mark.parametrize`).
   2. pytest keeps one instance of the fixture for each parameter value. pytest compares the parameter of each test
      with the parameter of the current instance with `==`
      ([fixtures.py:1119-1120](../../../../../venv/lib/python3.12/site-packages/_pytest/fixtures.py#L1119-L1120)).
      If they differ, pytest tears down the current instance and starts a new one
      ([fixtures.py:1078-1096](../../../../../venv/lib/python3.12/site-packages/_pytest/fixtures.py#L1078-L1096)).
   3. `make_test` must list the pairs grouped by configuration. pytest does not regroup them: each pair has its own
      parameter index, and pytest sorts tests only by that index
      ([fixtures.py:213-214](../../../../../venv/lib/python3.12/site-packages/_pytest/fixtures.py#L213-L214)). With
      pairs in mixed order, pytest starts the fixture of one configuration more than one time.
   4. Configuration objects compare by identity. Reason: then the `==` of pytest never compares the nested settings
      of two configurations. Two different configurations can never be equal anyway, because their names are unique
      ([test_definition.md](test_definition.md), section 7).
2. **Second start.** The fixture keeps the set of names of the configurations that started in the pytest process.
   At its start, it raises an exception if its configuration is already in the set. Then every case of this second
   start gets the outcome "error". The message says that the configuration started a second time in this session,
   because pytest ran its cases in two parts, for example with `--ff` (section 7). The names are unique
   ([test_definition.md](test_definition.md), section 7), so the name identifies the configuration.
3. **Selected cases.** The fixture gets the selected cases of its configuration from `request.session.items`: the
   tests whose configuration parameter is its configuration. The prototype does the same for a module in
   `selected_cases()` ([harness.py:27-39](../_prototypes/pytest_integration/harness.py#L27-L39)). Thus the instances
   capture and replay only the selected cases. The pipeline runs are complete in all cases.
4. It takes the lock of the configuration, prepares the configuration directory, and decides between a fresh run
   and a reuse run ([files.md](files.md), sections 5 and 6).
5. For a fresh run:
   1. It writes the state ABOUT_TO_START for each selected case.
   2. It runs the launcher of A.
   3. If the instance status of A is CLEAN, it runs the launcher of B.
6. It raises an exception only for a second start (item 2), if another session holds the lock of the
   configuration, if the configuration directory is not empty and has no marker file ([files.md](files.md),
   section 6), or if a launcher exits with a nonzero status ([processes.md](processes.md), section 2). All cases
   of the configuration then get the outcome "error".
7. Before each launcher, it writes a header line with the configuration name, the instance and the selected cases.

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

1. The configuration fixture runs each launcher through `run_in_pty()`. The output of the workers goes live to the
   console in every capture mode, through `console_writer()` of the prototype
   ([harness.py:42-52](../_prototypes/pytest_integration/harness.py#L42-L52)).
2. `console_writer()` uses the internal pytest API `CaptureManager.global_and_fixture_disabled()`. A future pytest
   version can change it. The prototype was tested with pytest 9.0.3.
3. All ranks share one pseudo-terminal. Progress bars that detect a terminal can mix their `\r` updates. This is
   only cosmetic.

## 7. Selection

1. `-k`, node IDs, `-m`, `--deselect` and `--lf` work, because the configuration fixture reads the final list of
   tests (section 3, item 3). `--lf` keeps the order of the tests.
2. `-k` matches substrings ([component_paths.md](component_paths.md), section 6). For example, `-k gfx950` selects
   the cases of every configuration whose name contains `gfx950`.
3. Selection by a node ID that contains nested brackets, for example
   `test_component[wan2_2.quantgemm_quantattn.gfx950:pipe.transformer.blocks[0]]`, is not verified.
4. `--ff` (failed tests first) is not supported. It moves the failed tests to the front after pytest made the list
   of tests ([cacheprovider.py:369-399](../../../../../venv/lib/python3.12/site-packages/_pytest/cacheprovider.py#L369-L399)).
   The cases of one configuration can then run in two parts. The configuration fixture then starts a second time,
   and the check of a second start gives "error" to the cases of the second part (section 3, item 2).

# pytest integration demo for expensive per-group processes, per-case selection, live output, no leftover processes

This directory is a small, working prototype. It shows how to run test groups with pytest when each group
needs expensive external processes to setup the test cases, and each test is only a cheap comparison of their results.

This demo shall stay in the repo until the approach is verified to work and none of the tests here are needed anymore.

1. Each **group** runs process A, then process B, **once**. Each process produces one result per test case.
2. Each **test** compares the A and B results of **one case**.
3. You select groups **and individual cases** with the normal pytest command line. The processes then run only
   for the selected cases.
4. The output of the processes appears **live** on the console in every pytest capture mode, locally and in CI.
5. When a process run ends, **no process of the run survives**, even if pytest itself is killed.

Contents:

1. [Quick start](#1-quick-start)
2. [Requirements and restrictions](#2-requirements-and-restrictions)
3. [How it works](#3-how-it-works)
4. [Alternatives considered and rejected](#4-alternatives-considered-and-rejected)
5. [Limitations](#5-limitations)
6. [What you must know to use it](#6-what-you-must-know-to-use-it)
7. [Tests and verification](#7-tests-and-verification)

## 1. Quick start

Requirements: Linux with `/proc`, Python 3.10 or later, and pytest. It was tested only with Python 3.12.3,
pytest 9.0.3, and Linux 5.15. Run all commands in this directory.

| Command | Effect |
|---|---|
| `pytest` | Everything: the 3 demo groups, `terminal_tests/`, and `harness_output_tests/` |
| `pytest test_group_*.py` | The 3 demo groups only |
| `pytest test_group_b.py` | One group |
| `pytest "test_group_c.py::test_compare[case4]"` | One case: the processes receive only `4` |
| `pytest test_group_b.py -k "not case2"` | All cases of group B except case 2 |
| `pytest -s test_group_a.py` | Also shows what `test_compare` prints |
| `TEST_CASE=2 pytest test_group_*.py` | Process B marks case 2 as failed, so `test_compare[case2]` fails in every group |
| `FAIL_PROCESS=A pytest test_group_*.py` | Process A exits with status 1, so all tests of each group report ERROR (use `B` for process B) |

Output of `pytest test_group_a.py` (default capture, no `-s`):

```
test_group_a.py
[test_group_a] process_a.py 1 2
A1
A2
Case 2 chosen

[test_group_a] process_b.py 1 2
B1
B2
Case 2 chosen
..                                                       [ 20%]
```

The `[group] script cases` header lines come from the harness. The other lines are the live output of the
processes. With `-s`, each test also prints its two strings, for example `mytest_A1` and `mytest_B1`.

## 2. Requirements and restrictions

### 2.1 Test structure

1. There are several test groups, each with several tests. For a group with N tests, process A runs first and
   produces N results; then process B runs and produces N corresponding results. Both processes are
   expensive and need progress reporting. The comparison of the results is cheap.
2. The pytest command line must select not only groups but also individual test cases in a group.
3. Process A and process B cannot run in parallel, and different groups cannot run in parallel.
4. The processes can compute any subset of the cases of a group.
5. The demo has 3 groups, each in its own test file, with 2 to 5 tests per group (here 2, 3, and 5).

### 2.2 Mock processes

1. `run_a()` and `run_b()` start `process_a.py` and `process_b.py` as independent processes. They pass the
   case IDs (`1..N`) as command-line arguments.
2. Each process first gives each case ID the prefix `A` or `B` (`2` becomes `A2` or `B2`) and prints the cases
   to stdout. If case 2 is selected, it prints exactly `Case 2 chosen` to stderr. It writes the cases as a JSON
   array to `./a_result.json` or `./b_result.json`, and exits.
3. The mocks use only plain `print()`, to stdout and to stderr. They stand in for real processes that write to
   stdout and stderr from any layer, from ROCm internals up to the kernel. No special effort, such as flushing,
   is allowed inside the processes.
4. If `FAIL_PROCESS=A` or `FAIL_PROCESS=B` is set, that process prints its normal output and then exits with
   status 1 before it writes its result file.
5. If `TEST_CASE=<id>` is set (one value), process B gives that case the prefix `failed_` (`failed_B2`). This
   makes the assertion in the group test fail for that case.

### 2.3 Harness behavior

1. `run_*()` captures stdout and stderr of the process live and forwards them at once to the console of pytest.
2. If the process fails, `run_*()` raises an exception. Otherwise it reads the result file and gives each string
   the prefix `mytest_`.
3. `test_compare()` prints the two strings and asserts `a == f"mytest_A{case}"` and `b == f"mytest_B{case}"`.
4. Without `-s`, only the output of `run_*()` is visible, which emulates progress of long running computations.
   With `-s`, the prints of the test functions are visible too.
5. A header line before each process run shows the group, the script, and the cases.
6. The result files are written to this directory (the working directory of the processes).

### 2.4 Output fidelity

1. The output must be visible while the fixture runs, whatever the pytest configuration is. Configuration in code
   is allowed.
2. The forwarded output must be one combined stream, in correct order. The order within stdout and the order
   within stderr must be correct. A different relative order between stdout and stderr is tolerable.
3. Everything must work in CI in the same way as locally.
4. What you see while the fixture runs must match what you would see if you ran the processes directly.
5. The platform is Linux.

### 2.5 Process cleanup

1. However the run ends, including any abnormal end of the pytest process or of the processes under the
   supervisor, no process that the run started may stay alive.
2. Remaining processes get SIGTERM, then SIGKILL after a grace period. The grace period is a parameter with a
   default (5 seconds). It is passed to `run_*()` and from there to the supervisor process.

### 2.6 Code organization

1. Each group test file holds its own fixture, so that each group can be customized locally. Shared code
   lives in a common module that the group files import. Because they only pass cases as strings,
   `run_a()` and `run_b()` are shared. `selected_cases()` is shared too, and it must not know which cases
   belong to which group.
2. The pseudo-terminal is custom code and must be small. Its tests are in `terminal_tests/`.
3. The tests for the harness cover only the output, and they are in `harness_output_tests/`.
4. No `pytest.ini`. Tests in subdirectories import their module from the parent directory.

## 3. How it works

### 3.1 Files

| File | Purpose |
|---|---|
| [test_group_a.py](test_group_a.py), [test_group_b.py](test_group_b.py), [test_group_c.py](test_group_c.py) | One group each: `CASES`, the module-scoped fixture `group_data`, and `test_compare` |
| [harness.py](harness.py) | Shared code: case selection, console output past pytest's capture, `run_process`, `run_a`, `run_b` |
| [terminal.py](terminal.py) | Generic runner: pseudo-terminal, live output, subreaper supervisor. It does not depend on pytest. |
| [process_a.py](process_a.py), [process_b.py](process_b.py) | The mock processes |
| [terminal_tests/](terminal_tests/) | Tests for `terminal.py` |
| [harness_output_tests/](harness_output_tests/) | Tests that the harness output is visible in every capture mode |

### 3.2 Flow of one group

1. pytest collects `test_compare` with one test per case (`test_compare[case1]`, ...), and then applies the
   selection (`-k`, node IDs, and so on).
2. The first selected test of a module requests the module-scoped fixture `group_data`
   ([test_group_a.py:11-16](test_group_a.py#L11-L16)). pytest runs it once per module and reuses its value for the
   other tests of that module.
3. `group_data` calls `harness.selected_cases(request)` to get the selected cases of this module. Then it calls
   `harness.run_a(request, cases)` and `harness.run_b(request, cases)`.
4. For each process, the harness ([harness.py:91-104](harness.py#L91-L104)):
   1. Deletes the old result file. All groups use the same file, so a process that exits with 0 without writing
      the file must not hand over the data of the previous group.
   2. Writes the header line.
   3. Runs the process with `harness.run_process()`, which calls `terminal.run_in_pty()`. All output goes to the
      console while the process runs.
   4. Raises `subprocess.CalledProcessError` if the exit status is not 0.
   5. Reads the JSON file and gives each string the prefix `mytest_`.
5. `group_data` returns `{case: (result_a, result_b)}`. Each test takes its pair, prints it, and asserts it.
6. If the fixture raises, pytest caches the exception for the module. All tests of the group then report ERROR,
   and the processes do not run again for each test.

### 3.3 Case selection

[`selected_cases()`](harness.py#L27-L39) reads `request.session.items`. This is pytest's final list of tests, after
the `pytest_collection_modifyitems` hook, where `-k`, `-m`, `--deselect`, and `--lf` remove tests (see
`Session.perform_collect` in pytest's `main.py`). The function keeps the tests of the module that requested the
fixture and returns their `case` parameter values. It knows nothing about groups: the values come from the
`parametrize` call in the group file. A group file alone cannot know which of its cases are still selected,
because only the pytest session has that information.

### 3.4 Live output past pytest's capture

By default, pytest redirects fd 1 and fd 2 to temporary files while a test or fixture runs. Output written there
appears only in the report of a failed test. [`console_writer()`](harness.py#L42-L52) writes each chunk of process
output inside `CaptureManager.global_and_fixture_disabled()`. This context manager pauses pytest's global
capture and fixture capture for the duration of the write, so the bytes reach the real console. With `-s` it does
nothing, and with `-p no:capture` the harness writes directly. The output of the test functions keeps pytest's
normal capture rules. Section 3.8 explains why this internal API is used.

### 3.5 Ordered output: the pseudo-terminal

When stdout is a pipe, Python and C stdio **block-buffer** it, while Python's stderr is line-buffered (Python
`sys` documentation: stdout is "block-buffered" when not interactive, stderr "is line-buffered in both cases").
So, on a pipe, stdout lines arrive late and in the wrong order relative to stderr, and the program code cannot
be changed to fix this. On a terminal, stdout is line-buffered.

[`run_in_pty()`](terminal.py#L54-L117) therefore connects the program's stdout and stderr to one new pseudo-terminal:

1. The program sees a TTY on fd 1 and fd 2, also in CI, so it buffers and formats its output as in a direct run
   in a terminal. Both streams share one kernel buffer, so the caller reads them in write order.
2. Output processing (`OPOST`) is turned off. Otherwise the terminal changes every `\n` into `\r\n`, and CI
   logs would contain the `\r`. The bytes pass through unchanged, including `\r` progress bars and ANSI codes.
3. The window size is the size of pytest's console if that is a terminal. Otherwise, for example in CI, it is
   80 columns by 24 rows, the same fallback that `shutil.get_terminal_size()` uses.
4. stdin is `/dev/null`.
5. The program starts with no blocked signals, and with the default actions for SIGPIPE and SIGXFSZ (the Python
   start-up of the supervisor ignores them, and ignored signals are inherited through `exec`).
6. The caller reads the master side in a loop and passes each chunk to a callback immediately. The end of output
   is reported as `EIO` on Linux, after the last process that holds the terminal has closed it.

### 3.6 No leftovers: the subreaper supervisor

`run_in_pty()` does not start the program directly. It starts a small supervisor process (`terminal.py` run as a
script, [`_supervise()`](terminal.py#L193-L230)), and the supervisor starts the program:

1. **Subreaper.** The supervisor calls `prctl(PR_SET_CHILD_SUBREAPER)`. The kernel then reparents every orphaned
   descendant of the program to the supervisor instead of to init. This also applies to daemons that use
   `setsid()` and a double fork. So all descendants stay in the supervisor's process tree.
2. **Parent-death signal.** The supervisor calls `prctl(PR_SET_PDEATHSIG, SIGTERM)`. When its parent (pytest)
   dies, even by SIGKILL, the kernel sends SIGTERM to the supervisor. If the parent died before this call took
   effect, the supervisor sees a different parent PID and does not start the program.
3. **Process groups.** The supervisor moves to its own process group (`os.setpgid(0, 0)`,
   [terminal.py:205](terminal.py#L205-L205)) before it starts anything. It starts the program in the process group
   of pytest (`posix_spawnp(..., setpgroup=caller_group)`, [terminal.py:215-222](terminal.py#L215-L222)). So Ctrl-C,
   a terminal hang-up, and group signals from a CI runner reach the program as in a direct run, but not the
   supervisor.
4. **Synchronous signal handling.** The supervisor blocks SIGCHLD, SIGTERM, SIGINT, SIGQUIT, and SIGHUP and takes
   them with `sigwaitinfo()`. No signal handler can interrupt the cleanup. Only SIGTERM starts the cleanup.
   The other terminal signals are only taken from the queue, because the program gets them itself.
5. **Cleanup.** When the program ends, or SIGTERM arrives, [`_kill_descendants()`](terminal.py#L252-L276) scans
   `/proc` every 50 ms for live descendants. It sends SIGTERM to each new one, and SIGKILL to all remaining
   ones after the grace period. It reaps them and returns only when no descendant is left. Processes that fork
   during their shutdown are found in the next scan.
6. **Report.** The supervisor writes the exit status of the program (or the start error) to a status pipe and
   exits. `run_in_pty()` reads the report after the end of output and returns the status in the `subprocess`
   convention: the exit code, or `-N` for death by signal N. A missing program raises `FileNotFoundError`.
7. **Exceptions in the caller.** If the read loop raises (for example `KeyboardInterrupt`, or an exception from
   the output callback), `run_in_pty()` sends SIGTERM to the supervisor, discards further output, waits until the
   supervisor has cleaned up, and then re-raises. A second exception during that wait stops only the wait; the
   supervisor still finishes the cleanup by itself.
8. **Abnormal supervisor end.** If the supervisor disappears while some process still holds the terminal open,
   `run_in_pty()` raises `RuntimeError` instead of waiting for that process forever.

### 3.7 What happens when a CI runner sends SIGKILL to the whole process group

The kernel delivers a process-group signal (`kill(-PGID, SIGKILL)`) only to processes whose process group is
PGID. Step by step:

1. pytest, the program, and every descendant of the program that did not change its process group die at once,
   as in a direct run.
2. The supervisor is in its own process group (section 3.6, point 3), so it does not get the signal.
3. Because pytest died, the kernel sends SIGTERM to the supervisor (`PR_SET_PDEATHSIG`). The supervisor takes
   it with `sigwaitinfo()`. If the SIGCHLD of the dead program arrives first, the supervisor reaps the program
   instead. Both paths lead to the cleanup.
4. Descendants that left the group before (for example a daemon after `setsid()`) were reparented to the
   supervisor (subreaper), so the `/proc` scan finds them. They get SIGTERM, and SIGKILL after the grace period.
5. The supervisor writes its report and exits. Nobody reads the report anymore, so the write fails with
   `BrokenPipeError`, but only after the cleanup is complete. This step follows from the code; no test checks it.

What guarantees the cleanup:

1. Kernel mechanisms, not code that runs in the killed processes: the process-group semantics of `kill()`,
   `PR_SET_PDEATHSIG`, and subreaper reparenting.
2. The supervisor cannot be interrupted during the cleanup (synchronous signal handling), and it loops until
   the `/proc` scan finds no live descendant.
3. The timing windows are closed:
   1. Before `os.setpgid(0, 0)`, the supervisor is still in pytest's group. But it has not started the program
      yet, so a group SIGKILL in this window leaves nothing behind.
   2. If pytest dies before `PR_SET_PDEATHSIG` takes effect, the supervisor sees the changed parent PID and does
      not start the program.
4. Evidence: the test `test_sigkill_of_the_process_group_of_the_caller_kills_a_daemon` sends SIGKILL to the
   process group of a caller, and checks that a `setsid` daemon of the program dies. With the
   `os.setpgid(0, 0)` call removed, this test fails, and the daemon survives.

What is **not** guaranteed:

1. If the runner also kills the supervisor itself with SIGKILL (for example, when it kills each process of the
   job one by one), descendants that left pytest's process group survive, unless the runner kills them too. If
   the runner stops a container or kills a cgroup, the kernel kills all processes in it anyway.
2. A process stuck in uninterruptible sleep (state `D`, for example in a driver call) dies only when it leaves
   that state.

### 3.8 Why an undocumented pytest API

`CaptureManager.global_and_fixture_disabled()` is not part of pytest's documented API. It is used because no
documented API gives **live** output from a **module-scoped** fixture while capture is active. Each documented
option was tried or examined:

1. **`capsys.disabled()` / `capfd.disabled()`** (documented, and built on the same method). `capsys` and `capfd` are
   function-scoped fixtures, and the expensive group fixture is module-scoped. Tried: pytest fails with
   `ScopeMismatch: You tried to access the function scoped fixture capsys with a module scoped request object`.
2. **The terminal reporter** (`config.pluginmanager.getplugin("terminalreporter").write(...)`). It writes to the
   `sys.stdout` object from configuration time, and while a fixture runs, fd 1 points to pytest's capture file
   (pytest 9.0.3 `capture.py`: `os.dup2(self.tmpfile.fileno(), self.targetfd)`). Tried: a flushed write was lost,
   and an unflushed write appeared only after the fixture ended (2 seconds late in the experiment).
3. **`sys.__stdout__`**. It also writes to fd 1. Tried: the output was visible only with `-s`.
4. **Live logging (`log_cli`)** (documented). It needs pytest configuration (an ini file or command-line
   options), while the requirement is visibility whatever the configuration is, and without `pytest.ini`.
   It also formats records line by line, adds `live log setup` section headers, and forces verbose mode. So
   raw bytes and `\r` progress bars would not match a direct run. Its handler (`_LiveLoggingStreamHandler.emit`)
   uses `global_and_fixture_disabled()` internally anyway.
5. **Writing to `/dev/tty`**. There is no controlling terminal in CI, and output to `/dev/tty` ignores a
   redirection of pytest's stdout to a file or a CI log.
6. **Saving a duplicate of fd 1 before capture starts.** pytest starts capture before it imports `conftest.py`
   files and test modules, so only a plugin that loads very early (`-p`, `PYTEST_PLUGINS`, or an entry point)
   could do this. That is extra configuration. Not tried.
7. **`suspend_global_capture()` / `resume_global_capture()`**. Also internal, and at a lower level: it does not
   pause fixture capture. `global_and_fixture_disabled()` is the method that `capsys.disabled()` uses.
8. **Public API only, by restructuring.** Run the expensive part in a function-scoped fixture that can use
   `capfd.disabled()`, and cache the group result manually per module. This re-implements pytest's fixture
   caching and error caching, and it makes every group file more complex. Rejected.

How the risk is contained:

1. All uses are in one function, [`_capture_suspended()`](harness.py#L107-L112).
2. [harness_output_tests/](harness_output_tests/) detects a break. Without the pause, 4 of its 6 tests fail
   (default capture, `--capture=sys`, `--capture=tee-sys`, and the terminal run).
3. pytest's documented features `capsys.disabled()` and live logging depend on this method. That makes a
   silent removal less likely, but it is not a guarantee. This is an inference, not a pytest statement.

## 4. Alternatives considered and rejected

### 4.1 Test structure and pytest features

1. **A custom harness instead of pytest.** It would have to re-implement test selection, reporting, JUnit XML,
   `--lf`, and assertion messages. pytest covers the structure directly: a group is a higher-scoped fixture, and
   a case is a parametrized test.
2. **One test per group with subtests** (`subtests`, built into pytest 9.0). Subtests are created at run time, so
   the command line cannot select a single case.
3. **pytest-asyncio.** It runs async tests one after another on purpose, so it gives no concurrency.
   **pytest-asyncio-cooperative** needs isolated coroutine tests with no shared resources, and the group fixture
   is a shared resource.
4. **pytest-xdist and pipelining of groups in the background.** These would only help if processes or groups could
   run in parallel, and they cannot (section 2.1, point 3).
5. **Indirect parametrization of one fixture over all groups in a single file** (the first skeleton). Each group
   has its own file and its own fixture instead, so a plain module-scoped fixture and a plain `parametrize`
   are enough.

### 4.2 Ordered output

1. **Two pipes, stdout and stderr.** The relative order is lost, and stdout is still block-buffered.
2. **One pipe (`stderr=subprocess.STDOUT`).** stdout is still block-buffered by the program, so stdout lines arrive
   late and out of order with stderr.
3. **One pipe with `PYTHONUNBUFFERED=1`.** This fixes only Python-level output. Native code (C/C++ extensions,
   ROCm libraries, binaries) still block-buffers its stdio output on a pipe. Also, the program does not see a
   TTY, so its output differs from a direct run in a terminal.
4. **`stdbuf -oL`** (not tried). From memory: it works through `LD_PRELOAD` on C stdio only, so it has no effect
   on Python's own buffering or on statically linked programs.

### 4.3 Pseudo-terminal implementation

1. **`pty.spawn()`** (standard library). It switches the caller's terminal to raw mode (`setraw(STDIN_FILENO)`),
   copies stdin to the child, and writes the output to fd 1, which pytest captures.
2. **`pexpect`.** A third-party dependency that CI would need too. It still needs the forwarding code, and its
   output has `\r\n` line endings.
3. Chosen: own code on top of `os.openpty()`, `termios`, `subprocess`, and `select`. Together with the supervisor,
   `terminal.py` has about 220 lines of code, not counting docstrings, comments, and blank lines.

### 4.4 Cleanup of descendants

1. **Process group only (`killpg`).** It misses descendants that leave the group (`setsid`, daemons), and it does
   nothing when pytest itself is killed.
2. **cgroup v2 `cgroup.kill`, or a PID namespace.** These are the strongest options, but they need privileges. A
   cgroup needs delegated write access. Docker's default seccomp profile blocks `unshare` and `clone` with
   namespace flags without `CAP_SYS_ADMIN`. So they are not available in typical CI containers.
   (The default profile does allow `prctl` and `setpgid`, which the chosen solution uses.)
3. **Making pytest itself a subreaper.** This would change process handling for everything else in the test
   session (orphans of unrelated subprocesses would need reaping).
4. **A supervisor in a new session (`setsid`) instead of a new process group.** Then the program could not join
   pytest's process group, because `setpgid()` works only within one session.
5. **A supervisor in pytest's process group.** A group SIGKILL from a CI runner would kill it before the cleanup,
   and a terminal hang-up or Ctrl-\ would too.
6. **Signal handlers that raise an exception in the supervisor.** A second signal could interrupt the cleanup
   itself. Blocked signals with `sigwaitinfo()` cannot.
7. **The supervisor kills itself with the program's signal to pass on the exit status.** That can produce core
   dumps, and it mixes the status of the program with the status of the supervisor. A status pipe keeps them apart.
8. **A time limit for the cleanup that then kills the supervisor with SIGKILL.** That would leak exactly the
   processes that are slow to die. Instead, `run_in_pty()` waits, and a second Ctrl-C stops only the wait.

### 4.5 Tests and configuration

1. **`pytester` for the harness output tests.** It must be enabled with `pytest_plugins` in a top-level
   `conftest.py`, or with `-p pytester`. Its in-process runs also do not have a real stdout. The tests start
   separate pytest processes with `subprocess` instead: with a pipe (as in CI) and inside the pseudo-terminal
   (as in a local terminal).

## 5. Limitations

1. **Supervisor killed by SIGKILL.** Descendants that left pytest's process group survive (section 3.7).
2. **Processes in uninterruptible sleep.** There is no overall time limit. `run_in_pty()` returns only when every
   descendant is dead.
3. **Internal pytest API.** `global_and_fixture_disabled()` can change in a future pytest version (section 3.8).
   Only pytest 9.0.3 was tested.
4. **The program always sees a terminal.** Programs that check `isatty()` can write ANSI colors and `\r` progress
   updates into CI logs, and interactive programs can start a pager or ask a question. stdin is `/dev/null`.
5. **One combined stream.** stdout and stderr cannot be separated, for example to save stderr on its own.
6. **Output during an abort is discarded.** After Ctrl-C or an exception in the caller, the shutdown output of the
   processes (for example the program's own Ctrl-C traceback) is not shown, while a direct run would show it.
7. **Platform.** Linux only (`prctl`, `/proc`, `posix_spawn` with process-group control). Python 3.10 or later,
   because the group files use `zip(strict=True)`.
8. **Start cost.** The supervisor is one extra Python start per process run: about 26 ms for `true`, against
   0.6 ms for a plain `subprocess.run`. This was measured on a shared machine, so it is only an estimate.
9. **Shared result files.** The demo processes write `a_result.json` and `b_result.json` in this directory. The
   files stay after a run, and parallel runs in the same directory (for example with pytest-xdist) would conflict.
   pytest-xdist was not tested.
10. **Global switches.** `TEST_CASE` and `FAIL_PROCESS` apply to every group in the run, because the processes
    do not know their group. Run one file to limit them to one group.
11. **PID reuse.** Between a `/proc` scan and the signal, a PID could in theory be reused by an unrelated process.
    That needs the kernel to cycle through the whole PID range in a few milliseconds, so the risk is negligible.

## 6. What you must know to use it

### 6.1 Everywhere

1. Use pytest's default import mode. The group files import `harness` because pytest puts their directory on
   `sys.path`. Other import modes, such as `--import-mode=importlib`, were not tested.
2. Call `run_in_pty()` (and so `run_a()`, `run_b()`, and `run_process()`) from a thread that lives until the call
   returns, such as pytest's main thread. The kernel sends the parent-death signal when the parent *thread* ends.
3. The grace period between SIGTERM and SIGKILL is the `grace_period` parameter of `run_a()`, `run_b()`,
   `run_process()`, and `run_in_pty()`. The default is `terminal.DEFAULT_GRACE_PERIOD` (5 seconds). Programs that
   need longer to release resources (for example GPU memory) need a longer period.
4. Every process run ends with a kill of all its remaining descendants. A background service that a process
   starts and expects to keep running after the process exits will be killed too.
5. Case IDs in test IDs are `case1`, `case2`, and so on. `-k` matches substrings, so with 10 or more cases,
   `-k case1` also selects `case10`. Use node IDs, or `-k "case1 and not case10"`.

### 6.2 Locally

1. Ctrl-C reaches the program directly, as in a direct run, because it is in pytest's process group. pytest then
   stops the remaining processes (SIGTERM, then SIGKILL after the grace period). A second Ctrl-C stops the wait,
   and the supervisor finishes the cleanup by itself.
2. The program sees a terminal of the same size as your console.
3. On a failed test, pytest shows that test's captured prints in the failure report, even without `-s`. This is
   standard pytest behavior; `--show-capture=no` hides it.

### 6.3 In CI

1. No terminal is needed. The program still sees a terminal of 80 columns by 24 rows, so the output is the same
   as locally, including colors if the program prints them. The harness passes the environment unchanged, so
   variables such as `NO_COLOR`, `TERM`, or `PAGER=cat` work as they would for a direct run.
2. Stopping the job with SIGTERM or SIGKILL to pytest, or to its process group, triggers the cleanup (section 3.7).
   It needs up to the grace period plus a few scans.
3. In a container, the supervisor becomes a child of PID 1 after pytest dies. If PID 1 does not reap child
   processes, an exited supervisor stays as a zombie entry until the container ends. It holds no other
   resources. An init process that reaps, such as `docker run --init`, avoids this.

### 6.4 Adapting the demo to real processes

1. Replace `_run_mock()` in `harness.py` with code that builds the real command line from the selected case IDs and
   reads the real results. Keep the deletion of old result files before each run.
2. For any other command, call `harness.run_process(request, argv, cwd=..., grace_period=...)`. It returns the exit
   status, and the output goes live to the console.
3. `terminal.run_in_pty()` does not depend on pytest and can be used on its own, with any output callback.

## 7. Tests and verification

1. [terminal_tests/test_terminal.py](terminal_tests/test_terminal.py) (22 tests). They check:
   1. TTY on stdout and stderr, end of file on stdin, and the signal state of a direct run.
   2. Unchanged bytes, the order within stdout and within stderr, and live output from Python `print` and from C
      `printf` without a flush.
   3. Complete large output (2 MiB), the last bytes before exit, and output of grandchildren.
   4. Window size, process group, exit status (code and signal), and no file-descriptor leaks.
   5. Cleanup of a leftover grandchild, of a `setsid` daemon, and of a grandchild that ignores SIGTERM (which
      also checks that the grace period reaches the supervisor).
   6. Cleanup after an exception in the caller, after SIGKILL of the caller, and after SIGKILL of the caller's
      whole process group.
2. [harness_output_tests/test_harness_output.py](harness_output_tests/test_harness_output.py) (6 tests). Each runs
   an inner pytest session: with default capture, `-s`, `--capture=sys`, `--capture=tee-sys`, and
   `-p no:capture` on a pipe, and with default capture in a pseudo-terminal. Each checks that the process output
   (stdout and stderr) appears on pytest's stdout before the session summary, with the right terminal size. With
   default capture, the test print stays hidden; with `-s`, it appears.
3. Mutation checks confirmed that the tests detect real failures. Each of these changes made the named tests
   fail:
   1. Without the pause of pytest's capture: 4 of the harness output tests.
   2. Without the signal reset for the program: the signal-state test.
   3. Without `os.setpgid(0, 0)` in the supervisor: the process-group SIGKILL test.
4. Scenarios run on the demo:
   1. Default capture and `-s`.
   2. Selection with a node ID plus `-k`: the processes got only the selected cases.
   3. `TEST_CASE=2`: only `test_compare[case2]` failed, with `'mytest_failed_B2' == 'mytest_B2'`.
   4. `FAIL_PROCESS=A` and `FAIL_PROCESS=B`: all tests of each group reported ERROR, and each failing process
      ran once per group.

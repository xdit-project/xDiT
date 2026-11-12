"""Tests for terminal.run_in_pty(): live and exact output through the pseudo-terminal, and process cleanup."""

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

_PROTOTYPE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROTOTYPE_DIR))
import terminal  # noqa: E402

_GRACE = 1.0  # A short grace period keeps the cleanup tests fast.
_TIMEOUT = 10.0  # Upper limit for waits that must end much earlier.


def _run(code, on_output=None, **kwargs):
    """Run Python `code` in the pseudo-terminal; return (exit status, all output)."""
    chunks = []

    def collect(chunk):
        chunks.append(chunk)
        if on_output is not None:
            on_output(b"".join(chunks))

    argv = [sys.executable, "-c", textwrap.dedent(code)]
    status = terminal.run_in_pty(argv, collect, grace_period=kwargs.pop("grace_period", _GRACE), **kwargs)
    return status, b"".join(chunks)


def _alive(pid):
    """True if `pid` runs and is not a zombie."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as stat_file:
            stat = stat_file.read()
    except FileNotFoundError:
        return False
    return stat[stat.rindex(b")") + 2 :].split()[0] != b"Z"


def _wait_for_file(path, timeout=_TIMEOUT):
    deadline = time.monotonic() + timeout
    while not path.exists() or not path.read_text():
        assert time.monotonic() < deadline, f"{path} did not appear"
        time.sleep(0.01)


def _wait_until_dead(pids, timeout):
    deadline = time.monotonic() + timeout
    while any(_alive(pid) for pid in pids):
        assert time.monotonic() < deadline, f"still alive: {[pid for pid in pids if _alive(pid)]}"
        time.sleep(0.05)


def _open_fds():
    return sorted(os.listdir("/proc/self/fd"))


# The program starts a sleeping grandchild and writes "<program pid> <grandchild pid>" to the file.
_START_SLEEPER = """
import os, subprocess, sys
sleeper = subprocess.Popen([sys.executable, "-c", {sleeper_code!r}], stdout=subprocess.PIPE)
assert sleeper.stdout.readline() == b"armed\\n"
with open({pid_file!r}, "w") as pid_file:
    pid_file.write(f"{{os.getpid()}} {{sleeper.pid}}")
"""
_SLEEPER = "import time; print('armed', flush=True); time.sleep(60)"
_SLEEPER_IGNORING_SIGTERM = (
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('armed', flush=True); time.sleep(60)"
)


def _start_sleeper(pid_file, sleeper_code=_SLEEPER):
    return _START_SLEEPER.format(sleeper_code=sleeper_code, pid_file=str(pid_file))


# The program starts a daemon (setsid and a double fork) and writes "<program pid> <daemon pid>" to the file.
# The daemon leaves the process group and the session of the program, and its parent ends at once, so only the
# subreaper can still find it. It also holds the terminal open, so without the cleanup a run lasts 60 seconds.
_START_DAEMON = """
import os, time
read_end, write_end = os.pipe()
if os.fork() == 0:
    os.setsid()
    if os.fork() == 0:
        os.write(write_end, str(os.getpid()).encode())
        time.sleep(60)
    os._exit(0)
os.close(write_end)
daemon = os.read(read_end, 100).decode()
with open({pid_file!r}, "w") as pid_file:
    pid_file.write(f"{{os.getpid()}} {{daemon}}")
"""


def _start_daemon(pid_file):
    return _START_DAEMON.format(pid_file=str(pid_file))


def _start_caller(program_code, **popen_kwargs):
    """Start a separate process that calls run_in_pty() for Python `program_code`."""
    caller_code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(_PROTOTYPE_DIR)!r})
        import terminal
        terminal.run_in_pty([sys.executable, "-c", {program_code!r}], lambda chunk: None, grace_period={_GRACE})
    """)
    return subprocess.Popen([sys.executable, "-c", caller_code], **popen_kwargs)


def _read_pids(pid_file):
    return [int(pid) for pid in pid_file.read_text().split()]


# ---- Output ---------------------------------------------------------------------------------------------------


def test_stdout_and_stderr_are_a_terminal_and_stdin_is_at_eof():
    status, output = _run("import os, sys; print(os.isatty(0), os.isatty(1), os.isatty(2), repr(sys.stdin.read()))")
    assert (status, output) == (0, b"False True True ''\n")


def test_program_gets_the_signal_state_of_a_direct_run():
    # Blocked and ignored signals are inherited through exec. The supervisor blocks some signals, and the Python
    # start-up ignores SIGPIPE and SIGXFSZ, so a missing reset would change the behavior of the program.
    # The program is not Python, because the Python start-up changes its own signal state.
    argv = ["grep", "-E", "^Sig(Blk|Ign):", "/proc/self/status"]
    chunks = []
    status = terminal.run_in_pty(argv, chunks.append, grace_period=_GRACE)
    direct = subprocess.run(argv, capture_output=True, check=True).stdout
    assert (status, b"".join(chunks)) == (0, direct)


def test_bytes_pass_unchanged():
    data = b"a\nb\r\nc\rd\te\xc3\xa9\x1b[1mbold\x1b[0m"
    status, output = _run(f"import os; os.write(1, {data!r})")
    assert (status, output) == (0, data)


def test_order_within_stdout_and_within_stderr():
    status, output = _run("""
        import sys
        for i in range(300):
            print(f"out {i}")
            print(f"err {i}", file=sys.stderr)
    """)
    lines = output.decode().splitlines()
    assert status == 0
    assert [line for line in lines if line.startswith("out ")] == [f"out {i}" for i in range(300)]
    assert [line for line in lines if line.startswith("err ")] == [f"err {i}" for i in range(300)]


@pytest.mark.parametrize(
    "write_ready",
    [
        pytest.param("print('ready')", id="python-print"),
        pytest.param("import ctypes; ctypes.CDLL(None).printf(b'ready\\n')", id="c-printf"),
    ],
)
def test_output_arrives_while_the_program_runs(tmp_path, write_ready):
    """The program writes a line without a flush and then waits until the caller has seen it.

    If the line came only at exit (block buffering on a pipe), the program would end with status 3.
    """
    seen = tmp_path / "seen"

    def on_output(output):
        if b"ready\n" in output:
            seen.write_text("yes")

    status, output = _run(
        f"""
        import os, sys, time
        {write_ready}
        deadline = time.monotonic() + {_TIMEOUT}
        while not os.path.exists({str(seen)!r}):
            if time.monotonic() > deadline:
                sys.exit(3)
            time.sleep(0.01)
        """,
        on_output,
    )
    assert (status, output) == (0, b"ready\n")


def test_large_output_is_complete():
    data = bytes(range(256)) * 8192  # 2 MiB, more than any terminal buffer
    status, output = _run("import sys; sys.stdout.buffer.write(bytes(range(256)) * 8192)")
    assert status == 0
    assert output == data


def test_last_output_before_exit_is_kept():
    for _ in range(20):
        assert _run("import os; os.write(1, b'last words'); os._exit(0)") == (0, b"last words")


def test_grandchild_output_arrives():
    status, output = _run("""
        import subprocess, sys
        subprocess.run([sys.executable, "-c", "print('from grandchild')"], check=True)
    """)
    assert (status, output) == (0, b"from grandchild\n")


def test_window_size():
    assert _run("import os; print(*os.get_terminal_size())", size=(33, 111)) == (0, b"111 33\n")


def test_program_runs_in_the_process_group_of_the_caller():
    # As in a direct run, so that Ctrl-C and signals to the process group reach the program.
    assert _run("import os; print(os.getpgrp())") == (0, f"{os.getpgrp()}\n".encode())


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        pytest.param("pass", 0, id="success"),
        pytest.param("import sys; sys.exit(7)", 7, id="exit-code"),
        pytest.param("import os, signal; os.kill(os.getpid(), signal.SIGKILL)", -signal.SIGKILL, id="signal"),
    ],
)
def test_exit_status(code, expected):
    assert _run(code)[0] == expected


def test_no_fd_leak_after_a_run():
    before = _open_fds()
    _run("print('x')")
    assert _open_fds() == before


def test_missing_program_raises_and_leaks_no_fds():
    before = _open_fds()
    with pytest.raises(FileNotFoundError):
        terminal.run_in_pty(["/nonexistent/program"], lambda chunk: None, grace_period=_GRACE)
    assert _open_fds() == before


# ---- Cleanup of descendants -------------------------------------------------------------------------------------


def test_leftover_grandchild_is_killed_after_normal_exit(tmp_path):
    pid_file = tmp_path / "pids"
    status, _ = _run(_start_sleeper(pid_file))
    program, sleeper = _read_pids(pid_file)
    assert status == 0
    assert not _alive(program) and not _alive(sleeper)


def test_daemon_after_setsid_and_double_fork_is_killed(tmp_path):
    pid_file = tmp_path / "pids"
    status, _ = _run(_start_daemon(pid_file))
    program, daemon = _read_pids(pid_file)
    assert status == 0
    assert not _alive(program) and not _alive(daemon)


def test_grandchild_that_ignores_sigterm_is_killed_after_the_grace_period(tmp_path):
    pid_file = tmp_path / "pids"
    start = time.monotonic()
    status, _ = _run(_start_sleeper(pid_file, _SLEEPER_IGNORING_SIGTERM))
    elapsed = time.monotonic() - start
    _, sleeper = _read_pids(pid_file)
    assert status == 0
    assert not _alive(sleeper)
    # At least the grace period given here, and less than the default: so the parameter reached the supervisor.
    assert _GRACE <= elapsed < terminal.DEFAULT_GRACE_PERIOD


def test_exception_in_the_caller_kills_the_program_and_its_descendants(tmp_path):
    pid_file = tmp_path / "pids"

    class Boom(Exception):
        pass

    def on_output(output):
        raise Boom

    with pytest.raises(Boom):
        _run(_start_sleeper(pid_file) + "print('go', flush=True)\nimport time; time.sleep(60)", on_output)
    program, sleeper = _read_pids(pid_file)
    assert not _alive(program) and not _alive(sleeper)


def test_sigkill_of_the_caller_kills_the_program_and_its_descendants(tmp_path):
    pid_file = tmp_path / "pids"
    caller = _start_caller(_start_sleeper(pid_file) + "import time; time.sleep(60)")
    try:
        _wait_for_file(pid_file)
    finally:
        caller.kill()
        caller.wait()
    _wait_until_dead(_read_pids(pid_file), _GRACE + _TIMEOUT)


def test_sigkill_of_the_process_group_of_the_caller_kills_a_daemon(tmp_path):
    # CI runners often stop a job with SIGKILL to its process group. The program dies with the group, but the
    # daemon is outside of it: only a supervisor outside of the group survives to kill the daemon.
    pid_file = tmp_path / "pids"
    caller = _start_caller(_start_daemon(pid_file) + "import time; time.sleep(60)", start_new_session=True)
    try:
        _wait_for_file(pid_file)
    finally:
        os.killpg(caller.pid, signal.SIGKILL)
        caller.wait()
    _wait_until_dead(_read_pids(pid_file), _GRACE + _TIMEOUT)

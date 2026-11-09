"""Run a program under a pseudo-terminal, forward its output live, and leave no process of the run behind.

Why a pseudo-terminal: on a pipe, Python and C stdio block-buffer stdout, so output arrives late and out of
order with stderr. On a terminal they line-buffer it. The program gets one pseudo-terminal as stdout and
stderr, so the caller reads one byte stream in write order, and the program behaves as in a direct run in a
terminal, also in CI.

Why a supervisor process: the program runs as a child of a small supervisor (this file, run as a script).
The supervisor is a "child subreaper" (prctl(2) PR_SET_CHILD_SUBREAPER): the kernel reparents every orphaned
descendant of the program to it, also daemons after setsid() and a double fork, so all descendants stay in
its process tree. When the program ends, or the supervisor gets SIGTERM, the supervisor sends SIGTERM to all
descendants, waits for the grace period, sends SIGKILL to the remaining ones, and exits only when none is
left. The kernel sends SIGTERM to the supervisor when its parent dies, also by SIGKILL (PR_SET_PDEATHSIG).

Process groups: the program runs in the process group of the caller, so Ctrl-C in a terminal, a terminal
hang-up, and signals of a CI runner to the process group reach it as in a direct run. The supervisor runs in its
own process group, so these signals do not kill it before it can clean up.

Known gap: if the supervisor itself gets SIGKILL, the descendants survive.

Linux only.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import select
import signal
import struct
import subprocess
import sys
import termios
import time
from collections.abc import Callable, Sequence

DEFAULT_GRACE_PERIOD = 5.0
"""Seconds between SIGTERM and SIGKILL when processes of a run are stopped."""

DEFAULT_SIZE = (24, 80)
"""(rows, columns) of the pseudo-terminal. The same fallback as shutil.get_terminal_size() uses."""

_CHUNK_SIZE = 65536
_POLL_INTERVAL = 0.05
_THIS_FILE = os.path.abspath(__file__)

# Values from <linux/prctl.h>. The Python standard library has no prctl() wrapper.
_PR_SET_PDEATHSIG = 1
_PR_SET_CHILD_SUBREAPER = 36


def run_in_pty(
    argv: Sequence[str],
    on_output: Callable[[bytes], None],
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: dict[str, str] | None = None,
    size: tuple[int, int] = DEFAULT_SIZE,
    grace_period: float = DEFAULT_GRACE_PERIOD,
) -> int:
    """Run `argv` with stdout and stderr on a new pseudo-terminal, and give its output to `on_output`.

    Contract:
    - stdin of the program is /dev/null. The program starts with no blocked signals and default signal
      dispositions, except the ones that the caller ignores.
    - `on_output` gets each chunk of raw bytes as soon as it is read. The terminal does no output processing,
      so the bytes are exactly the bytes that the program wrote ("\\n" stays "\\n").
    - `size` is (rows, columns) of the pseudo-terminal.
    - Returns the exit status in the `subprocess` convention: the exit code, or -N if signal N killed it.
    - When the function returns or raises, the program and all its descendants are dead. Descendants that are
      still alive when the program ends get SIGTERM, and SIGKILL after `grace_period` seconds.
    - On an exception (for example KeyboardInterrupt, or an exception from `on_output`), the whole tree is
      stopped the same way before the exception propagates. Output during this shutdown is discarded. A second
      exception during the shutdown stops only the wait; the supervisor still finishes the cleanup.
    - Raises OSError (for example FileNotFoundError) if the program cannot start.
    - The calling thread must live until the function returns, because the kernel sends PR_SET_PDEATHSIG when
      the parent *thread* ends.
    """
    if not argv:
        raise ValueError("argv must not be empty")
    master, slave = os.openpty()
    status_read, status_write = os.pipe()
    try:
        try:
            _configure_terminal(slave, size)
            supervisor = subprocess.Popen(
                [sys.executable, "-I", _THIS_FILE, str(grace_period), str(os.getpid()), str(status_write), *argv],
                stdin=subprocess.DEVNULL,
                stdout=slave,
                stderr=slave,
                cwd=cwd,
                env=env,
                pass_fds=(status_write,),
            )
        finally:
            # The supervisor has its own copies. If this process keeps its copies, the master never reports
            # the end of output, and the status pipe never reports EOF.
            os.close(slave)
            os.close(status_write)
        try:
            ended_cleanly = _forward_output(master, supervisor, on_output)
        except BaseException:
            _stop_supervisor(master, supervisor)
            raise
        supervisor.wait()
        report = _read_all(status_read)
    finally:
        os.close(master)
        os.close(status_read)
    if not ended_cleanly:
        raise RuntimeError(
            f"the supervisor of {argv[0]!r} ended (exit status {supervisor.returncode}) while processes "
            "still held the terminal open; these processes can still run"
        )
    return _parse_report(report, supervisor.returncode, argv[0])


def _configure_terminal(fd: int, size: tuple[int, int]) -> None:
    attributes = termios.tcgetattr(fd)
    # No output processing: by default the terminal changes "\n" to "\r\n", and CI logs would get the "\r".
    attributes[1] &= ~termios.OPOST
    termios.tcsetattr(fd, termios.TCSANOW, attributes)
    rows, columns = size
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))


def _forward_output(master: int, supervisor: subprocess.Popen, on_output: Callable[[bytes], None]) -> bool:
    """Give output to `on_output` until no process holds the terminal open.

    Returns False if the supervisor ended while some process still held the terminal open. That happens only
    if the supervisor died before it finished the cleanup (for example by SIGKILL).
    """
    supervisor_ended = False
    while True:
        # A finite timeout lets the loop see a supervisor that died without the cleanup. Without it, a surviving
        # descendant that holds the terminal open would block the read for its whole life.
        if select.select([master], [], [], 0 if supervisor_ended else _POLL_INTERVAL)[0]:
            chunk = _read_chunk(master)
            if not chunk:
                return True
            on_output(chunk)
        elif supervisor_ended:
            return False
        else:
            supervisor_ended = supervisor.poll() is not None


def _read_chunk(master: int) -> bytes:
    """Read from the master; return b"" when all slave copies are closed (Linux reports this as EIO)."""
    try:
        return os.read(master, _CHUNK_SIZE)
    except OSError as error:
        if error.errno != errno.EIO:
            raise
        return b""


def _stop_supervisor(master: int, supervisor: subprocess.Popen) -> None:
    """Make the supervisor stop the program tree, and wait until it has exited."""
    if supervisor.poll() is None:
        supervisor.send_signal(signal.SIGTERM)
    while supervisor.poll() is None:
        # Read on: a process that writes during its shutdown must not block on a full terminal buffer.
        if select.select([master], [], [], _POLL_INTERVAL)[0] and not _read_chunk(master):
            supervisor.wait()


def _read_all(fd: int) -> bytes:
    chunks = []
    while chunk := os.read(fd, 4096):
        chunks.append(chunk)
    return b"".join(chunks)


def _parse_report(report: bytes, supervisor_status: int, program: str) -> int:
    kind, _, value = report.decode().partition(" ")
    if kind == "exit":
        return int(value)
    if kind == "spawn-error":
        code = int(value)
        raise OSError(code, os.strerror(code), program)
    raise RuntimeError(
        f"the supervisor of {program!r} ended without the exit status of the program "
        f"(supervisor exit status {supervisor_status}, report {report!r})"
    )


# ---- Supervisor side: the code below runs in the supervisor process. ----------------------------------------


def _supervise(grace_period: float, parent_pid: int, status_fd: int, argv: list[str]) -> None:
    """Run `argv`, wait until it ends or SIGTERM arrives, kill all descendants, write a report to `status_fd`.

    Report format: "exit <status>", "spawn-error <errno>", "terminated", or "failed".
    """
    os.set_inheritable(status_fd, False)  # Only the supervisor may hold the write end of the status pipe.
    # The signals are blocked and taken with sigwaitinfo(), so no handler can interrupt the cleanup. Only SIGTERM
    # starts the cleanup. The other terminal signals are watched only to take them from the queue: the
    # supervisor must stay to clean up, and the program gets these signals itself through the process group.
    watched = {signal.SIGCHLD, signal.SIGTERM, signal.SIGINT, signal.SIGQUIT, signal.SIGHUP}
    signal.pthread_sigmask(signal.SIG_BLOCK, watched)
    caller_group = os.getpgrp()
    os.setpgid(0, 0)  # See "Process groups" in the module docstring.
    report = "failed"
    try:
        _prctl(_PR_SET_CHILD_SUBREAPER, 1)
        _prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
        if os.getppid() != parent_pid:
            return  # The parent died before PR_SET_PDEATHSIG took effect, so the signal will never come.
        try:
            # setsigmask and setsigdef give the program the signal state of a direct run: no blocked signals,
            # and default SIGPIPE and SIGXFSZ, which the Python start-up sets to "ignore".
            pid = os.posix_spawnp(
                argv[0],
                argv,
                os.environ,
                setpgroup=caller_group,
                setsigmask=(),
                setsigdef=(signal.SIGPIPE, signal.SIGXFSZ),
            )
        except OSError as error:
            report = f"spawn-error {error.errno}"
            return
        status = _wait_for_program(pid, watched)
        report = "terminated" if status is None else f"exit {status}"
    finally:
        _kill_descendants(grace_period)
        os.write(status_fd, report.encode())


def _wait_for_program(pid: int, watched: set[signal.Signals]) -> int | None:
    """Return the exit status of `pid` in the `subprocess` convention, or None if SIGTERM came first."""
    while True:
        signum = signal.sigwaitinfo(watched).si_signo
        if signum == signal.SIGTERM:
            return None
        if signum == signal.SIGCHLD:
            # One SIGCHLD can stand for several ended children: the program, and orphans reparented to us.
            while True:
                try:
                    ended, status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if ended == 0:
                    break
                if ended == pid:
                    return os.waitstatus_to_exitcode(status)


def _kill_descendants(grace_period: float) -> None:
    """Stop all descendants: SIGTERM, then SIGKILL after `grace_period`. Return when none is left.

    The loop scans again on each pass, because processes can fork during their shutdown. A process that forks
    after the scan is caught on the next pass: when its parent dies, it is reparented to this subreaper.
    Process IDs from a scan can be reused before the signal only after the kernel uses the whole PID range,
    so the risk to signal an unrelated process is negligible.
    """
    root = os.getpid()
    deadline = time.monotonic() + grace_period
    sent_sigterm: set[int] = set()
    while True:
        _reap_children()
        alive = _live_descendants(root)
        if not alive:
            break
        kill_now = time.monotonic() >= deadline
        for pid in alive:
            if kill_now:
                _send(pid, signal.SIGKILL)
            elif pid not in sent_sigterm:
                _send(pid, signal.SIGTERM)
                sent_sigterm.add(pid)
        time.sleep(_POLL_INTERVAL)
    _reap_children()


def _reap_children() -> None:
    while True:
        try:
            ended, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if ended == 0:
            return


def _live_descendants(root: int) -> list[int]:
    """Return the PIDs of all descendants of `root` that are not zombies, from a scan of /proc."""
    children: dict[int, list[int]] = {}
    zombies: set[int] = set()
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            with open(f"/proc/{entry.name}/stat", "rb") as stat_file:
                stat = stat_file.read()
        except OSError:
            continue  # The process ended during the scan.
        # The command name in parentheses can contain spaces and ")", so parse after the last ")".
        state, ppid = stat[stat.rindex(b")") + 2 :].split()[:2]
        pid = int(entry.name)
        children.setdefault(int(ppid), []).append(pid)
        if state == b"Z":
            zombies.add(pid)
    descendants = []
    stack = [root]
    while stack:
        for pid in children.get(stack.pop(), ()):
            descendants.append(pid)
            stack.append(pid)
    return [pid for pid in descendants if pid not in zombies]


def _send(pid: int, signum: signal.Signals) -> None:
    try:
        os.kill(pid, signum)
    except ProcessLookupError:
        pass  # The process ended after the scan.


def _prctl(option: int, value: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(option, ctypes.c_ulong(value), ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0)) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


if __name__ == "__main__":
    # Internal command line, written only by run_in_pty(): <grace period> <parent pid> <status fd> <argv...>
    _supervise(float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:])

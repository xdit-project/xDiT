"""Code shared by the group test files: case selection, and running a process with live output on the console.

The output of a process goes straight to the console in every pytest capture mode (default capture, -s,
--capture=sys, --capture=tee-sys, -p no:capture), so the user can follow the expensive part of a group.
The output of the test functions keeps the normal capture rules of pytest.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import terminal

if TYPE_CHECKING:
    import pytest

_HERE = Path(__file__).resolve().parent


def selected_cases(request: pytest.FixtureRequest, param: str = "case") -> list:
    """Return the values of parameter `param` of the selected tests in the module of the fixture `request`.

    The list follows the final selection of pytest (after -k, -m, node IDs, --deselect, --lf), so a group runs
    its processes only for the cases that pytest will test. The order is the test order, without duplicates.
    The function knows nothing about groups: the values come from the parametrization in the module.
    """
    cases: dict = {}
    for item in request.session.items:
        callspec = getattr(item, "callspec", None)
        if getattr(item, "module", None) is request.module and callspec is not None and param in callspec.params:
            cases[callspec.params[param]] = None
    return list(cases)


def console_writer(config: pytest.Config) -> Callable[[bytes], None]:
    """Return a function that writes bytes to the console at once, past the output capture of pytest."""
    capture_manager = config.pluginmanager.getplugin("capturemanager")

    def write(data: bytes) -> None:
        with _capture_suspended(capture_manager):
            sys.stdout.flush()  # Text that pytest wrote before must come first.
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()

    return write


def run_process(
    request: pytest.FixtureRequest,
    argv: Sequence[str],
    *,
    cwd: str | os.PathLike[str] | None = None,
    grace_period: float = terminal.DEFAULT_GRACE_PERIOD,
) -> int:
    """Run `argv` and copy its output live to the console; return its exit status.

    The process sees a terminal of the same size as the console (or terminal.DEFAULT_SIZE if the console is
    not a terminal, for example in CI). See terminal.run_in_pty() for the full contract, including the cleanup
    of all descendants and the meaning of `grace_period`.
    """
    return terminal.run_in_pty(
        argv,
        console_writer(request.config),
        cwd=cwd,
        size=_console_size(request.config),
        grace_period=grace_period,
    )


def run_a(
    request: pytest.FixtureRequest, cases: Sequence, grace_period: float = terminal.DEFAULT_GRACE_PERIOD
) -> list[str]:
    """Run process_a.py for `cases`; return its results in the order of `cases`."""
    return _run_mock(request, "process_a.py", "a_result.json", cases, grace_period)


def run_b(
    request: pytest.FixtureRequest, cases: Sequence, grace_period: float = terminal.DEFAULT_GRACE_PERIOD
) -> list[str]:
    """Run process_b.py for `cases`; return its results in the order of `cases`."""
    return _run_mock(request, "process_b.py", "b_result.json", cases, grace_period)


def _run_mock(
    request: pytest.FixtureRequest, script: str, result_name: str, cases: Sequence, grace_period: float
) -> list[str]:
    case_args = [str(case) for case in cases]
    result_path = _HERE / result_name
    # All groups use the same result file. A process that ends without writing it must not hand over the
    # data of the previous group.
    result_path.unlink(missing_ok=True)
    console_writer(request.config)(f"\n[{request.module.__name__}] {script} {' '.join(case_args)}\n".encode())
    argv = [sys.executable, str(_HERE / script), *case_args]
    status = run_process(request, argv, cwd=_HERE, grace_period=grace_period)
    if status != 0:
        raise subprocess.CalledProcessError(status, argv)
    return [f"mytest_{result}" for result in json.loads(result_path.read_text())]


def _capture_suspended(capture_manager) -> contextlib.AbstractContextManager:
    # global_and_fixture_disabled() is not documented API, but capsys.disabled() and the live logging of
    # pytest use it. With -p no:capture there is no capture manager and nothing to suspend.
    if capture_manager is None:
        return contextlib.nullcontext()
    return capture_manager.global_and_fixture_disabled()


def _console_size(config: pytest.Config) -> tuple[int, int]:
    # With capture active, fd 1 is a temporary file, so the console can be queried only while capture is off.
    with _capture_suspended(config.pluginmanager.getplugin("capturemanager")):
        if sys.stdout.isatty():
            size = os.get_terminal_size(sys.stdout.fileno())
            if size.lines and size.columns:
                return size.lines, size.columns
    return terminal.DEFAULT_SIZE

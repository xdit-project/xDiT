"""Keep a spawned rank's interpreter start-up out of its hang budget (#817).

A spawned rank first starts an interpreter and imports torch and xfuser, which can take
minutes on a slow filesystem. Each rank imports what its worker needs and reports ready
before the caller's hang budget starts, so that budget covers only the code under test.
Set ``XFUSER_TEST_SPAWN_STARTUP_TIMEOUT`` (seconds) to change the start-up allowance.
"""

import importlib
import os
import queue
import time

STARTUP_TIMEOUT_ENV = "XFUSER_TEST_SPAWN_STARTUP_TIMEOUT"
DEFAULT_STARTUP_TIMEOUT = 900.0


def startup_timeout():
    return float(os.environ.get(STARTUP_TIMEOUT_ENV, DEFAULT_STARTUP_TIMEOUT))


def import_and_signal_ready(ready_queue, modules):
    """Import ``modules`` in this rank, then report ready even if an import failed.

    The worker repeats a failed import and reports the error itself.
    """
    try:
        for module in modules:
            importlib.import_module(module)
    except Exception:  # noqa: BLE001
        pass
    finally:
        ready_queue.put(None)


def import_then_run(ready_queue, modules, worker, *args):
    """Process target: import ``modules``, report ready, then run ``worker(*args)``."""
    import_and_signal_ready(ready_queue, modules)
    worker(*args)


def wait_until_started(processes, ready_queue):
    """Wait until every process has reported ready, one has exited, or start-up times out."""
    deadline = time.monotonic() + startup_timeout()
    started = 0
    while started < len(processes) and time.monotonic() < deadline:
        try:
            ready_queue.get(timeout=1)
            started += 1
        except queue.Empty:
            if not all(process.is_alive() for process in processes):
                return

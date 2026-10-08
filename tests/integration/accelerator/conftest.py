from pathlib import Path

import pytest

from _spawn import spawn_accelerator_ranks

_ACCELERATOR_ROOT = Path(__file__).resolve().parent


@pytest.fixture
def accelerator_ranks(tmp_path):
    """Spawn NCCL/RCCL ranks for one multi-GPU test."""

    def launch(worker, *, world_size, timeout=600, init_filename="dist-init", args=()):
        spawn_accelerator_ranks(
            worker,
            tmp_path,
            world_size=world_size,
            timeout=timeout,
            init_filename=init_filename,
            args=args,
        )

    return launch


def pytest_collection_modifyitems(items):
    """Mark device tests in this directory only.

    pytest invokes this hook with the whole session's items. Limiting the marker
    to this tree keeps ``not accelerator`` meaningful when unit tests are collected
    alongside it.
    """
    accelerator = pytest.mark.accelerator
    for item in items:
        path = Path(str(item.path)).resolve()
        if path == _ACCELERATOR_ROOT or _ACCELERATOR_ROOT in path.parents:
            item.add_marker(accelerator)

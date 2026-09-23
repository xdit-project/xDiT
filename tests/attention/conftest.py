"""Shared single-rank environment for the attention tests.

Several backends read sequence-parallel degrees from global state, so they need
an initialised process group even at world_size 1.

Scoped per module and torn down after, matching tests/layers/usp_test.py. A
session-scoped fixture would be cheaper but leaks: it holds the process group
open past this directory, and usp_test's setUp then calls
initialize_model_parallel unguarded and fails. Leaving global state behind for
the next test file is not worth one NCCL init.
"""

import os

import pytest
import torch

from xfuser.core.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
    initialize_runtime_state,
)
from xfuser.core.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    model_parallel_is_initialized,
)


def ensure_attention_environment() -> None:
    """Idempotent: safe from a fixture, from a test, or from __main__."""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12363")
    os.environ.setdefault("LOCAL_RANK", "0")

    if not torch.distributed.is_initialized():
        init_distributed_environment(rank=0, world_size=1)
    initialize_runtime_state()
    if not model_parallel_is_initialized():
        initialize_model_parallel(ring_degree=1, ulysses_degree=1)


def teardown_attention_environment() -> None:
    destroy_model_parallel()
    destroy_distributed_environment()


@pytest.fixture(scope="module", autouse=True)
def attention_environment():
    if not torch.cuda.is_available():
        yield
        return
    ensure_attention_environment()
    yield
    teardown_attention_environment()

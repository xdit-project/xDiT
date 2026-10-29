from unittest.mock import patch

import pytest
import torch

from xfuser.config.args import xFuserArgs
from xfuser.core.distributed import parallel_state, runtime_state


@pytest.fixture
def single_rank(tmp_path, monkeypatch):
    """A real single-rank Gloo group with xDiT's model-parallel groups and a runtime state on SDPA.

    Yields the engine config, for tests that initialize the runtime state again for their pipeline.
    """
    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this process")
    # Restore whatever runtime state the session had once this test is done.
    monkeypatch.setattr(runtime_state, "_RUNTIME", None)
    with patch.object(parallel_state, "set_device"):
        parallel_state.init_distributed_environment(
            backend="gloo",
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{tmp_path / 'dist-init'}",
        )
    try:
        parallel_state.initialize_model_parallel(backend="gloo")
        args = xFuserArgs(attention_backend="SDPA")
        engine_config, _ = args.create_config()
        runtime_state.initialize_runtime_state(engine_config=engine_config)
        yield engine_config
    finally:
        parallel_state.destroy_model_parallel()
        parallel_state.destroy_distributed_environment()

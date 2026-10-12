"""Accelerator checks for xfuser.core.distributed.sharding.shard_component.

One rank cannot show that parameters were split. These tests cover the behavior
that is visible on one device: dtype conversion, a missing wrap attribute, and
preserving the parameter count.
"""

import gc
import queue
import traceback

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from xfuser.core.distributed.sharding import shard_component


def _make_dit():
    class DiTBlock(nn.Module):
        def __init__(self, dim=256):
            super().__init__()
            self.attn = nn.Linear(dim, dim)
            self.mlp = nn.Linear(dim, dim)

        def forward(self, x):
            return self.mlp(self.attn(x))

    class DiT(nn.Module):
        def __init__(self, num_blocks=2):
            super().__init__()
            self.blocks = nn.ModuleList([DiTBlock(dim=256) for _ in range(num_blocks)])
            self.proj_out = nn.Linear(256, 256)

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return self.proj_out(x)

    return DiT(num_blocks=2)


def _check_dtype():
    sharded_model = shard_component(
        _make_dit(),
        wrap_attrs=["blocks"],
        device_id=0,
        dtype=torch.bfloat16,
    )
    assert isinstance(sharded_model, FSDP)
    for param in sharded_model.parameters():
        if param.dtype.is_floating_point:
            assert param.dtype == torch.bfloat16, f"Param dtype should be bfloat16, got {param.dtype}"


def _check_invalid_attr():
    try:
        shard_component(
            _make_dit(),
            wrap_attrs=["nonexistent_blocks"],
            device_id=0,
        )
    except AttributeError as error:
        if "nonexistent_blocks" not in str(error):
            raise
    else:
        raise AssertionError("expected AttributeError for nonexistent_blocks")


def _check_parameter_count():
    model = _make_dit()
    original_param_count = sum(p.numel() for p in model.parameters())
    sharded_model = shard_component(
        model,
        wrap_attrs=["blocks"],
        device_id=0,
    )
    sharded_param_count = sum(p.numel() for p in sharded_model.parameters())
    assert original_param_count == sharded_param_count, (
        f"Parameter count changed: {original_param_count} -> {sharded_param_count}"
    )


def _guard(init_method, check, result_queue):
    torch.cuda.set_device(0)
    try:
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=0,
            world_size=1,
        )
        check()
    except Exception:
        result_queue.put(traceback.format_exc())
        raise
    finally:
        gc.collect()
        try:
            torch.cuda.synchronize()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            torch.cuda.empty_cache()
    result_queue.put(None)


def _run_isolated(check, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator device")
    if not dist.is_nccl_available():
        pytest.skip("NCCL/RCCL is unavailable")

    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_guard,
        args=(f"file://{tmp_path / 'nccl-init'}", check, result_queue),
    )
    process.start()
    # Include interpreter startup and cold imports on slow filesystems (#817).
    process.join(600)
    if process.is_alive():
        process.terminate()
        process.join(5)
    if process.is_alive():
        process.kill()
        process.join(5)

    errors = []
    while True:
        try:
            item = result_queue.get(timeout=1)
        except queue.Empty:
            break
        if item is not None:
            errors.append(item)

    if process.exitcode != 0 or errors:
        details = [f"exit code: {process.exitcode}", *errors]
        pytest.fail("\n".join(details))


def test_shard_component_with_dtype(tmp_path):
    """FSDP wrapping converts floating parameters to the requested dtype."""
    _run_isolated(_check_dtype, tmp_path)


def test_shard_component_invalid_attr(tmp_path):
    """A missing wrap attribute fails at the getattr that looks it up."""
    _run_isolated(_check_invalid_attr, tmp_path)


def test_parameter_count_preserved(tmp_path):
    """Wrapping does not drop or duplicate parameters."""
    _run_isolated(_check_parameter_count, tmp_path)

"""Distributed tests for broadcasts inside parallel subgroups."""

import traceback
from functools import partial
from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.gloo, pytest.mark.slow]

_WORLD_SIZE = 4
_DP_DEGREE = 2
_SUBGROUPS = ([0, 1], [2, 3])
_OPERATIONS = ("object", "tensor_dict")
# With data parallelism 2, each mode below yields the subgroups in _SUBGROUPS.
# The pipeline coordinator does not call GroupCoordinator.__init__, so it is
# covered separately.
_PARALLEL_MODES = {
    "sequence": ("get_sp_group", {"sequence_parallel_degree": 2, "ulysses_degree": 2}),
    "pipeline": ("get_pp_group", {"pipeline_parallel_degree": 2}),
}


def _payload(operation, sender):
    """Build what global rank ``sender`` broadcasts for ``operation``."""
    import torch

    metadata = {"sender": sender, "labels": ["broadcast", None, True], "empty": {}}
    if operation == "object":
        return metadata
    return {
        "metadata": metadata,
        "tensors": {
            "values": torch.tensor([[sender, sender + 0.5]], dtype=torch.float32),
            "ids": torch.tensor([sender], dtype=torch.int64),
            "empty": torch.empty((0, 2), dtype=torch.float64),
        },
    }


def _to_plain(value):
    """Convert tensors to ``(dtype, shape, values)`` so results compare exactly across processes."""
    import torch

    if isinstance(value, torch.Tensor):
        return str(value.dtype), tuple(value.shape), value.tolist()
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value


def _broadcast_worker(rank, world_size, init_method, result_queue, *, parallel_mode):
    dist = None
    try:
        import torch.distributed as dist

        from xfuser.core.distributed import parallel_state

        with patch.object(parallel_state, "set_device"):
            parallel_state.init_distributed_environment(
                backend="gloo",
                distributed_init_method=init_method,
                local_rank=rank,
                rank=rank,
                world_size=world_size,
            )
        get_group, degrees = _PARALLEL_MODES[parallel_mode]
        parallel_state.initialize_model_parallel(backend="gloo", data_parallel_degree=_DP_DEGREE, **degrees)

        group = getattr(parallel_state, get_group)()
        received = {}
        for operation in _OPERATIONS:
            broadcast = group.broadcast_object if operation == "object" else group.broadcast_tensor_dict
            # `src` is group-local: in group [2, 3], src=1 is global rank 3.
            for src, sender in enumerate(group.ranks):
                value = _payload(operation, sender) if rank == sender else None
                received[operation, src] = _to_plain(broadcast(value, src=src))

        result_queue.put(("returned", rank, group.ranks, received))
    except Exception:  # noqa: BLE001 - report arbitrary child failures to the parent
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("parallel_mode", sorted(_PARALLEL_MODES))
def test_subgroup_broadcasts_from_every_group_local_src(tmp_path, run_spawned, parallel_mode):
    torch = pytest.importorskip("torch", reason="PyTorch is required for distributed process-group tests")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")

    processes, hung, survivors, results = run_spawned(
        torch,
        partial(_broadcast_worker, parallel_mode=parallel_mode),
        f"file://{tmp_path / f'{parallel_mode}-broadcast-gloo-init'}",
        world_size=_WORLD_SIZE,
        # The shared deadline includes interpreter startup and imports for all
        # four ranks; allow slow CI hosts while still bounding hangs.
        timeout=600,
    )

    assert not survivors, f"workers survived SIGKILL: {survivors}"
    assert not hung, f"broadcast hung worker pids: {hung}"
    errors = [result[2] for result in results if result[0] == "error"]
    assert not errors, "\n".join(errors)
    assert [process.exitcode for process in processes] == [0] * _WORLD_SIZE

    expected = {}
    for ranks in _SUBGROUPS:
        received = {
            (operation, src): _to_plain(_payload(operation, sender))
            for operation in _OPERATIONS
            for src, sender in enumerate(ranks)
        }
        expected.update({rank: (ranks, received) for rank in ranks})
    assert {rank: (ranks, received) for _, rank, ranks, received in results} == expected

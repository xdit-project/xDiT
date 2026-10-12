"""Pipeline-parallel stages renegotiate transfer shapes for every request."""

import queue
import time
import traceback
from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.gloo, pytest.mark.slow]

_WORLD_SIZE = 2
_HIDDEN = 8
# Two requests at the same resolution whose text sequence lengths differ, as
# with Flux called with max_sequence_length=512 and then 256.
_SEQUENCE_LENGTHS = (512, 256, 512)


def _handshake_worker(rank, world_size, init_method, result_queue):
    dist = None
    try:
        import torch
        import torch.distributed as dist

        import xfuser.envs as envs
        from xfuser.config.config import InputConfig, RuntimeConfig
        from xfuser.core.distributed import group_coordinator, parallel_state
        from xfuser.core.distributed.runtime_state import DiTRuntimeState

        cpu = torch.device("cpu")
        # Keep every rank on CPU even when accelerators are visible.
        with (
            patch.object(parallel_state, "set_device"),
            patch.object(envs, "get_device", lambda local_rank: cpu),
            patch.object(group_coordinator, "synchronize", lambda: None),
        ):
            parallel_state.init_distributed_environment(
                backend="gloo",
                distributed_init_method=init_method,
                local_rank=rank,
                rank=rank,
                world_size=world_size,
            )
            parallel_state.initialize_model_parallel(backend="gloo", pipeline_parallel_degree=world_size)

            state = DiTRuntimeState.__new__(DiTRuntimeState)
            state.runtime_config = RuntimeConfig(dtype=torch.float32)
            state.input_config = InputConfig(height=256, width=256, batch_size=1)
            state.ready = True
            pp_group = parallel_state.get_pp_group()

            received = []
            for request_idx, sequence_length in enumerate(_SEQUENCE_LENGTHS):
                state.set_input_parameters(
                    height=256,
                    width=256,
                    batch_size=1,
                    num_inference_steps=2,
                    max_condition_sequence_length=sequence_length,
                    split_text_embed_in_sp=False,
                )
                if parallel_state.is_pipeline_first_stage():
                    encoder_hidden_state = torch.full((1, sequence_length, _HIDDEN), float(request_idx))
                    pp_group.pipeline_send(encoder_hidden_state, name="encoder_hidden_state")
                else:
                    encoder_hidden_state = pp_group.pipeline_recv(0, "encoder_hidden_state")
                    received.append(
                        (
                            tuple(encoder_hidden_state.shape),
                            bool(torch.all(encoder_hidden_state == float(request_idx))),
                        )
                    )

        result_queue.put(("returned", rank, received))
    except Exception:  # noqa: BLE001 - report arbitrary child failures to the parent
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def _run_spawned(torch, worker, init_method, *, world_size, timeout):
    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(target=worker, args=(rank, world_size, init_method, result_queue)) for rank in range(world_size)
    ]
    for process in processes:
        process.start()

    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))

    hung = [process.pid for process in processes if process.is_alive()]
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(5)

    results = []
    while len(results) < world_size:
        try:
            results.append(result_queue.get(timeout=1))
        except queue.Empty:
            break
    return processes, hung, results


def test_requests_with_different_text_lengths_renegotiate_pipeline_shapes(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed with the gloo backend is unavailable")

    processes, hung, results = _run_spawned(
        torch,
        _handshake_worker,
        f"file://{tmp_path / 'pp-handshake-init'}",
        world_size=_WORLD_SIZE,
        timeout=120,
    )

    assert not hung, f"pipeline send/recv hung; results so far: {results}"
    errors = [result for result in results if result[0] == "error"]
    assert not errors, "\n".join(result[2] for result in errors)
    assert [process.exitcode for process in processes] == [0] * _WORLD_SIZE
    received = {rank: payload for _, rank, payload in results}
    assert received == {
        0: [],
        1: [((1, sequence_length, _HIDDEN), True) for sequence_length in _SEQUENCE_LENGTHS],
    }

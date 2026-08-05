"""Spawned distributed regressions for roadmap task 2."""

import queue
import time
import traceback
from types import SimpleNamespace

import pytest


def _layout_mismatch_worker(rank, world_size, init_method, result_queue):
    dist = None
    try:
        import torch
        import torch.distributed as dist

        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        from xfuser.model_executor.models.runner_models.loading.meta_load import (
            _collective_assert_same_layout,
        )

        class Group:
            def __init__(self):
                self.rank_in_group = rank
                self.world_size = world_size

            @staticmethod
            def broadcast_object_list(box, src=0):
                dist.broadcast_object_list(box, src=src)

            @staticmethod
            def all_reduce(tensor):
                dist.all_reduce(tensor)
                return tensor

        layouts = (
            (("parameter", "weight"), ("buffer", "scale")),
            (("buffer", "scale"), ("parameter", "weight")),
        )
        try:
            _collective_assert_same_layout(layouts[rank], Group(), device="cpu")
        except RuntimeError as error:
            result_queue.put(("raised", rank, str(error)))
        else:
            result_queue.put(("returned", rank, "layout mismatch was accepted"))
    except BaseException:
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def _source_error_worker(rank, world_size, init_method, result_queue):
    dist = None
    try:
        import torch.distributed as dist

        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        from xfuser.model_executor.models.runner_models.loading.meta_load import (
            _collective_source_call,
        )

        class Group:
            def __init__(self):
                self.rank_in_group = rank
                self.world_size = world_size

            @staticmethod
            def broadcast_object_list(box, src=0):
                dist.broadcast_object_list(box, src=src)

        def source_operation():
            if rank != 0:
                raise AssertionError("peer executed rank0-only operation")
            raise OSError("safetensors read failed")

        try:
            _collective_source_call(
                Group(),
                is_src=rank == 0,
                operation=source_operation,
                context="loading blocks.0.weight",
            )
        except RuntimeError as error:
            result_queue.put(("raised", rank, str(error)))
        else:
            result_queue.put(("returned", rank, "source error was accepted"))
    except BaseException:
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def _build_error_worker(rank, world_size, init_method, result_queue):
    dist = None
    try:
        import torch.distributed as dist

        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        from xfuser.model_executor.models.runner_models.loading.meta_load import (
            _collective_build_call,
        )

        class Group:
            def __init__(self):
                self.rank_in_group = rank
                self.world_size = world_size

            @staticmethod
            def broadcast_object_list(box, src=0):
                dist.broadcast_object_list(box, src=src)

        def build():
            if rank == 1:
                raise ValueError("rank-local config failure")
            return "built"

        try:
            _collective_build_call(Group(), build, context="meta transformer")
        except RuntimeError as error:
            result_queue.put(("raised", rank, str(error)))
        else:
            result_queue.put(("returned", rank, "build mismatch was accepted"))
    except BaseException:
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def _quantize_error_worker(rank, world_size, init_method, result_queue):
    dist = None
    try:
        import torch.distributed as dist

        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        from xfuser.core.distributed.sharding import _collective_quantize_call

        def quantize():
            if rank == 1:
                raise ValueError("rank-local quantization failed")

        try:
            _collective_quantize_call(
                quantize,
                process_group=dist.group.WORLD,
                context="quantizing block 0",
            )
        except RuntimeError as error:
            result_queue.put(("raised", rank, str(error)))
        else:
            result_queue.put(("returned", rank, "quantization failure was accepted"))
    except BaseException:
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def _te_source_error_worker(rank, world_size, init_method, result_queue):
    dist = None
    try:
        import torch
        import torch.distributed as dist

        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        from xfuser.model_executor.models.runner_models.loading import meta_load

        class Group:
            def __init__(self):
                self.rank_in_group = rank
                self.world_size = world_size

            @staticmethod
            def broadcast_object_list(box, src=0):
                dist.broadcast_object_list(box, src=src)

        class Source:
            def state_dict(self):
                raise OSError("text encoder state dict failed")

        meta_load.get_fs_group = lambda: Group()
        loader = object.__new__(meta_load.MemoryEfficientLoader)
        loader.model = SimpleNamespace(
            settings=SimpleNamespace(
                fsdp_strategy={"text_encoder": {"wrap_attrs": []}}
            )
        )
        loader._load_rank0_source = lambda *args: Source()
        loader._release_rank0_source = lambda *args: None
        try:
            loader._broadcast_load_component(
                torch.nn.Linear(1, 1), "text_encoder", offload=False
            )
        except RuntimeError as error:
            result_queue.put(("raised", rank, str(error)))
        else:
            result_queue.put(("returned", rank, "source error was accepted"))
    except BaseException:
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def _mxfp4_fsdp2_worker(rank, world_size, init_method, result_queue):
    dist = None
    try:
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        from torch.distributed._composable.fsdp import fully_shard
        from torch.distributed.device_mesh import DeviceMesh
        from xfuser.model_executor.layers import mxfp4_linear

        quant_type = SimpleNamespace(per_1x32=object())

        def get_hip_quant(_):
            def quantize(weight, shuffle=True):
                return (
                    torch.zeros(
                        (weight.shape[0], weight.shape[1] // 2), dtype=torch.uint8
                    ),
                    torch.ones((weight.shape[0], 1), dtype=torch.float32),
                )

            return quantize

        mxfp4_linear.aiter = SimpleNamespace(
            QuantType=quant_type, get_hip_quant=get_hip_quant
        )
        mxfp4_linear.shuffle_weight = lambda weight, layout: weight

        full_precision_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in mxfp4_linear.xFuserMXFP4Linear(
                8, 4, bias=False
            ).state_dict().items()
        }
        layer = mxfp4_linear.xFuserMXFP4Linear(8, 4, bias=False)
        layer._quantize_weights()
        packed_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in layer.state_dict().items()
        }
        full_numel = layer.weight_shuffle.numel()
        layer.to(f"cuda:{rank}")
        mesh = DeviceMesh.from_group(dist.group.WORLD, "cuda")
        fully_shard(layer, mesh=mesh)

        packed = layer.weight_shuffle
        local = packed.to_local()
        try:
            layer.load_state_dict(packed_state)
        except RuntimeError as error:
            load_error = str(error)
        else:
            load_error = None
        try:
            layer.load_state_dict(full_precision_state)
        except RuntimeError as error:
            full_precision_load_error = str(error)
        else:
            full_precision_load_error = None
        result_queue.put(
            (
                "sharded",
                rank,
                type(packed).__name__,
                local.numel(),
                full_numel,
                load_error,
                full_precision_load_error,
            )
        )
    except BaseException:
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def _run_spawned(torch, worker, init_method, *, timeout):
    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(target=worker, args=(rank, 2, init_method, result_queue))
        for rank in range(2)
    ]
    for process in processes:
        process.start()

    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))

    hung = [process.pid for process in processes if process.is_alive()]
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(5)

    results = []
    while len(results) < 2:
        try:
            results.append(result_queue.get(timeout=1))
        except queue.Empty:
            break
    return processes, hung, results


def test_two_rank_layout_mismatch_raises_on_all_ranks_without_hanging(tmp_path):
    torch = pytest.importorskip("torch", reason="PyTorch is required for gloo test")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")

    processes, hung, results = _run_spawned(
        torch,
        _layout_mismatch_worker,
        f"file://{tmp_path / 'gloo-init'}",
        timeout=20,
    )

    assert not hung, f"collective mismatch hung worker pids: {hung}"
    assert [process.exitcode for process in processes] == [0, 0]
    assert sorted(result[:2] for result in results) == [
        ("raised", 0),
        ("raised", 1),
    ], results
    assert all("ordered parameter/buffer layout mismatch" in result[2] for result in results)


def test_two_rank_source_error_raises_on_all_ranks_without_hanging(tmp_path):
    torch = pytest.importorskip("torch", reason="PyTorch is required for gloo test")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")

    processes, hung, results = _run_spawned(
        torch,
        _source_error_worker,
        f"file://{tmp_path / 'source-error-init'}",
        timeout=20,
    )

    assert not hung, f"source error hung worker pids: {hung}"
    assert [process.exitcode for process in processes] == [0, 0]
    assert sorted(result[:2] for result in results) == [
        ("raised", 0),
        ("raised", 1),
    ], results
    assert all("OSError: safetensors read failed" in result[2] for result in results)


def test_two_rank_meta_build_error_raises_on_all_ranks_without_hanging(tmp_path):
    torch = pytest.importorskip("torch", reason="PyTorch is required for gloo test")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")

    processes, hung, results = _run_spawned(
        torch,
        _build_error_worker,
        f"file://{tmp_path / 'build-error-init'}",
        timeout=20,
    )

    assert not hung, f"meta build error hung worker pids: {hung}"
    assert [process.exitcode for process in processes] == [0, 0]
    assert sorted(result[:2] for result in results) == [
        ("raised", 0),
        ("raised", 1),
    ], results
    assert all("rank 1: ValueError" in result[2] for result in results)


def test_two_rank_quantize_error_raises_on_all_ranks_without_hanging(tmp_path):
    torch = pytest.importorskip("torch", reason="PyTorch is required for gloo test")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")

    processes, hung, results = _run_spawned(
        torch,
        _quantize_error_worker,
        f"file://{tmp_path / 'quantize-error-init'}",
        timeout=20,
    )

    assert not hung, f"quantize error hung worker pids: {hung}"
    assert [process.exitcode for process in processes] == [0, 0]
    assert sorted(result[:2] for result in results) == [
        ("raised", 0),
        ("raised", 1),
    ], results
    assert all("rank 1: ValueError" in result[2] for result in results)


def test_two_rank_text_encoder_source_error_exits_before_scatter(tmp_path):
    torch = pytest.importorskip("torch", reason="PyTorch is required for gloo test")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")

    processes, hung, results = _run_spawned(
        torch,
        _te_source_error_worker,
        f"file://{tmp_path / 'te-source-error-init'}",
        timeout=20,
    )

    assert not hung, f"text-encoder source error hung worker pids: {hung}"
    assert [process.exitcode for process in processes] == [0, 0]
    assert sorted(result[:2] for result in results) == [
        ("raised", 0),
        ("raised", 1),
    ], results
    assert all("OSError: text encoder state dict failed" in result[2] for result in results)


def test_mxfp4_packed_weight_is_sharded_by_fsdp2(tmp_path):
    import inspect

    torch = pytest.importorskip("torch", reason="PyTorch is required for FSDP2 test")
    if torch.cuda.device_count() < 2:
        pytest.skip("FSDP2 shard-size assertion requires two CUDA devices")
    if not torch.distributed.is_available() or not torch.distributed.is_nccl_available():
        pytest.skip("torch.distributed NCCL backend is unavailable")
    try:
        from torch.distributed._composable.fsdp import fully_shard
        from torch.distributed.device_mesh import DeviceMesh
    except ImportError:
        pytest.skip("this PyTorch build does not provide composable FSDP2")
    if "mesh" not in inspect.signature(fully_shard).parameters or not hasattr(
        DeviceMesh, "from_group"
    ):
        pytest.skip("this PyTorch build lacks the FSDP2 device-mesh API used by xDiT")

    processes, hung, results = _run_spawned(
        torch,
        _mxfp4_fsdp2_worker,
        f"file://{tmp_path / 'nccl-init'}",
        timeout=60,
    )

    assert not hung, f"FSDP2 sharding hung worker pids: {hung}"
    assert [process.exitcode for process in processes] == [0, 0]
    assert len(results) == 2
    assert all(result[0] == "sharded" for result in results), results
    assert all(result[2] == "DTensor" for result in results)
    assert all(result[3] < result[4] for result in results)
    assert all(
        "packed state cannot be loaded after FSDP" in result[5]
        for result in results
    )
    assert all(
        "full-precision state cannot replace an FSDP-managed packed parameter"
        in result[6]
        for result in results
    )

"""Spawned two-rank regressions: a load that fails must fail on every rank, not hang."""

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
        loader = object.__new__(meta_load.ModelLoader)
        loader.model = SimpleNamespace(
            settings=SimpleNamespace(fsdp_strategy={"text_encoder": {"wrap_attrs": []}})
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


def _run_replicated_te_validation_worker(
    rank, world_size, init_method, result_queue, *, scenario
):
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

            @staticmethod
            def all_reduce(tensor):
                dist.all_reduce(tensor)
                return tensor

            @staticmethod
            def broadcast(tensor, src=0):
                dist.broadcast(tensor, src=src)

        component = torch.nn.Module()
        if scenario == "persistence_mismatch":
            component.register_buffer(
                "cache",
                torch.ones(2),
                persistent=rank == 0,
            )
        elif scenario == "spec_reconcile" and rank == 0:
            component.register_parameter(
                "weight",
                torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)),
            )
        elif scenario == "spec_reconcile":
            component.register_parameter(
                "weight",
                torch.nn.Parameter(torch.empty(2, dtype=torch.float16, device="meta")),
            )
        else:
            component.left = torch.nn.Module()
            component.right = torch.nn.Module()
            if rank == 0:
                left = torch.nn.Parameter(
                    torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
                )
                right = (
                    left
                    if scenario == "source_tied"
                    else torch.nn.Parameter(
                        torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)
                    )
                )
            else:
                left = torch.nn.Parameter(
                    torch.empty(2, dtype=torch.float16, device="meta")
                )
                right = (
                    left
                    if scenario == "peer_tied"
                    else torch.nn.Parameter(
                        torch.empty(2, dtype=torch.float16, device="meta")
                    )
                )
            component.left.register_parameter("weight", left)
            component.right.register_parameter("weight", right)

        def set_tensor(module, name, device, *, value, dtype=None):
            parent_name, _, local_name = name.rpartition(".")
            owner = module.get_submodule(parent_name) if parent_name else module
            value = value.to(device=device, dtype=dtype or value.dtype)
            if local_name in owner._parameters:
                owner._parameters[local_name] = torch.nn.Parameter(
                    value,
                    requires_grad=False,
                )
            else:
                owner._buffers[local_name] = value

        loader = object.__new__(meta_load.ModelLoader)
        try:
            loader._fill_te_replicated(
                component,
                "cpu",
                Group(),
                set_tensor,
            )
        except RuntimeError as error:
            result_queue.put(("raised", rank, str(error)))
        else:
            if scenario == "spec_reconcile":
                result_queue.put(
                    (
                        "returned",
                        rank,
                        tuple(component.weight.shape),
                        component.weight.dtype == torch.float32,
                        component.weight.tolist(),
                    )
                )
            elif scenario in {"source_tied", "peer_tied"}:
                result_queue.put(
                    (
                        "returned",
                        rank,
                        component.left.weight is component.right.weight,
                        component.left.weight.tolist(),
                        component.right.weight.tolist(),
                    )
                )
            else:
                result_queue.put(
                    ("returned", rank, "persistence mismatch was accepted")
                )
    except BaseException:
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def _replicated_te_reconcile_worker(rank, world_size, init_method, result_queue):
    _run_replicated_te_validation_worker(
        rank,
        world_size,
        init_method,
        result_queue,
        scenario="spec_reconcile",
    )


def _replicated_te_persistence_worker(rank, world_size, init_method, result_queue):
    _run_replicated_te_validation_worker(
        rank,
        world_size,
        init_method,
        result_queue,
        scenario="persistence_mismatch",
    )


def _replicated_te_source_tied_worker(rank, world_size, init_method, result_queue):
    _run_replicated_te_validation_worker(
        rank,
        world_size,
        init_method,
        result_queue,
        scenario="source_tied",
    )


def _replicated_te_peer_tied_worker(rank, world_size, init_method, result_queue):
    _run_replicated_te_validation_worker(
        rank,
        world_size,
        init_method,
        result_queue,
        scenario="peer_tied",
    )


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
            for name, tensor in mxfp4_linear.xFuserMXFP4Linear(8, 4, bias=False)
            .state_dict()
            .items()
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


def _pinned_fp32_component(torch):
    """A two-block stand-in for a diffusers transformer that pins a norm to fp32."""

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(32, 32, bias=False)
            self.norm2 = torch.nn.LayerNorm(32)

        def forward(self, hidden):
            return self.norm2(self.proj(hidden).float()).to(hidden.dtype)

    class Component(torch.nn.Module):
        _keep_in_fp32_modules = ["norm2"]

        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block(), Block()])

        def forward(self, hidden):
            for block in self.blocks:
                hidden = block(hidden)
            return hidden

    return Component()


def _pinned_fp32_sharding_worker(rank, world_size, init_method, result_queue):
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
        from xfuser.core.distributed.sharding import shard_component

        facts = {}
        for label, kwargs in (
            ("fsdp2", {"memory_efficient_init": True}),
            ("fsdp1", {}),
        ):
            component = _pinned_fp32_component(torch)
            sharded = shard_component(
                component,
                ["blocks"],
                dist.group.WORLD,
                rank,
                torch.bfloat16,
                sync_module_states=False,
                **kwargs,
            )
            hidden = torch.randn(2, 32, device=f"cuda:{rank}", dtype=torch.bfloat16)
            output = sharded(hidden)
            # FSDP1 prefixes wrapped names, so match on the leaf rather than the full path.
            named = dict(sharded.named_parameters())
            pinned = next(v for k, v in named.items() if k.endswith("norm2.weight"))
            projected = next(v for k, v in named.items() if k.endswith("proj.weight"))
            facts[label] = {
                "forward_dtype": str(output.dtype),
                "pinned_dtype": str(pinned.dtype),
                "pinned_class": type(pinned).__name__,
                "pinned_numel": pinned.numel(),
                "proj_dtype": str(projected.dtype),
                "proj_class": type(projected).__name__,
            }
        result_queue.put(("sharded", rank, facts))
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


def _require_gloo():
    torch = pytest.importorskip("torch", reason="PyTorch is required for gloo test")
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_gloo_available()
    ):
        pytest.skip("torch.distributed gloo backend is unavailable")
    return torch


@pytest.mark.parametrize(
    ("worker", "init_name", "expected"),
    [
        pytest.param(
            _layout_mismatch_worker,
            "gloo-init",
            "ordered parameter/buffer layout mismatch",
            id="transformer_layout_mismatch",
        ),
        pytest.param(
            _source_error_worker,
            "source-error-init",
            "OSError: safetensors read failed",
            id="transformer_source_error",
        ),
        pytest.param(
            _build_error_worker,
            "build-error-init",
            "rank 1: ValueError",
            id="meta_build_error",
        ),
        pytest.param(
            _quantize_error_worker,
            "quantize-error-init",
            "rank 1: ValueError",
            id="quantize_error",
        ),
        pytest.param(
            _te_source_error_worker,
            "te-source-error-init",
            "OSError: text encoder state dict failed",
            id="text_encoder_source_error",
        ),
        pytest.param(
            _replicated_te_persistence_worker,
            "te-persistence-init",
            "ordered parameter/buffer layout mismatch",
            id="replicated_text_encoder_layout_mismatch",
        ),
    ],
)
def test_two_rank_load_failure_raises_on_every_rank_without_hanging(
    tmp_path, worker, init_name, expected
):
    """Every rank has to leave a failed load by raising. A rank that returns early, or that reports
    the error and then waits, parks its peer in a collective that never completes."""

    torch = _require_gloo()

    processes, hung, results = _run_spawned(
        torch,
        worker,
        f"file://{tmp_path / init_name}",
        timeout=20,
    )

    assert not hung, f"hung worker pids: {hung}"
    assert [process.exitcode for process in processes] == [0, 0]
    assert sorted(result[:2] for result in results) == [
        ("raised", 0),
        ("raised", 1),
    ], results
    assert all(expected in result[2] for result in results)


def test_two_rank_replicated_te_reconciles_specs_without_hanging(tmp_path):
    torch = _require_gloo()

    processes, hung, results = _run_spawned(
        torch,
        _replicated_te_reconcile_worker,
        f"file://{tmp_path / 'te-reconcile-init'}",
        timeout=20,
    )

    assert not hung, f"text-encoder reconciliation hung worker pids: {hung}"
    assert [process.exitcode for process in processes] == [0, 0]
    assert len(results) == 2
    assert all(result[0] == "returned" for result in results), results
    assert all(result[2:] == ((3,), True, [1.0, 2.0, 3.0]) for result in results)


@pytest.mark.parametrize(
    ("worker", "expected_tied", "expected_right", "init_name"),
    [
        (
            _replicated_te_source_tied_worker,
            True,
            [1.0, 2.0, 3.0],
            "te-source-tied-init",
        ),
        (
            _replicated_te_peer_tied_worker,
            False,
            [4.0, 5.0, 6.0],
            "te-peer-tied-init",
        ),
    ],
)
def test_two_rank_replicated_te_matches_source_aliases_without_hanging(
    tmp_path, worker, expected_tied, expected_right, init_name
):
    torch = _require_gloo()

    processes, hung, results = _run_spawned(
        torch,
        worker,
        f"file://{tmp_path / init_name}",
        timeout=20,
    )

    assert not hung, f"text-encoder alias reconciliation hung worker pids: {hung}"
    assert [process.exitcode for process in processes] == [0, 0]
    assert len(results) == 2
    assert all(result[0] == "returned" for result in results), results
    assert all(result[2] is expected_tied for result in results)
    assert all(result[3] == [1.0, 2.0, 3.0] for result in results)
    assert all(result[4] == expected_right for result in results)


def test_sharding_keeps_the_models_fp32_modules_out_of_the_shards(tmp_path):
    """A unit is one flat allocation, so FSDP rejects a mixture of dtypes inside it. The pinned norm
    therefore has to stay a plain replicated fp32 parameter while the block's weights shard, or
    xFuser has to demote it and load a lower-precision model than an ordinary load."""

    torch = pytest.importorskip("torch", reason="PyTorch is required for FSDP test")
    if torch.cuda.device_count() < 2:
        pytest.skip("sharding assertions require two CUDA devices")
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_nccl_available()
    ):
        pytest.skip("torch.distributed NCCL backend is unavailable")

    processes, hung, results = _run_spawned(
        torch,
        _pinned_fp32_sharding_worker,
        f"file://{tmp_path / 'nccl-init'}",
        timeout=180,
    )

    assert not hung, f"sharding hung worker pids: {hung}"
    assert [process.exitcode for process in processes] == [0, 0]
    assert len(results) == 2
    assert all(result[0] == "sharded" for result in results), results

    for _, rank, facts in results:
        for label, fact in facts.items():
            assert fact["forward_dtype"] == "torch.bfloat16", (label, rank, fact)
            assert fact["pinned_dtype"] == "torch.float32", (label, rank, fact)
            assert fact["pinned_class"] == "Parameter", (label, rank, fact)
            # Replicated, so every rank holds all 32 elements rather than a shard of them.
            assert fact["pinned_numel"] == 32, (label, rank, fact)
            assert fact["proj_dtype"] == "torch.bfloat16", (label, rank, fact)
        assert facts["fsdp2"]["proj_class"] == "DTensor", facts


def test_mxfp4_packed_weight_is_sharded_by_fsdp2(tmp_path):
    import inspect

    torch = pytest.importorskip("torch", reason="PyTorch is required for FSDP2 test")
    if torch.cuda.device_count() < 2:
        pytest.skip("FSDP2 shard-size assertion requires two CUDA devices")
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_nccl_available()
    ):
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
    from xfuser.model_executor.models.runner_models.loading.format_backends import (
        _probe_fsdp_non_float_parameters,
    )

    non_float_ok, non_float_reason = _probe_fsdp_non_float_parameters()
    if not non_float_ok:
        # The same probe gates the load contract, so a run on this PyTorch is
        # rejected in preflight rather than reaching the sharding path below.
        pytest.skip(non_float_reason)

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
        "packed state cannot be loaded after FSDP" in result[5] for result in results
    )
    assert all(
        "full-precision state cannot replace an FSDP-managed packed parameter"
        in result[6]
        for result in results
    )

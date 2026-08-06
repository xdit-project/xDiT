"""CPU-only regression tests for collective/meta-load safety helpers."""

from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def runtime():
    torch = pytest.importorskip(
        "torch", reason="PyTorch is required for meta-load tests"
    )
    from xfuser.core.distributed import sharding
    from xfuser.model_executor.models.runner_models.loading import meta_load
    from xfuser.model_executor.models.runner_models.loading import shard

    return SimpleNamespace(torch=torch, meta=meta_load, shard=shard, sharding=sharding)


class FakeGroup:
    def __init__(self, reference_layout, *, rank=1, global_mismatches=1):
        self.rank_in_group = rank
        self.world_size = 2
        self.reference_layout = reference_layout
        self.global_mismatches = global_mismatches
        self.calls = []

    def broadcast_object_list(self, box, src=0):
        self.calls.append("broadcast_object_list")
        box[0] = self.reference_layout

    def all_reduce(self, tensor):
        self.calls.append("all_reduce")
        tensor.fill_(self.global_mismatches)
        return tensor


def test_tensor_layout_preserves_order_kind_and_duplicate_names(runtime):
    torch = runtime.torch
    module = torch.nn.Module()
    shared = torch.nn.Parameter(torch.ones(1))
    module.left = torch.nn.Module()
    module.right = torch.nn.Module()
    module.left.register_parameter("weight", shared)
    module.right.register_parameter("weight", shared)
    module.register_buffer("cache", torch.ones(1))

    assert runtime.meta._tensor_layout(module) == (
        ("parameter", "left.weight"),
        ("parameter", "right.weight"),
        ("buffer", "cache"),
    )


def test_tensor_layout_contract_captures_specs_ties_and_persistence(runtime):
    torch = runtime.torch
    module = torch.nn.Module()
    shared = torch.nn.Parameter(torch.ones(2, dtype=torch.float16))
    module.left = torch.nn.Module()
    module.right = torch.nn.Module()
    module.left.register_parameter("weight", shared)
    module.right.register_parameter("weight", shared)
    module.register_buffer("saved", torch.ones(3, dtype=torch.float32), persistent=True)
    module.register_buffer(
        "cache", torch.ones(4, dtype=torch.float64), persistent=False
    )

    assert runtime.meta._tensor_layout_contract(module) == (
        (
            "parameter",
            "left.weight",
            (2,),
            torch.float16,
            "left.weight",
            None,
        ),
        (
            "parameter",
            "right.weight",
            (2,),
            torch.float16,
            "left.weight",
            None,
        ),
        (
            "buffer",
            "saved",
            (3,),
            torch.float32,
            "saved",
            True,
        ),
        (
            "buffer",
            "cache",
            (4,),
            torch.float64,
            "cache",
            False,
        ),
    )


@pytest.mark.parametrize(
    "field,local",
    [
        ("shape", (("parameter", "weight", (3,), "bf16", "weight", None),)),
        ("dtype", (("parameter", "weight", (2,), "fp32", "weight", None),)),
        ("tie", (("parameter", "weight", (2,), "bf16", "other", None),)),
        (
            "persistence",
            (("buffer", "cache", (2,), "bf16", "cache", False),),
        ),
    ],
)
def test_layout_contract_rejects_tensor_metadata_before_data_collective(
    runtime, field, local
):
    reference = (("parameter", "weight", (2,), "bf16", "weight", None),)
    if field == "persistence":
        reference = (("buffer", "cache", (2,), "bf16", "cache", True),)
    group = FakeGroup(reference)

    with pytest.raises(RuntimeError, match="layout mismatch"):
        runtime.meta._collective_assert_same_layout(local, group, device="cpu")

    assert group.calls == ["broadcast_object_list", "all_reduce"]


def test_layout_validation_rejects_same_count_in_different_order_collectively(runtime):
    reference = (("parameter", "a"), ("buffer", "b"))
    local = (("buffer", "b"), ("parameter", "a"))
    group = FakeGroup(reference)

    with pytest.raises(RuntimeError, match="ordered parameter/buffer layout mismatch"):
        runtime.meta._collective_assert_same_layout(local, group, device="cpu")

    assert group.calls == ["broadcast_object_list", "all_reduce"]


def test_layout_validation_makes_source_raise_when_only_a_peer_mismatches(runtime):
    reference = (("parameter", "a"),)
    group = FakeGroup(reference, rank=0, global_mismatches=1)

    with pytest.raises(RuntimeError, match="mismatch on 1 of 2 ranks"):
        runtime.meta._collective_assert_same_layout(reference, group, device="cpu")

    assert group.calls == ["broadcast_object_list", "all_reduce"]


def test_disk_broadcast_reconciles_peer_shape_and_dtype_before_data(runtime):
    torch = runtime.torch
    module = torch.nn.Module()
    module.register_parameter(
        "weight", torch.nn.Parameter(torch.empty(2, dtype=torch.float32))
    )

    class Group:
        rank_in_group = 1
        world_size = 2
        calls = 0

        @classmethod
        def broadcast_object_list(cls, box, src=0):
            cls.calls += 1
            if cls.calls == 1:
                box[0] = [("weight", (3,), torch.float16)]

    runtime.meta._collective_reconcile_tensor_specs(
        module, ["weight"], Group(), device="cpu"
    )

    assert tuple(module.weight.shape) == (3,)
    assert module.weight.dtype == torch.float16


def test_replicated_te_reconciles_specs_before_exact_layout_validation(runtime):
    torch = runtime.torch
    component = torch.nn.Module()
    component.register_parameter(
        "weight", torch.nn.Parameter(torch.empty(2, dtype=torch.float16))
    )
    source_contract = (
        (
            "parameter",
            "weight",
            (3,),
            torch.float32,
            "weight",
            None,
        ),
    )

    class Group:
        rank_in_group = 1
        world_size = 2

        def __init__(self):
            self.calls = []

        def broadcast_object_list(self, box, src=0):
            if not self.calls:
                self.calls.append("spec")
                box[0] = source_contract
            elif len(self.calls) < 3:
                self.calls.append(f"reconcile-{src}")
                box[0] = None
            elif len(self.calls) < 5:
                self.calls.append(f"materialize-{src}")
                box[0] = None
            else:
                self.calls.append("layout")
                box[0] = source_contract

        def all_reduce(self, tensor):
            self.calls.append("validate")
            tensor.zero_()
            return tensor

        def broadcast(self, tensor, src=0):
            self.calls.append("data")
            tensor.fill_(7)

    def set_tensor(module, name, device, *, value, dtype=None):
        module._parameters[name] = torch.nn.Parameter(
            value.to(dtype=dtype or value.dtype),
            requires_grad=False,
        )

    group = Group()
    loader = object.__new__(runtime.meta.MemoryEfficientLoader)
    loader._fill_te_replicated(
        component,
        "cpu",
        group,
        set_tensor,
    )

    assert tuple(component.weight.shape) == (3,)
    assert component.weight.dtype == torch.float32
    assert component.weight.tolist() == [7, 7, 7]
    assert group.calls == [
        "spec",
        "reconcile-0",
        "reconcile-1",
        "materialize-0",
        "materialize-1",
        "layout",
        "validate",
        "validate",
        "data",
    ]


def test_replicated_te_rebuilds_peer_aliases_from_tied_source_contract(runtime):
    torch = runtime.torch
    component = torch.nn.Module()
    component.left = torch.nn.Module()
    component.right = torch.nn.Module()
    component.left.register_parameter(
        "weight",
        torch.nn.Parameter(torch.empty(2, dtype=torch.float16, device="meta")),
    )
    component.right.register_parameter(
        "weight",
        torch.nn.Parameter(torch.empty(2, dtype=torch.float16, device="meta")),
    )
    source_contract = (
        (
            "parameter",
            "left.weight",
            (3,),
            torch.float32,
            "left.weight",
            None,
        ),
        (
            "parameter",
            "right.weight",
            (3,),
            torch.float32,
            "left.weight",
            None,
        ),
    )

    class Group:
        rank_in_group = 1
        world_size = 2

        def __init__(self):
            self.object_calls = 0
            self.data_broadcasts = 0

        def broadcast_object_list(self, box, src=0):
            self.object_calls += 1
            if self.object_calls == 1:
                box[0] = source_contract
            elif self.object_calls in (2, 3, 4, 5):
                box[0] = None
            else:
                box[0] = source_contract

        @staticmethod
        def all_reduce(tensor):
            return tensor

        def broadcast(self, tensor, src=0):
            self.data_broadcasts += 1
            tensor.fill_(11)

    def set_tensor(module, name, device, *, value, dtype=None):
        parent_name, _, local_name = name.rpartition(".")
        owner = module.get_submodule(parent_name)
        owner._parameters[local_name] = torch.nn.Parameter(
            value.to(dtype=dtype or value.dtype),
            requires_grad=False,
        )

    group = Group()
    loader = object.__new__(runtime.meta.MemoryEfficientLoader)
    loader._fill_te_replicated(component, "cpu", group, set_tensor)

    assert component.left.weight is component.right.weight
    assert tuple(component.left.weight.shape) == (3,)
    assert component.left.weight.dtype == torch.float32
    assert component.left.weight.tolist() == [11, 11, 11]
    assert group.data_broadcasts == 2


@pytest.mark.parametrize("rank", [0, 1])
def test_source_failure_status_makes_every_rank_raise(runtime, rank):
    calls = []

    class SourceStatusGroup:
        rank_in_group = rank
        world_size = 2

        @staticmethod
        def broadcast_object_list(box, src=0):
            calls.append("broadcast_object_list")
            box[0] = ("OSError", "checkpoint read failed")

    def operation():
        calls.append("operation")
        raise OSError("checkpoint read failed")

    with pytest.raises(
        RuntimeError, match="checkpoint map.*OSError.*checkpoint read failed"
    ):
        runtime.meta._collective_source_call(
            SourceStatusGroup(),
            is_src=rank == 0,
            operation=operation,
            context="checkpoint map",
        )

    assert calls == (
        ["operation", "broadcast_object_list"]
        if rank == 0
        else ["broadcast_object_list"]
    )


def test_collective_build_gate_reports_peer_failure_to_successful_rank(runtime):
    calls = []

    class Group:
        rank_in_group = 0
        world_size = 2

        @staticmethod
        def broadcast_object_list(box, src=0):
            calls.append(("broadcast", src))
            box[0] = None if src == 0 else ("ValueError", "config diverged")

    with pytest.raises(RuntimeError, match="meta transformer.*rank 1.*ValueError"):
        runtime.meta._collective_build_call(
            Group(), lambda: "built", context="meta transformer"
        )

    assert calls == [("broadcast", 0), ("broadcast", 1)]


def test_collective_build_gate_is_noop_for_single_rank(runtime):
    calls = []
    group = SimpleNamespace(
        rank_in_group=0,
        world_size=1,
        broadcast_object_list=lambda *args, **kwargs: calls.append("broadcast"),
    )

    assert (
        runtime.meta._collective_build_call(
            group, lambda: "built", context="meta transformer"
        )
        == "built"
    )
    assert not calls


def test_persistent_buffers_are_selected_from_each_owning_module(runtime):
    torch = runtime.torch
    module = torch.nn.Module()
    module.register_buffer("root_saved", torch.ones(1), persistent=True)
    module.register_buffer("root_cache", torch.ones(1), persistent=False)
    module.child = torch.nn.Module()
    module.child.register_buffer("saved", torch.ones(1), persistent=True)
    module.child.register_buffer("cache", torch.ones(1), persistent=False)

    assert [name for name, _ in runtime.meta._persistent_named_buffers(module)] == [
        "root_saved",
        "child.saved",
    ]


def test_block_fill_requires_and_broadcasts_only_persistent_buffers(runtime):
    torch = runtime.torch
    block = torch.nn.Module()
    block.register_parameter("weight", torch.nn.Parameter(torch.ones(1)))
    block.register_buffer("saved", torch.ones(1), persistent=True)
    block.register_buffer("runtime_cache", torch.ones(1), persistent=False)

    filler = object.__new__(runtime.meta._TransformerDiskFiller)
    filler.is_src = True
    filler.subfolder = "transformer"
    filler._id2fqn = {id(block): "blocks.0"}
    filler._ckpt_key = lambda root, name: name
    filler.weight_map = {
        "blocks.0.weight": "weights.safetensors",
        "blocks.0.saved": "weights.safetensors",
    }
    fills = []
    filler._fill = lambda module, local_name, key, required: fills.append(
        (local_name, key, required)
    )
    broadcasts = []
    filler.group = SimpleNamespace(
        rank_in_group=0,
        world_size=1,
        broadcast=lambda tensor, src=0: broadcasts.append(tensor),
        broadcast_object_list=lambda box, src=0: None,
        all_reduce=lambda tensor: tensor,
    )

    filler.fill_block(block, 1)

    assert fills == [
        ("weight", "blocks.0.weight", True),
        ("saved", "blocks.0.saved", True),
    ]
    assert len(broadcasts) == 2
    assert broadcasts[0].data_ptr() == block.weight.data_ptr()
    assert broadcasts[1].data_ptr() == block.saved.data_ptr()


@pytest.mark.parametrize("rank", [0, 1])
def test_block_read_failure_is_collective_before_tensor_broadcast(runtime, rank):
    torch = runtime.torch
    block = torch.nn.Module()
    block.register_parameter("weight", torch.nn.Parameter(torch.ones(1)))
    object_broadcasts = 0
    tensor_broadcasts = []
    layout_contract = runtime.meta._tensor_layout_contract(block)

    class Group:
        rank_in_group = rank
        world_size = 2

        @staticmethod
        def broadcast_object_list(box, src=0):
            nonlocal object_broadcasts
            object_broadcasts += 1
            if object_broadcasts == 1:
                box[0] = layout_contract
            elif object_broadcasts == 2:
                box[0] = []
            else:
                box[0] = ("OSError", "safetensors read failed")

        @staticmethod
        def broadcast(tensor, src=0):
            tensor_broadcasts.append(tensor)

        @staticmethod
        def all_reduce(tensor):
            return tensor

    filler = object.__new__(runtime.meta._TransformerDiskFiller)
    filler.is_src = rank == 0
    filler.group = Group()
    filler.subfolder = "transformer"
    filler._id2fqn = {id(block): "blocks.0"}
    filler._ckpt_key = lambda root, name: name
    filler.weight_map = {"blocks.0.weight": "weights.safetensors"} if rank == 0 else {}
    filler._fill = lambda *args, **kwargs: (_ for _ in ()).throw(
        OSError("safetensors read failed")
    )

    with pytest.raises(RuntimeError, match="loading checkpoint tensor.*OSError"):
        filler.fill_block(block, 1)

    assert object_broadcasts == 3
    assert not tensor_broadcasts


def test_missing_persistent_checkpoint_key_is_reported_to_peers_before_broadcast(
    runtime,
):
    filler = object.__new__(runtime.meta._TransformerDiskFiller)
    filler.is_src = False
    filler.subfolder = "transformer"
    filler.weight_map = {}
    calls = []

    def send_missing(box, src=0):
        calls.append("broadcast_object_list")
        box[0] = ["blocks.0.saved"]

    filler.group = SimpleNamespace(broadcast_object_list=send_missing)

    with pytest.raises(
        RuntimeError, match="missing checkpoint tensors.*blocks.0.saved"
    ):
        filler._require_checkpoint_keys(["blocks.0.saved"])

    assert calls == ["broadcast_object_list"]


def test_shard_component_restores_nonpersistent_block_buffers_before_disk_callback(
    runtime, monkeypatch
):
    torch = runtime.torch

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("weight", torch.nn.Parameter(torch.ones(1)))
            self.register_buffer("runtime_cache", torch.tensor([3.0]), persistent=False)

        def to_empty(self, *, device, recurse):
            self.runtime_cache = torch.empty_like(self.runtime_cache).fill_(-1)
            self.weight = torch.nn.Parameter(torch.empty_like(self.weight))
            return self

    component = torch.nn.Module()
    component.blocks = torch.nn.ModuleList([Block()])
    seen = []
    import torch.distributed._composable.fsdp as composable_fsdp

    monkeypatch.setattr(composable_fsdp, "fully_shard", lambda module, **kwargs: module)

    runtime.sharding.shard_component(
        component,
        wrap_attrs=["blocks"],
        process_group=None,
        device_id=0,
        quantize_fn=lambda block, index: None,
        load_block_fn=lambda block, index: seen.append(
            block.runtime_cache.detach().clone()
        ),
    )

    assert len(seen) == 1
    assert torch.equal(seen[0], torch.tensor([3.0]))
    assert torch.equal(component.blocks[0].runtime_cache, torch.tensor([3.0]))


def test_replicated_transformer_restores_nonpersistent_buffers_before_fill(
    runtime, monkeypatch
):
    torch = runtime.torch

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("weight", torch.nn.Parameter(torch.ones(1)))
            self.register_buffer("runtime_cache", torch.tensor([5.0]), persistent=False)

        def to_empty(self, *, device, recurse):
            self.runtime_cache = torch.empty_like(self.runtime_cache).fill_(-1)
            self.weight = torch.nn.Parameter(torch.empty_like(self.weight))
            return self

    component = torch.nn.Module()
    component.blocks = torch.nn.ModuleList([Block()])
    loader = object.__new__(runtime.meta.MemoryEfficientLoader)
    loader.model = SimpleNamespace()
    seen = []
    loader.build_transformer_disk_loaders = lambda *args, **kwargs: (
        lambda block, index: seen.append(block.runtime_cache.detach().clone()),
        lambda component: None,
    )
    monkeypatch.setattr(
        runtime.shard,
        "build_block_quantize_fn",
        lambda *args, **kwargs: None,
    )

    loader._fill_transformer_replicated(
        component,
        "transformer",
        {"wrap_attrs": ["blocks"]},
        "cpu",
        SimpleNamespace(local_rank=0),
    )

    assert len(seen) == 1
    assert torch.equal(seen[0], torch.tensor([5.0]))
    assert torch.equal(component.blocks[0].runtime_cache, torch.tensor([5.0]))


@pytest.mark.parametrize("rank", [0, 1])
def test_replicated_quantize_failure_stops_all_ranks_before_next_block(
    runtime, monkeypatch, rank
):
    torch = runtime.torch
    component = torch.nn.Module()
    component.blocks = torch.nn.ModuleList(
        [torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)]
    )
    fills = []
    finalized = []
    loader = object.__new__(runtime.meta.MemoryEfficientLoader)
    loader.model = SimpleNamespace()
    loader.build_transformer_disk_loaders = lambda *args, **kwargs: (
        lambda block, index: fills.append(index),
        lambda component: finalized.append(True),
    )

    def quantize(block, index):
        if rank == 1:
            raise ValueError("rank-local quantization failed")

    monkeypatch.setattr(
        runtime.shard,
        "build_block_quantize_fn",
        lambda *args, **kwargs: quantize,
    )

    class Group:
        rank_in_group = rank
        world_size = 2
        local_rank = rank

        @staticmethod
        def broadcast_object_list(box, src=0):
            box[0] = (
                ("ValueError", "rank-local quantization failed") if src == 1 else None
            )

    with pytest.raises(RuntimeError, match="quantizing replicated transformer block 0"):
        loader._fill_transformer_replicated(
            component,
            "transformer",
            {"wrap_attrs": ["blocks"]},
            "cpu",
            Group(),
        )

    assert fills == [0]
    assert not finalized


@pytest.mark.parametrize("rank", [0, 1])
def test_text_encoder_source_state_failure_is_collective_before_scatter(
    runtime, monkeypatch, rank
):
    torch = runtime.torch
    object_broadcasts = []
    scatters = []
    releases = []

    class Group:
        rank_in_group = rank
        world_size = 2

        @staticmethod
        def broadcast_object_list(box, src=0):
            object_broadcasts.append(box[0])
            box[0] = ("OSError", "state dict construction failed")

    class Source:
        def state_dict(self):
            raise OSError("state dict construction failed")

    loader = object.__new__(runtime.meta.MemoryEfficientLoader)
    loader.model = SimpleNamespace(
        settings=SimpleNamespace(fsdp_strategy={"text_encoder": {"wrap_attrs": []}})
    )
    loader._load_rank0_source = lambda *args: Source()
    loader._release_rank0_source = lambda is_src, name: releases.append((is_src, name))
    monkeypatch.setattr(runtime.meta, "get_fs_group", lambda: Group())
    import torch.distributed.checkpoint.state_dict as checkpoint_state

    monkeypatch.setattr(
        checkpoint_state,
        "set_model_state_dict",
        lambda *args, **kwargs: scatters.append(args),
    )

    with pytest.raises(RuntimeError, match="loading text_encoder source.*OSError"):
        loader._broadcast_load_component(
            torch.nn.Linear(1, 1), "text_encoder", offload=False
        )

    assert len(object_broadcasts) == 1
    assert not scatters
    assert releases == [(rank == 0, "text_encoder")]

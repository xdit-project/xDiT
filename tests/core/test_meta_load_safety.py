"""CPU-only regression tests for collective/meta-load safety helpers."""

import inspect
from contextlib import ExitStack, nullcontext
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
    loader = object.__new__(runtime.meta.ModelLoader)
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
    loader = object.__new__(runtime.meta.ModelLoader)
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

    filler = object.__new__(runtime.meta._BlockwiseDiskFiller)
    filler._used_keys = set()
    filler.is_src = True
    filler.subfolder = "transformer"
    filler._id2fqn = {id(block): "blocks.0"}
    filler._ckpt_key = lambda root, name: name
    filler._handle_cache = {}
    filler._stack = ExitStack()
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


def _handle_counting_filler(runtime, monkeypatch, device="cpu"):
    """A filler wired to a fake safe_open, reporting the open/close order it drives."""
    import safetensors

    opened, closed = [], []

    class FakeHandle:
        def __init__(self, path, device):
            self.path = path
            self.device = device

        def __enter__(self):
            opened.append((self.path, self.device))
            return self

        def __exit__(self, *exc):
            closed.append((self.path, self.device))
            return False

    monkeypatch.setattr(
        safetensors,
        "safe_open",
        lambda path, **kwargs: FakeHandle(path, kwargs.get("device")),
    )

    filler = object.__new__(runtime.meta._BlockwiseDiskFiller)
    filler._used_keys = set()
    filler.device = device
    filler._handle_cache = {}
    filler._stack = ExitStack()
    return filler, opened, closed


def test_shards_are_read_onto_the_device_being_filled(runtime, monkeypatch):
    """A handle opened on host retains a copy of every tensor read through it; one opened on the
    accelerator retains nothing and skips the host staging copy, so the read follows the fill."""
    filler, opened, _ = _handle_counting_filler(runtime, monkeypatch, device="cuda:1")

    filler._handle("shard-0.safetensors")

    assert opened == [("shard-0.safetensors", "cuda:1")]


def test_a_cpu_fill_still_reads_onto_the_host(runtime, monkeypatch):
    """Offload fills target CPU, where there is no device to read onto."""
    filler, opened, _ = _handle_counting_filler(runtime, monkeypatch, device="cpu")

    filler._handle("shard-0.safetensors")

    assert opened == [("shard-0.safetensors", "cpu")]


def test_only_one_shard_stays_mapped_during_a_fill(runtime, monkeypatch):
    """An open safe_open handle retains a host copy of every tensor read through it, so holding one
    per shard makes host anon track the whole transformer rather than the shard being read - the
    cost this per-block fill exists to avoid."""
    filler, opened, closed = _handle_counting_filler(runtime, monkeypatch)

    filler._handle("shard-0.safetensors")
    filler._handle("shard-1.safetensors")

    assert [path for path, _ in opened] == [
        "shard-0.safetensors",
        "shard-1.safetensors",
    ]
    assert [path for path, _ in closed] == ["shard-0.safetensors"]
    assert list(filler._handle_cache) == ["shard-1.safetensors"]


def test_reads_within_one_shard_reuse_its_handle(runtime, monkeypatch):
    """Releasing is what costs time, so a shard must not be reopened per tensor or per block."""
    filler, opened, closed = _handle_counting_filler(runtime, monkeypatch)

    for _ in range(4):
        filler._handle("shard-0.safetensors")

    assert [path for path, _ in opened] == ["shard-0.safetensors"]
    assert closed == []


def test_releasing_handles_lets_a_later_read_reopen(runtime, monkeypatch):
    """finalize releases at the end of the component; the cache must not then hand back a closed
    handle if anything reads again."""
    filler, opened, closed = _handle_counting_filler(runtime, monkeypatch)

    filler._handle("shard-0.safetensors")
    filler._release_handles()
    filler._handle("shard-0.safetensors")

    assert [path for path, _ in opened] == ["shard-0.safetensors"] * 2
    assert [path for path, _ in closed] == ["shard-0.safetensors"]


def test_local_block_fill_uses_no_collective_transport(runtime):
    torch = runtime.torch
    block = torch.nn.Module()
    block.register_parameter("weight", torch.nn.Parameter(torch.ones(1)))
    block.register_buffer("saved", torch.ones(1), persistent=True)

    filler = object.__new__(runtime.meta._BlockwiseDiskFiller)
    filler._used_keys = set()
    filler.group = None
    filler.is_src = True
    filler.device = "cpu"
    filler.subfolder = "transformer"
    filler._id2fqn = {id(block): "blocks.0"}
    filler._ckpt_key = lambda root, name: name
    filler._handle_cache = {}
    filler._stack = ExitStack()
    filler.weight_map = {
        "blocks.0.weight": "weights.safetensors",
        "blocks.0.saved": "weights.safetensors",
    }
    fills = []
    filler._fill = lambda module, local_name, key, required: fills.append(
        (local_name, key)
    )

    filler.fill_block(block, 1)

    assert fills == [
        ("weight", "blocks.0.weight"),
        ("saved", "blocks.0.saved"),
    ]


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

    filler = object.__new__(runtime.meta._BlockwiseDiskFiller)
    filler._used_keys = set()
    filler.is_src = rank == 0
    filler.group = Group()
    filler.subfolder = "transformer"
    filler._id2fqn = {id(block): "blocks.0"}
    filler._ckpt_key = lambda root, name: name
    filler.weight_map = {"blocks.0.weight": "weights.safetensors"} if rank == 0 else {}
    filler._fill = lambda *args, **kwargs: (_ for _ in ()).throw(
        OSError("safetensors read failed")
    )

    with pytest.raises(
        RuntimeError, match="loading transformer checkpoint tensors.*OSError"
    ):
        filler.fill_block(block, 1)

    # Layout, required-keys, then one status for the whole block's reads. The reads share a status
    # exchange rather than paying one each; test_a_failing_read_names_the_tensor_it_could_not_read
    # covers the key surviving that batching.
    assert object_broadcasts == 3
    assert not tensor_broadcasts


def test_one_rank_does_the_reading_however_wide_the_group_is(runtime):
    """Spreading the read across ranks was measured slower, not faster, and is deliberately not done.

    Rotating by block cost 37.2s cold against 32.7s for a single reader, because it puts every rank
    at a different offset in the same shard and defeats readahead, and it took resident page cache
    from ~10GB to 35.6GB as each rank mapped each shard. Pinning this keeps the next reader of the
    code from re-deriving it the expensive way.
    """
    filler = object.__new__(runtime.meta._BlockwiseDiskFiller)
    filler._used_keys = set()
    filler.group = SimpleNamespace(world_size=8, rank_in_group=3)

    readers = [filler._reader_for_block(i) for i in range(9)]

    assert readers == [0] * 9


def test_a_single_rank_fill_still_reads_on_itself(runtime):
    """The reader choice must never name a rank that does not exist."""
    filler = object.__new__(runtime.meta._BlockwiseDiskFiller)
    filler._used_keys = set()
    filler.group = None

    assert filler._reader_for_block(5) == 0


def test_every_rank_gets_the_checkpoint_map_so_any_of_them_can_read(runtime):
    """Resolving stays on rank0 to avoid redundant hub HEADs, but the result has to reach everyone.

    A peer with an empty weight map would report every key missing the moment it was asked to read.
    """
    filler = object.__new__(runtime.meta._BlockwiseDiskFiller)
    filler._used_keys = set()
    calls = []

    class Group:
        world_size = 2
        rank_in_group = 1

        @staticmethod
        def broadcast_object_list(box, src=0):
            calls.append(src)
            # First the read's failure status, then the map itself.
            box[0] = None if len(calls) == 1 else {"blocks.0.weight": "shard-0.safetensors"}

    filler.group = Group()
    filler.is_src = False

    shared = filler._share_from_source(
        lambda: pytest.fail("a peer must not resolve the map itself"),
        context="resolving",
    )

    assert shared == {"blocks.0.weight": "shard-0.safetensors"}
    assert calls == [0, 0]


def test_a_shard_is_dropped_only_once_nothing_still_needs_it(runtime, monkeypatch):
    """A handle closing does not mean a shard is finished, so it is the wrong moment to drop.

    The tail fill reaches back for keys in shards the block walk already closed, and dropping on
    close evicted pages that were about to be read again. The shard's last key being consumed is the
    condition that actually means finished.
    """
    dropped = []
    monkeypatch.setattr(runtime.meta, "drop_file_page_cache", lambda paths: dropped.extend(paths))

    filler = object.__new__(runtime.meta._BlockwiseDiskFiller)
    filler._used_keys = set()
    filler.weight_map = {
        "blocks.0.weight": "shard-a",
        "blocks.1.weight": "shard-a",
        "blocks.2.weight": "shard-b",
    }
    filler._unread_by_shard = {
        "shard-a": {"blocks.0.weight", "blocks.1.weight"},
        "shard-b": {"blocks.2.weight"},
    }

    filler._retire_keys(["blocks.0.weight"])
    assert dropped == []

    filler._retire_keys(["blocks.1.weight"])
    assert dropped == ["shard-a"]

    filler._retire_keys(["blocks.2.weight"])
    assert dropped == ["shard-a", "shard-b"]


def test_a_shard_is_streamed_in_before_it_is_mapped(runtime, monkeypatch):
    """mmap faults are what bounded the read, so the shard is streamed first to make them cache hits.

    Measured 0.62 GB/s faulting through mmap against 3.21 GB/s streaming the same file with pread,
    and 9.05s -> 4.39s reading a real 10GB shard to device. Warming has to happen once per shard and
    before the handle exists, or it is either repeated per tensor or too late to help.
    """
    warmed = []
    monkeypatch.setenv("XDIT_WARM_SHARDS", "1")
    monkeypatch.setattr(runtime.meta, "warm_file_page_cache", lambda path: warmed.append(path))
    # _handle imports safe_open at call time, so the module it imports from is what has to be patched.
    import safetensors

    monkeypatch.setattr(safetensors, "safe_open", lambda *a, **k: nullcontext("handle"))

    filler = object.__new__(runtime.meta._BlockwiseDiskFiller)
    filler._used_keys = set()
    filler.device = "cpu"
    filler._handle_cache = {}
    filler._stack = ExitStack()

    filler._handle("shard-a")
    filler._handle("shard-a")

    assert warmed == ["shard-a"], "a shard must be streamed once, not once per tensor read"

    filler._handle("shard-b")

    assert warmed == ["shard-a", "shard-b"]


def _prefetch_filler(runtime, monkeypatch, warmed, order, unread=None):
    """A filler wired to record warms instead of touching disk."""
    import safetensors

    monkeypatch.setenv("XDIT_WARM_SHARDS", "2")
    monkeypatch.setattr(runtime.meta, "warm_file_page_cache", lambda path: warmed.append(path))
    monkeypatch.setattr(runtime.meta, "drop_file_page_cache", lambda paths: None)
    monkeypatch.setattr(safetensors, "safe_open", lambda *a, **k: nullcontext("handle"))

    filler = object.__new__(runtime.meta._BlockwiseDiskFiller)
    filler._used_keys = set()
    filler.device = "cpu"
    filler._handle_cache = {}
    filler._stack = ExitStack()
    filler._next_shard = dict(zip(order, order[1:]))
    filler.weight_map = {f"k{i}": p for i, p in enumerate(order)}
    # Each shard owes exactly its own key, so retiring that key retires the shard.
    filler._unread_by_shard = {
        p: {k for k, q in filler.weight_map.items() if q == p} for p in order
    }
    if unread is not None:
        filler._unread_by_shard = {p: set(unread.get(p, ())) for p in order}
    filler._streamed = set()
    filler._prefetch_thread = None
    return filler


def test_the_next_shard_is_streamed_while_this_one_is_consumed(runtime, monkeypatch):
    """Warming is the part of the read still serialised against it, so it overlaps the compute.

    The work after a read -- quantise, broadcast, shard -- needs no disk, which is what the next
    shard's stream hides under. Opening a shard has to leave the following one already in flight.
    """
    warmed = []
    filler = _prefetch_filler(runtime, monkeypatch, warmed, ["a", "b", "c"])

    filler._handle("a")
    filler._join_prefetch()

    assert warmed == ["a", "b"], "opening 'a' must stream 'a' and then run ahead to 'b'"

    filler._handle("b")
    filler._join_prefetch()

    assert warmed == ["a", "b", "c"], "'b' was already streamed, so it must not be streamed twice"


def test_only_one_shard_is_ever_in_flight(runtime, monkeypatch):
    """Two shards resident is the deliberate cost; an unbounded run-ahead would be the whole file.

    That is the failure that sank rotation, so the depth is pinned rather than left to chance.
    """
    warmed = []
    filler = _prefetch_filler(runtime, monkeypatch, warmed, ["a", "b", "c", "d", "e"])

    filler._handle("a")
    filler._join_prefetch()

    assert warmed == ["a", "b"], "run-ahead must stop at one shard, not stream the rest"


def test_a_retired_shard_is_not_streamed_back_in(runtime, monkeypatch):
    """Re-reading a finished shard would restore the cache that dropping it exists to release.

    Nothing left unread is exactly the condition _retire_keys drops on, so it is the condition that
    has to suppress the run-ahead too.
    """
    warmed = []
    filler = _prefetch_filler(
        runtime, monkeypatch, warmed, ["a", "b"], unread={"a": {"k0"}, "b": set()}
    )

    filler._handle("a")
    filler._join_prefetch()

    assert warmed == ["a"], "'b' is already retired, so streaming it back in undoes the drop"


def test_a_stream_never_outlives_the_fill_that_wanted_it(runtime, monkeypatch):
    """finalize drops the page cache, and a prefetch still running would put it straight back."""
    warmed = []
    filler = _prefetch_filler(runtime, monkeypatch, warmed, ["a", "b"])

    filler._handle("a")
    filler._join_prefetch()

    assert filler._prefetch_thread is None


def test_a_reopened_shard_is_not_streamed_twice(runtime, monkeypatch):
    """The tail fill reaches back for keys in shards the block walk already closed.

    Those pages are still cached, so restreaming one would pay for the whole shard again to learn
    nothing. Only retiring it, which actually drops the pages, makes a restream the right call.
    """
    warmed = []
    filler = _prefetch_filler(runtime, monkeypatch, warmed, ["a", "b"])

    filler._handle("a")
    filler._join_prefetch()
    filler._release_handles()
    filler._handle("a")

    assert warmed == ["a", "b"], "'a' is still cached, so reopening it must not restream it"

    filler._retire_keys(["k0"])
    filler._release_handles()
    filler._handle("a")

    assert warmed == ["a", "b", "a"], "once retired its pages are gone, so it must be restreamed"


def test_finalize_stops_the_stream_before_dropping_the_cache(runtime):
    """Order matters: joining after the drop would leave the restored pages behind."""
    source = inspect.getsource(runtime.meta._BlockwiseDiskFiller.finalize)
    join, drop = source.index("_join_prefetch"), source.index("drop_file_page_cache(self.shard_paths)")

    assert join < drop, "the stream has to be stopped before the cache is dropped, not after"


@pytest.mark.parametrize(
    "raw, depth",
    [
        ("0", 0),
        ("no", 0),
        ("1", 1),
        ("2", 2),
        ("true", 2),
        # A typo in a performance knob must not quietly hand back the slow path.
        ("yes-please", 2),
    ],
)
def test_how_many_shards_may_be_held_warm_is_configurable(runtime, monkeypatch, raw, depth):
    """The depth is the read-speed against host-cache trade, so it is stated rather than implied."""
    monkeypatch.setenv("XDIT_WARM_SHARDS", raw)

    assert runtime.meta._warm_shard_depth() == depth


def test_nothing_is_streamed_when_the_depth_is_zero(runtime, monkeypatch):
    """The escape hatch has to actually reach the read, not just the run-ahead.

    Storage where a sequential pre-read is not free, such as a network mount that would pay for the
    bytes twice, needs the whole thing off.
    """
    warmed = []
    filler = _prefetch_filler(runtime, monkeypatch, warmed, ["a", "b"])
    monkeypatch.setenv("XDIT_WARM_SHARDS", "0")

    filler._handle("a")
    filler._join_prefetch()

    assert warmed == []


def test_the_run_ahead_can_be_dropped_without_losing_the_warm(runtime, monkeypatch):
    """Depth 1 is the middle setting: fast reads, one shard resident instead of two."""
    warmed = []
    filler = _prefetch_filler(runtime, monkeypatch, warmed, ["a", "b"])
    monkeypatch.setenv("XDIT_WARM_SHARDS", "1")

    filler._handle("a")
    filler._join_prefetch()

    assert warmed == ["a"]


def test_only_the_reading_rank_streams_a_shard(runtime):
    """Warming on every rank is the cache balloon that sank rotation, so it must sit behind the read.

    Every rank warming every shard would take resident page cache from ~10GB to the whole checkpoint
    while reading no faster, since the bytes still arrive over one stream to one reader.
    """
    # _fill and the tensor read it delegates to, since the handle may sit in either
    source = inspect.getsource(
        runtime.meta._BlockwiseDiskFiller._fill
    ) + inspect.getsource(runtime.meta._BlockwiseDiskFiller._tensor_for)

    assert "self._handle(" in source, "warming must stay behind _fill, which only the reader runs"
    assert "warm_file_page_cache" not in inspect.getsource(
        runtime.meta._BlockwiseDiskFiller._read_tensors
    ), "warming above the source call would run it on every rank"


def test_phase_timing_stays_off_unless_asked_for(runtime, monkeypatch):
    """The breakdown synchronises to attribute time correctly, and that costs the fill 2.3x.

    So it has to be opt-in. A default-on breakdown would make every production load pay for a
    diagnostic, which is the opposite of what this load path exists to do.
    """
    monkeypatch.delenv("XDIT_FILL_PHASE_TIMING", raising=False)
    filler = object.__new__(runtime.meta._BlockwiseDiskFiller)
    filler._used_keys = set()

    with filler._timed("read"):
        pass

    assert not getattr(filler, "_phase_seconds", None)

    monkeypatch.setenv("XDIT_FILL_PHASE_TIMING", "1")
    with filler._timed("read"):
        pass

    assert "read" in filler._phase_seconds


def test_a_failing_read_names_the_tensor_it_could_not_read(runtime):
    """One status exchange per block must not cost the reader which tensor failed.

    Peers only ever see the message text the status carries, so the key has to be inside it; without
    that, a missing weight would name the block and leave fifteen candidates.
    """
    torch = runtime.torch
    module = torch.nn.Module()
    filler = object.__new__(runtime.meta._BlockwiseDiskFiller)
    filler._used_keys = set()
    filler.group = None
    filler.is_src = True
    filler.subfolder = "transformer"
    filler._fill = lambda *args, **kwargs: (_ for _ in ()).throw(
        OSError("safetensors read failed")
    )

    with pytest.raises(RuntimeError, match="blocks.0.attn.qkv.weight"):
        filler._read_tensors(module, [("weight", "blocks.0.attn.qkv.weight")])


def test_missing_persistent_checkpoint_key_is_reported_to_peers_before_broadcast(
    runtime,
):
    filler = object.__new__(runtime.meta._BlockwiseDiskFiller)
    filler._used_keys = set()
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

    # shard_component places blocks on device_id, so the restored buffers are on
    # the accelerator wherever one exists; compare on the host.
    assert len(seen) == 1
    assert torch.equal(seen[0].cpu(), torch.tensor([3.0]))
    assert torch.equal(component.blocks[0].runtime_cache.cpu(), torch.tensor([3.0]))


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
    loader = object.__new__(runtime.meta.ModelLoader)
    loader.model = SimpleNamespace()
    seen = []
    loader.build_blockwise_disk_loaders = lambda *args, **kwargs: (
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


def test_local_transformer_fills_and_quantizes_each_block_before_tail(
    runtime,
):
    torch = runtime.torch
    component = torch.nn.Module()
    component.blocks = torch.nn.ModuleList(
        [torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)]
    )
    loader = object.__new__(runtime.meta.ModelLoader)
    loader.model = SimpleNamespace()
    events = []

    def fill(block, index):
        block.weight.data.fill_(index + 1)
        events.append(f"fill:{index}")

    def finalize(module):
        events.append("tail")

    def quantize(block, index):
        assert block.weight.item() == index + 1
        events.append(f"quantize:{index}")

    loader.build_blockwise_disk_loaders = lambda *args, **kwargs: (
        fill,
        finalize,
    )

    loader.fill_transformer_local(
        component,
        "transformer",
        {"wrap_attrs": ["blocks"]},
        "cpu",
        quantize_fn=quantize,
    )

    assert events == ["fill:0", "quantize:0", "fill:1", "quantize:1", "tail"]


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
    loader = object.__new__(runtime.meta.ModelLoader)
    loader.model = SimpleNamespace()
    loader.build_blockwise_disk_loaders = lambda *args, **kwargs: (
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

    loader = object.__new__(runtime.meta.ModelLoader)
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

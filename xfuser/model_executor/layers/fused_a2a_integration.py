"""Feature-gated MoRI fused all-to-all integration for Wan USP."""

import inspect
import os
import socket

import torch
import torch.distributed as dist

from xfuser.logger import init_logger

try:
    from torch._higher_order_ops.effects import (
        _EffectType,
        _register_effectful_op,
        with_effects,
    )
except (AttributeError, ImportError):
    _EffectType = None
    _register_effectful_op = None
    with_effects = None

try:
    from torch.fx.node import has_side_effect
except (AttributeError, ImportError):
    has_side_effect = None

from xfuser.config.attention_a2a import AttentionA2AConfig

logger = init_logger(__name__)

_CUSTOM_OP_OPTIONS = (
    {"tags": (torch.Tag.cudagraph_unsafe,)}
    if hasattr(torch, "Tag") and "tags" in inspect.signature(torch.library.custom_op).parameters
    else {}
)
_ATTENTION_A2A_POLICY = AttentionA2AConfig()
_ATTENTION_A2A_CONFIG = AttentionA2AConfig()
_ATTENTION_A2A_ENABLED = False
_FUSED_A2A_SIDESTREAM = False
_FUSED_A2A_INTERLEAVE = False
_FUSED_A2A_HADAMARD_PLACEMENT = "none"
_TRANSPORT_HADAMARD = False
_FUSED_A2A_CODECS = _ATTENTION_A2A_CONFIG.codecs
_FUSED_A2A_PACKED = False
_FUSED_A2A_V_PACK = "default"

_MORI_GROUP_KEY = None
_MORI_CPU_GROUP = None
_OP_CACHE = {}
_INPUT_SIDE_STREAMS = {}
_INPUT_CONSUMER_DONE = {}
_INPUT_PENDING = {}
_INPUT_COLLECTIVE_WAITS = {}
_INPUT_COLLECTIVE_LIB = None
_FUSED_A2A_COLLECTIVE = False
_ATTENTION_A2A_TIER2_ENABLED = False
_ATTENTION_A2A_TIER2_REASON = "Attention A2A is disabled"
_ATTENTION_A2A_TIER1_ENABLED = False
_ATTENTION_A2A_TIER1_REASON = "Attention A2A is disabled"
_ATTENTION_A2A_PACKED_LAUNCHER = None
_MAX_OP_CACHE_ENTRIES = 4
_ATTENTION_A2A_POISONED = None


def _call_contract_error(name, function, *args, **kwargs):
    if not callable(function):
        return f"{name} is unavailable or not callable"
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as exc:
        return f"{name} has no inspectable signature: {exc}"
    try:
        signature.bind(*args, **kwargs)
    except TypeError as exc:
        return f"{name} has an incompatible signature {signature}: {exc}"
    return None


def _tier2_support_status():
    """Probe the private Inductor contract needed by the async lowering."""
    actual = (torch.__version__, torch.version.git_version)
    try:
        from torch._inductor import config, ir
        from torch._inductor.lowering import (
            add_layout_constraint,
            constrain_to_fx_strides,
            register_lowering,
        )
    except (AttributeError, ImportError) as exc:
        return False, f"required Inductor lowering API is unavailable: {exc}"
    collective = getattr(ir, "_CollectiveKernel", None)
    wait = getattr(ir, "_WaitKernel", None)
    tensor_box = getattr(ir, "TensorBox", None)
    placeholder = object()
    contracts = (
        (
            "_CollectiveKernel.create_out_of_place",
            getattr(collective, "create_out_of_place", None),
            (placeholder,) * 9,
            {},
        ),
        (
            "_WaitKernel.create_wait",
            getattr(wait, "create_wait", None),
            (placeholder, placeholder),
            {},
        ),
        (
            "register_lowering",
            register_lowering,
            (placeholder,),
            {"type_promotion_kind": None},
        ),
        (
            "add_layout_constraint",
            add_layout_constraint,
            (placeholder, placeholder),
            {},
        ),
        (
            "constrain_to_fx_strides",
            constrain_to_fx_strides,
            (placeholder,),
            {},
        ),
        (
            "has_side_effect",
            has_side_effect,
            (placeholder,),
            {},
        ),
        (
            "TensorBox.create",
            getattr(tensor_box, "create", None),
            (placeholder,),
            {},
        ),
    )
    for name, function, args, kwargs in contracts:
        error = _call_contract_error(name, function, *args, **kwargs)
        if error is not None:
            return False, f"{error}; Torch build={actual}"
    required_config = (
        "reorder_for_compute_comm_overlap",
        "cpp_wrapper",
    )
    missing_config = [name for name in required_config if not hasattr(config, name)]
    if missing_config:
        return (
            False,
            "required Inductor config option(s) unavailable: " + ", ".join(missing_config) + f"; Torch build={actual}",
        )
    if getattr(getattr(torch, "Tag", None), "cudagraph_unsafe", None) is None:
        return False, f"torch.Tag.cudagraph_unsafe is unavailable; Torch build={actual}"
    return True, None


def _tier1_support_status():
    """Return whether Torch can preserve ordered side-effecting custom ops."""
    required = (
        _EffectType,
        _register_effectful_op,
        with_effects,
        has_side_effect,
        getattr(torch.library, "custom_op", None),
    )
    if any(symbol is None for symbol in required):
        return (
            False,
            "required Torch ordered-effect custom-op APIs are unavailable",
        )
    return True, None


def _register_ordered_effect(op):
    """Register one Tier-1 op when the current Torch exposes ordered effects."""
    if not _tier1_support_status()[0]:
        return
    _register_effectful_op(op, _EffectType.ORDERED)
    has_side_effect(op)


def _register_input_collective():
    from torch._inductor import config, ir
    from torch._inductor.lowering import (
        add_layout_constraint,
        constrain_to_fx_strides,
        register_lowering,
    )

    config.reorder_for_compute_comm_overlap = True
    lib = torch.library.Library("xfuser", "FRAGMENT")
    lib.define(
        "fused_a2a_input_collective(Tensor input, Tensor lifetime, "
        "Tensor[] previous, int role, str profile, str group_name, "
        "int rank, int world_size) -> Tensor[]",
        tags=(torch.Tag.cudagraph_unsafe,),
    )
    lib.define(
        "fused_a2a_input_wait(Tensor input) -> Tensor",
        tags=(torch.Tag.cudagraph_unsafe,),
    )

    def submit(input, lifetime, previous, role, profile, group_name, rank, world_size):
        del lifetime, previous
        _require_active_profile(profile)
        group = dist.distributed_c10d._resolve_process_group(group_name)
        handle = (group_name, rank, input.device.index)
        pending = _INPUT_PENDING.get(handle)
        if role == 0 and pending is not None:
            raise ValueError("previous interleaved input has not been consumed")
        pending = _submit_input_role(input, role, group, rank, pending)
        _INPUT_PENDING[handle] = pending
        result = pending["results"][-1]
        payload = result.payload
        scale = result.scale
        done = torch.cuda.Event()
        done.record(pending["side"])
        for tensor in (payload, scale):
            _INPUT_COLLECTIVE_WAITS[tensor.data_ptr()] = (done, handle, role)
        return [payload, scale]

    def submit_fake(input, lifetime, previous, role, profile, group_name, rank, world_size):
        del lifetime, previous, group_name, rank
        head_major = input.transpose(1, 2)
        payloads, scales = _fake_packed_raw_outputs(
            head_major,
            profile,
            world_size,
        )
        return [payloads[role], scales[role]]

    def wait(input):
        done, handle, role = _INPUT_COLLECTIVE_WAITS.pop(input.data_ptr())
        torch.cuda.current_stream(input.device).wait_event(done)
        if role == 2:
            _INPUT_PENDING.pop(handle, None)
        return input

    lib.impl("fused_a2a_input_collective", submit, "CUDA")
    lib.impl("fused_a2a_input_wait", wait, "CUDA")
    torch.library.register_fake("xfuser::fused_a2a_input_collective", submit_fake)
    torch.library.register_fake("xfuser::fused_a2a_input_wait", lambda input: input)
    submit_op = torch.ops.xfuser.fused_a2a_input_collective.default
    wait_op = torch.ops.xfuser.fused_a2a_input_wait.default
    for op in (submit_op, wait_op):
        add_layout_constraint(op, constrain_to_fx_strides)
        has_side_effect(op)

    @register_lowering(submit_op, type_promotion_kind=None)
    def lower_submit(input, lifetime, previous, role, profile, group_name, rank, world_size):
        outputs = ir._CollectiveKernel.create_out_of_place(
            submit_op,
            input,
            lifetime,
            previous,
            role,
            profile,
            group_name,
            rank,
            world_size,
        )
        return [ir.TensorBox.create(output) for output in outputs]

    @register_lowering(wait_op, type_promotion_kind=None)
    def lower_wait(input):
        if config.cpp_wrapper:
            raise RuntimeError("Attention A2A Tier-2 wait does not support the C++ wrapper")
        ir._WaitKernel.create_wait(wait_op, input)
        return input

    return lib


def _heap_size_bytes(value):
    suffixes = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    text = value.strip().upper()
    suffix = text[-1] if text and text[-1].isalpha() else ""
    number = text[:-1] if suffix else text
    if suffix not in suffixes:
        raise ValueError(f"unsupported size suffix {suffix!r}")
    return int(number) * suffixes[suffix]


def _prepare_attention_a2a_packed_launcher():
    """Resolve the traceable packed MHA launcher before model compilation."""
    global _ATTENTION_A2A_PACKED_LAUNCHER
    if _ATTENTION_A2A_PACKED_LAUNCHER is None:
        from xfuser.core.attention.backends.aiter_mha_v4.kernel import (
            mha_v4_attention_a2a_packed,
        )

        _ATTENTION_A2A_PACKED_LAUNCHER = mha_v4_attention_a2a_packed


def preflight_attention_a2a(config: AttentionA2AConfig) -> None:
    """Collectively validate the optional A2A runtime before model execution."""
    if not config.enabled:
        return
    error = None
    try:
        import mori.shmem as ms
        from aiter.ops.flydsl.attention_a2a_intranode import (
            AttentionA2AIntraNodeOp,
            PackedRoleResult,
        )

        _prepare_attention_a2a_packed_launcher()
        if getattr(PackedRoleResult, "_fields", ()) != ("payload", "scale"):
            raise RuntimeError("installed AITER has an incompatible PackedRoleResult contract")
        source = inspect.getsource(AttentionA2AIntraNodeOp.__init__)
        required_codecs = {"e4m3_pc", "mxfp6_p"} if config.is_auto else set(config.codecs)
        missing = sorted(codec for codec in required_codecs & {"e4m3_pc", "mxfp6_p"} if repr(codec) not in source)
        if missing:
            raise RuntimeError("installed AITER lacks Attention A2A codec(s) " + ", ".join(missing))
        required_mori = (
            "shmem_torch_process_group_init",
            "mori_shmem_create_tensor",
            "shmem_finalize",
        )
        missing_mori = [name for name in required_mori if not hasattr(ms, name)]
        if missing_mori:
            raise RuntimeError("installed MORI lacks " + ", ".join(missing_mori))
        heap = os.environ.get("MORI_SHMEM_HEAP_SIZE")
        if not heap or _heap_size_bytes(heap) <= 0:
            raise RuntimeError(
                "MORI_SHMEM_HEAP_SIZE must be set before enabling Attention A2A "
                "(12G is the validated Wan configuration)"
            )
        arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]
        if arch not in {"gfx942", "gfx950"}:
            raise RuntimeError(f"Attention A2A requires gfx942 or gfx950, found {arch}")
        if (config.is_auto or any(codec.startswith("mxfp") for codec in config.codecs)) and arch != "gfx950":
            raise RuntimeError("MX Attention A2A profiles require gfx950")
        if dist.is_initialized():
            hosts = [None] * dist.get_world_size()
            dist.all_gather_object(hosts, socket.gethostname())
            if len(set(hosts)) != 1:
                raise RuntimeError("Attention A2A is intranode-only; all ranks must share one host")
    except Exception as exc:  # noqa: BLE001 - must report any rank-local preflight failure
        error = f"{type(exc).__name__}: {exc}"

    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
        failures = [f"rank {rank}: {message}" for rank, message in enumerate(errors) if message is not None]
        if failures:
            raise RuntimeError("Attention A2A preflight failed collectively: " + "; ".join(failures))
    elif error is not None:
        raise RuntimeError(f"Attention A2A preflight failed: {error}")


def configure_attention_a2a(
    config: AttentionA2AConfig,
    attention_backend=None,
) -> None:
    """Install the process policy and select its initial backend recipe."""
    global _ATTENTION_A2A_POLICY
    global _ATTENTION_A2A_TIER2_ENABLED
    global _ATTENTION_A2A_TIER2_REASON
    global _ATTENTION_A2A_TIER1_ENABLED
    global _ATTENTION_A2A_TIER1_REASON
    global _INPUT_COLLECTIVE_LIB

    if not isinstance(config, AttentionA2AConfig):
        raise TypeError("Attention A2A configuration must be AttentionA2AConfig")
    if config != _ATTENTION_A2A_POLICY and (
        _OP_CACHE or _INPUT_SIDE_STREAMS or _INPUT_PENDING or _INPUT_CONSUMER_DONE or _MORI_GROUP_KEY is not None
    ):
        raise RuntimeError("Attention A2A policy cannot change after its runtime state is created")
    if not config.enabled:
        _ATTENTION_A2A_POLICY = config
        _ATTENTION_A2A_TIER2_ENABLED = False
        _ATTENTION_A2A_TIER2_REASON = "Attention A2A is disabled"
        _ATTENTION_A2A_TIER1_ENABLED = False
        _ATTENTION_A2A_TIER1_REASON = "Attention A2A is disabled"
        _activate_attention_a2a_config(AttentionA2AConfig())
        return
    tier2_enabled, tier2_reason = _tier2_support_status()
    tier1_enabled = False
    tier1_reason = None
    if not tier2_enabled:
        tier1_enabled, tier1_reason = _tier1_support_status()
        if not tier1_enabled:
            raise RuntimeError(
                "Attention A2A is unsupported by this Torch build: "
                f"Tier-2 unavailable ({tier2_reason}); "
                f"Tier-1 unavailable ({tier1_reason}). "
                "Disable quantized Attention A2A with "
                "--attention_a2a none or install a supported Torch build."
            )
    preflight_attention_a2a(config)
    _ATTENTION_A2A_POLICY = config
    _ATTENTION_A2A_TIER2_ENABLED = tier2_enabled
    _ATTENTION_A2A_TIER2_REASON = tier2_reason
    _ATTENTION_A2A_TIER1_ENABLED = tier1_enabled
    _ATTENTION_A2A_TIER1_REASON = tier1_reason
    if tier2_enabled:
        if _INPUT_COLLECTIVE_LIB is None:
            _INPUT_COLLECTIVE_LIB = _register_input_collective()
    elif not dist.is_initialized() or dist.get_rank() == 0:
        logger.warning(
            "[ATTENTION_A2A] async Inductor collective lowering unavailable; using the Tier-1 ordered AITER path (%s)",
            tier2_reason,
        )
    if attention_backend is None:
        if not config.is_auto:
            raise ValueError(f"Attention A2A profile {config.profile} requires backend {config.attention_backend}")
        _activate_attention_a2a_config(AttentionA2AConfig())
        return
    try:
        resolved = config.resolve_for_backend(attention_backend)
    except ValueError:
        # A hybrid schedule installs its first concrete backend immediately
        # before the first denoising step. The initialization backend itself is
        # not part of that schedule.
        if not config.is_auto:
            raise
        _activate_attention_a2a_config(AttentionA2AConfig())
        return
    _activate_attention_a2a_config(resolved)


def shutdown_attention_a2a() -> None:
    """Synchronously release MoRI and all process-local Attention A2A state."""
    global _ATTENTION_A2A_POLICY
    global _MORI_GROUP_KEY
    global _MORI_CPU_GROUP
    global _ATTENTION_A2A_POISONED
    global _ATTENTION_A2A_TIER2_ENABLED
    global _ATTENTION_A2A_TIER2_REASON
    global _ATTENTION_A2A_TIER1_ENABLED
    global _ATTENTION_A2A_TIER1_REASON

    finalize_error = None
    try:
        if torch.cuda.is_available() and (_OP_CACHE or _INPUT_SIDE_STREAMS or _MORI_GROUP_KEY is not None):
            torch.cuda.synchronize()
        if _MORI_GROUP_KEY is not None:
            import mori.shmem as ms

            ms.shmem_finalize()
    except Exception as exc:  # noqa: BLE001 - cleanup still has to release process state
        finalize_error = exc
    finally:
        _INPUT_PENDING.clear()
        _INPUT_COLLECTIVE_WAITS.clear()
        _INPUT_CONSUMER_DONE.clear()
        _INPUT_SIDE_STREAMS.clear()
        _OP_CACHE.clear()
        if _MORI_CPU_GROUP is not None and dist.is_initialized():
            try:
                dist.destroy_process_group(_MORI_CPU_GROUP)
            except Exception as exc:  # noqa: BLE001 - report after state is reset
                if finalize_error is None:
                    finalize_error = exc
        _MORI_CPU_GROUP = None
        _MORI_GROUP_KEY = None
        _ATTENTION_A2A_POISONED = None
        _ATTENTION_A2A_TIER2_ENABLED = False
        _ATTENTION_A2A_TIER2_REASON = "Attention A2A is disabled"
        _ATTENTION_A2A_TIER1_ENABLED = False
        _ATTENTION_A2A_TIER1_REASON = "Attention A2A is disabled"
        _ATTENTION_A2A_POLICY = AttentionA2AConfig()
        _activate_attention_a2a_config(AttentionA2AConfig())
    if finalize_error is not None:
        raise RuntimeError("failed to finalize MORI Attention A2A") from finalize_error


def activate_attention_a2a_backend(attention_backend) -> None:
    """Select the recipe matching the current denoising-step backend."""
    if not _ATTENTION_A2A_POLICY.enabled:
        return
    if _INPUT_PENDING:
        raise RuntimeError("cannot switch Attention A2A recipe while Q/K/V roles are pending")
    resolved = _ATTENTION_A2A_POLICY.resolve_for_backend(attention_backend)
    _activate_attention_a2a_config(resolved)


def _activate_attention_a2a_config(config: AttentionA2AConfig) -> None:
    global _ATTENTION_A2A_CONFIG
    global _ATTENTION_A2A_ENABLED
    global _FUSED_A2A_SIDESTREAM
    global _FUSED_A2A_INTERLEAVE
    global _FUSED_A2A_HADAMARD_PLACEMENT
    global _TRANSPORT_HADAMARD
    global _FUSED_A2A_CODECS
    global _FUSED_A2A_PACKED
    global _FUSED_A2A_V_PACK
    global _FUSED_A2A_COLLECTIVE

    previous = _ATTENTION_A2A_CONFIG
    runtime_enabled = config.enabled and (_ATTENTION_A2A_TIER2_ENABLED or _ATTENTION_A2A_TIER1_ENABLED)
    desired_collective = config.enabled and _ATTENTION_A2A_TIER2_ENABLED
    if config == previous and _ATTENTION_A2A_ENABLED == runtime_enabled and _FUSED_A2A_COLLECTIVE == desired_collective:
        return

    _ATTENTION_A2A_CONFIG = config
    _ATTENTION_A2A_ENABLED = runtime_enabled
    _FUSED_A2A_SIDESTREAM = runtime_enabled
    _FUSED_A2A_INTERLEAVE = runtime_enabled
    effective_hadamard = config.hadamard_placement if runtime_enabled else "none"
    _FUSED_A2A_HADAMARD_PLACEMENT = effective_hadamard
    _TRANSPORT_HADAMARD = effective_hadamard == "transport"
    _FUSED_A2A_CODECS = config.codecs
    _FUSED_A2A_PACKED = runtime_enabled
    _FUSED_A2A_V_PACK = config.v_pack
    _FUSED_A2A_COLLECTIVE = desired_collective
    if config.enabled and dist.is_initialized() and dist.get_rank() == 0:
        logger.info(
            "[ATTENTION_A2A_PROFILE] %s->%s hadamard=%s path=%s",
            previous.profile,
            config.profile,
            effective_hadamard,
            "tier2" if desired_collective else "tier1",
        )


def _input_side_stream(device):
    if not (_FUSED_A2A_SIDESTREAM and (_FUSED_A2A_PACKED or use_fused_a2a_interleave())):
        return None
    if device not in _INPUT_SIDE_STREAMS:
        _INPUT_SIDE_STREAMS[device] = torch.cuda.Stream(device=device)
        logger.debug(
            "Attention A2A sidestream rank=%s device=%s compute=%s side=%s hadamard=%s codecs=%s path=%s",
            dist.get_rank(),
            device,
            torch.cuda.current_stream(device).cuda_stream,
            _INPUT_SIDE_STREAMS[device].cuda_stream,
            _FUSED_A2A_HADAMARD_PLACEMENT,
            _FUSED_A2A_CODECS,
            "tier2" if _FUSED_A2A_COLLECTIVE else "tier1",
        )
    return _INPUT_SIDE_STREAMS[device]


@torch.library.custom_op(
    "xfuser::fused_a2a_consumer_done",
    mutates_args=(),
    **_CUSTOM_OP_OPTIONS,
)
def fused_a2a_input_consumer_done(consumed: torch.Tensor) -> None:
    if not (_FUSED_A2A_SIDESTREAM and (_FUSED_A2A_PACKED or use_fused_a2a_interleave())):
        return
    device = consumed.device
    done = torch.cuda.Event()
    done.record(torch.cuda.current_stream(device))
    _INPUT_CONSUMER_DONE[device] = done


@fused_a2a_input_consumer_done.register_fake
def _fused_a2a_consumer_done_fake(consumed):
    return None


_register_ordered_effect(torch.ops.xfuser.fused_a2a_consumer_done.default)


def get_fused_a2a_mode():
    """Return 1 for Attention A2A input transport, otherwise 0."""
    return int(_ATTENTION_A2A_ENABLED)


def get_fused_a2a_hadamard_placement():
    """Return where Q/K Hadamard is applied for the current process."""
    return _FUSED_A2A_HADAMARD_PLACEMENT


def use_fused_a2a_packed():
    return _FUSED_A2A_PACKED


def use_fused_a2a_collective():
    return _FUSED_A2A_COLLECTIVE


def get_attention_a2a_execution_path():
    if not _ATTENTION_A2A_CONFIG.enabled:
        return "disabled"
    return "tier2" if _FUSED_A2A_COLLECTIVE else "tier1"


def get_attention_a2a_tier2_reason():
    return _ATTENTION_A2A_TIER2_REASON


def get_attention_a2a_tier1_reason():
    return _ATTENTION_A2A_TIER1_REASON


def get_fused_a2a_codecs():
    return _FUSED_A2A_CODECS


def get_fused_a2a_profile():
    return _ATTENTION_A2A_CONFIG.profile


def get_fused_a2a_v_pack():
    return _FUSED_A2A_V_PACK


def launch_attention_a2a_packed(
    query,
    key,
    value,
    query_scale,
    key_scale,
    value_scale,
    profile,
    softmax_scale,
):
    """Call the plain packed MHA launcher resolved during A2A preflight."""
    launcher = _ATTENTION_A2A_PACKED_LAUNCHER
    if launcher is None:
        raise RuntimeError("Attention A2A packed MHA launcher was not initialized")
    return launcher(
        query,
        key,
        value,
        query_scale,
        key_scale,
        value_scale,
        profile,
        softmax_scale,
    )


def _require_active_profile(profile):
    if _ATTENTION_A2A_POISONED is not None:
        raise RuntimeError(
            f"Attention A2A runtime is poisoned after an earlier submission failure: {_ATTENTION_A2A_POISONED}"
        )
    if profile != _ATTENTION_A2A_CONFIG.profile:
        raise RuntimeError(
            f"compiled Attention A2A profile {profile!r} does not match active "
            f"profile {_ATTENTION_A2A_CONFIG.profile!r}"
        )


def use_fused_a2a_interleave():
    return _FUSED_A2A_INTERLEAVE and _FUSED_A2A_SIDESTREAM and _FUSED_A2A_PACKED


def fused_a2a_input_role(input, role, group, rank, pending=None):
    """Submit one role, retaining only a traceable handle outside the opaque op."""
    if _FUSED_A2A_COLLECTIVE:
        previous = [] if pending is None else list(pending)
        outputs = torch.ops.xfuser.fused_a2a_input_collective.default(
            input,
            input,
            previous,
            role,
            _ATTENTION_A2A_CONFIG.profile,
            group.group_name,
            rank,
            dist.get_world_size(group),
        )
        return (*previous, *outputs)
    handle = (group.group_name, rank, input.device.index)
    if role not in (0, 1, 2) or (role != 0 and pending != (*handle, role)):
        raise ValueError("interleave requires Q, K, V in order")
    _fused_a2a_submit_role(input, role, _ATTENTION_A2A_CONFIG.profile, group.group_name, rank)
    return (*handle, role + 1)


@torch.library.custom_op(
    "xfuser::fused_a2a_submit_role",
    mutates_args=(),
    **_CUSTOM_OP_OPTIONS,
)
def _fused_a2a_submit_role(
    input: torch.Tensor,
    role: int,
    profile: str,
    group_name: str,
    rank: int,
) -> None:
    # The ordered effect owns peer buffers, handshake state and host parity. CUDA
    # graph replay would bypass those host updates and the runtime pending bridge.
    _require_active_profile(profile)
    group = dist.distributed_c10d._resolve_process_group(group_name)
    handle = (group_name, rank, input.device.index)
    pending = _INPUT_PENDING.get(handle)
    if role == 0 and pending is not None:
        raise ValueError("previous interleaved input has not been consumed")
    try:
        _INPUT_PENDING[handle] = _submit_input_role(input, role, group, rank, pending)
    except Exception as exc:
        global _ATTENTION_A2A_POISONED
        _INPUT_PENDING.pop(handle, None)
        _ATTENTION_A2A_POISONED = f"{type(exc).__name__}: {exc}"
        raise


@_fused_a2a_submit_role.register_fake
def _fused_a2a_submit_role_fake(input, role, profile, group_name, rank):
    return None


_register_ordered_effect(torch.ops.xfuser.fused_a2a_submit_role.default)
# PyTorch 2.9 FX DCE must retain both the original no-return node and AOT's
# token wrapper; otherwise Inductor silently removes the ordered submissions.
if _tier1_support_status()[0]:
    has_side_effect(with_effects)


def _submit_input_role(input, role, group, rank, pending=None):
    """Submit one already-normalized sequence-major role without a compute join."""
    if not use_fused_a2a_interleave():
        raise RuntimeError("per-role input requires quantized sidestream interleave")
    if role == 0:
        in_op = _get_ops(group, rank, tuple(input.shape), input.dtype, input.device)
        side = _input_side_stream(input.device)
        consumer_done = _INPUT_CONSUMER_DONE.get(input.device)
        if consumer_done is not None:
            side.wait_event(consumer_done)
        pending = {
            "op": in_op,
            "side": side,
            "inputs": [],
            "results": [],
            "next_role": 0,
        }
    if pending is None or pending["next_role"] != role:
        raise ValueError("interleave requires Q, K, V in order")
    side = pending["side"]
    producer_done = torch.cuda.Event()
    producer_done.record(torch.cuda.current_stream(input.device))
    side.wait_event(producer_done)
    # Raw-pointer launchers do not inform the caching allocator about side reads.
    input.record_stream(side)
    pending["inputs"].append(input)
    result = pending["op"].submit_role(role, input, stream=side)
    if (
        result is None
        or not torch.is_tensor(getattr(result, "payload", None))
        or not torch.is_tensor(getattr(result, "scale", None))
    ):
        raise RuntimeError("AITER submit_role must return PackedRoleResult(payload, scale) for every packed Q/K/V role")
    pending["results"].append(result)
    pending["next_role"] += 1
    return pending


def fused_a2a_pad_multiple(world_size):
    if _FUSED_A2A_PACKED:
        return world_size * _ATTENTION_A2A_CONFIG.local_sequence_multiple
    return world_size


def _group_ranks(group):
    if hasattr(dist, "get_process_group_ranks"):
        return tuple(dist.get_process_group_ranks(group))
    return tuple(dist.get_global_rank(group, rank) for rank in range(dist.get_world_size(group)))


def _init_mori(group, ranks):
    global _MORI_GROUP_KEY, _MORI_CPU_GROUP

    group_key = (id(group), ranks)
    if _MORI_GROUP_KEY == group_key:
        return
    if _MORI_GROUP_KEY is not None:
        raise RuntimeError("Attention A2A supports one live MoRI Ulysses group per process")

    import mori.shmem as ms

    # Only this Ulysses group's members participate. This is intentionally not the
    # sequence-parallel CPU group, whose membership differs when ring degree > 1.
    # Members enter this lazy initialization together on their first fused call;
    # non-members need not participate in subgroup creation.
    _MORI_CPU_GROUP = dist.new_group(ranks=list(ranks), backend="gloo", use_local_synchronization=True)
    torch._C._distributed_c10d._register_process_group("mori", _MORI_CPU_GROUP)
    ms.shmem_torch_process_group_init("mori")
    _MORI_GROUP_KEY = group_key


def _get_ops(group, rank, shape, dtype, device, softmax_scale=None):
    from aiter.ops.flydsl.attention_a2a_intranode import (
        AttentionA2AIntraNodeOp,
    )
    from aiter.ops.mha_v4 import AttentionPack

    ranks = _group_ranks(group)
    group_key = (id(group), ranks)
    device_key = (device.type, device.index)
    b, s_local, h_total, d = shape
    if softmax_scale is None:
        softmax_scale = d**-0.5
    v_pack = AttentionPack.V_FOR_FP6_P if _FUSED_A2A_V_PACK == "fp6_p" else AttentionPack.DEFAULT
    key = (
        group_key,
        rank,
        device_key,
        dtype,
        b,
        s_local,
        h_total,
        d,
        softmax_scale,
        _FUSED_A2A_CODECS,
        v_pack,
        _TRANSPORT_HADAMARD,
    )
    op = _OP_CACHE.get(key)
    if op is None:
        if len(_OP_CACHE) >= _MAX_OP_CACHE_ENTRIES:
            raise RuntimeError(
                "Attention A2A symmetric-buffer cache limit reached; "
                "shutdown and reinitialize before using another shape/profile"
            )
        _init_mori(group, ranks)
        op = AttentionA2AIntraNodeOp(
            rank=rank,
            world_size=len(ranks),
            shape=shape,
            quant=(_FUSED_A2A_CODECS[0], _FUSED_A2A_CODECS[2]),
            return_packed=True,
            v_pack=v_pack,
            softmax_scale=softmax_scale,
            hadamard=_TRANSPORT_HADAMARD,
        )
        _OP_CACHE[key] = op
    return op


def _packed_output_views(outputs, scales, output_shape, codecs=None):
    """Rebuild the role-specific logical MHA-v4 views over A2A raw buffers."""
    from aiter.ops.mha_v4 import (
        mxfp4_k_view,
        mxfp4_v_view,
        mxfp6_k_view,
    )

    b, sequence, heads, d = output_shape
    q_raw, k_raw, v_raw = outputs
    q_scales, k_scales, v_scales = scales
    qk_codec, _, v_codec = codecs or _FUSED_A2A_CODECS
    scale_shape = (*output_shape[:-1], d // 32)
    tiles = (sequence + 127) // 128

    if qk_codec == "bf16":
        q = q_raw.view(b, heads, sequence, d).transpose(1, 2)
        k = k_raw.view(b, heads, sequence, d).transpose(1, 2)
    elif qk_codec in ("int8", "e4m3", "mxfp8"):
        q = q_raw.view(output_shape)
        k = k_raw.view(output_shape)
        if qk_codec not in ("int8", "e4m3"):
            q_scales = q_scales.view(scale_shape)
            k_scales = k_scales.view(scale_shape)
    elif qk_codec == "mxfp4":
        q_scales = q_scales.view(scale_shape)
        k_scales = k_scales.view(*output_shape[:-1], d // 32)
        q = q_raw.view(*output_shape[:-1], d // 2)
        k = mxfp4_k_view(k_raw, k_scales)
    elif qk_codec == "mxfp6":
        q_scales = q_scales.view(scale_shape)
        q = q_raw.view(*output_shape[:-1], d // 32 * 24)
        k, k_scales = mxfp6_k_view(k_raw, k_scales, b, sequence, heads)
    else:
        raise RuntimeError(f"unsupported packed Q/K codec {qk_codec!r}")

    if v_codec == "e4m3":
        v = v_raw.view(output_shape)
    elif v_codec == "e4m3_pc":
        v = v_raw.view(output_shape)
        v_scales = v_scales.view(b, heads, d)
    elif v_codec == "mxfp4":
        v_scales = v_scales.view(b, heads, tiles * 512)
        v = mxfp4_v_view(v_raw, v_scales, sequence)
    elif v_codec == "mxfp6_p":
        v_scales = v_scales.view(b, heads, tiles * 512)
        v = torch.as_strided(
            v_raw,
            (b, sequence, heads, d),
            (heads * tiles * 12288, 96, tiles * 12288, 1),
        )
    else:
        raise RuntimeError(f"unsupported packed V codec {v_codec!r}")

    return (
        (q, k, v),
        (q_scales, k_scales, v_scales),
    )


def _collect_packed_role_results(results):
    """Collect public per-role AITER results in Q/K/V order."""
    if len(results) != 3:
        raise ValueError(f"expected packed results for Q, K, and V; got {len(results)}")
    return (
        tuple(result.payload for result in results),
        tuple(result.scale for result in results),
    )


def _fused_a2a_input_runtime(
    query,
    key,
    value,
    group,
    rank,
    norm_q=None,
    norm_k=None,
    cos=None,
    sin=None,
    softmax_scale=None,
    pending=None,
):
    """Run the fused in-hop from USP head-major views."""
    sequence_major = tuple(tensor.transpose(1, 2) for tensor in (query, key, value))
    if not all(tensor.is_contiguous() for tensor in sequence_major):
        raise ValueError("fused A2A requires Q/K/V backed by contiguous [B,S_local,H,D] tensors")

    q, k, v = sequence_major
    in_op = _get_ops(group, rank, tuple(q.shape), q.dtype, q.device, softmax_scale)
    side_stream = _input_side_stream(q.device)
    if pending is not None:
        if pending != (group.group_name, rank, q.device.index, 3):
            raise ValueError("interleaved input handle must finish this group's Q/K/V trio")
        pending = _INPUT_PENDING.pop(pending[:3])
        if pending["op"] is not in_op or pending["next_role"] != 3:
            raise ValueError("interleaved input must finish the same op's Q/K/V trio")
        transport_done = torch.cuda.Event()
        transport_done.record(pending["side"])
        torch.cuda.current_stream(q.device).wait_event(transport_done)
        outputs = _collect_packed_role_results(pending["results"])
    elif side_stream is None:
        results = tuple(in_op.submit_role(role, tensor) for role, tensor in enumerate((q, k, v)))
        outputs = _collect_packed_role_results(results)
    else:
        compute_stream = torch.cuda.current_stream(q.device)
        consumer_done = _INPUT_CONSUMER_DONE.get(q.device)
        if consumer_done is not None:
            # Conservatively drain the previous consumer, not just the reused parity.
            side_stream.wait_event(consumer_done)
        producer_done = torch.cuda.Event()
        # Include lazy op initialization as well as Q/K/V and norm/RoPE production.
        producer_done.record(compute_stream)
        side_stream.wait_event(producer_done)
        for tensor in (q, k, v):
            tensor.record_stream(side_stream)
        results = tuple(in_op.submit_role(role, tensor, stream=side_stream) for role, tensor in enumerate((q, k, v)))
        outputs = _collect_packed_role_results(results)
        transport_done = torch.cuda.Event()
        transport_done.record(side_stream)
        compute_stream.wait_event(transport_done)
    b, s_local, h_total, d = q.shape
    world_size = dist.get_world_size(group)
    if _FUSED_A2A_PACKED:
        outputs, scales = outputs
        output_shape = (b, world_size * s_local, h_total // world_size, d)
        return _packed_output_views(outputs, scales, output_shape)
    output_shape = (b, h_total // world_size, world_size * s_local, d)
    return tuple(output.view(output_shape) for output in outputs)


def _owned_transport_tensor(tensor):
    # Custom-op results must own storage: cached symmetric buffers alias across
    # epochs, and tiled packed K/V consumers also read their backing padding.
    size = tensor.untyped_storage().nbytes() // tensor.element_size()
    storage = tensor.as_strided((size,), (1,), 0).clone()
    return storage.as_strided(tensor.shape, tensor.stride(), tensor.storage_offset())


def fused_a2a_input(
    query,
    key,
    value,
    group,
    rank,
    norm_q=None,
    norm_k=None,
    cos=None,
    sin=None,
    softmax_scale=None,
    pending=None,
):
    if _FUSED_A2A_COLLECTIVE and pending is not None:
        if len(pending) != 6:
            raise ValueError("Tier-2 input requires all three role outputs")
        outputs = [torch.ops.xfuser.fused_a2a_input_wait.default(tensor) for tensor in pending]
        b, h, s, d = query.shape
        output_shape = (
            b,
            s * dist.get_world_size(group),
            h // dist.get_world_size(group),
            d,
        )
        return _packed_output_views(
            (outputs[0], outputs[2], outputs[4]),
            (outputs[1], outputs[3], outputs[5]),
            output_shape,
            codecs=_ATTENTION_A2A_CONFIG.codecs,
        )
    if pending is not None and pending != (group.group_name, rank, query.device.index, 3):
        raise ValueError("interleaved input handle must finish this group's Q/K/V trio")
    outputs = _fused_a2a_wait(
        query,
        key,
        value,
        _ATTENTION_A2A_CONFIG.profile,
        group.group_name,
        rank,
        dist.get_world_size(group),
        norm_q,
        norm_k,
        cos,
        sin,
        softmax_scale,
        pending is not None,
    )
    if _FUSED_A2A_PACKED:
        return tuple(outputs[:3]), tuple(outputs[3:])
    return tuple(outputs)


@torch.library.custom_op(
    "xfuser::fused_a2a_wait",
    mutates_args=(),
    **_CUSTOM_OP_OPTIONS,
)
def _fused_a2a_wait(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    profile: str,
    group_name: str,
    rank: int,
    world_size: int,
    norm_q: torch.Tensor | None,
    norm_k: torch.Tensor | None,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    softmax_scale: float | None,
    interleaved: bool,
) -> list[torch.Tensor]:
    _require_active_profile(profile)
    group = dist.distributed_c10d._resolve_process_group(group_name)
    pending = (group_name, rank, query.device.index, 3) if interleaved else None
    outputs = _fused_a2a_input_runtime(
        query,
        key,
        value,
        group,
        rank,
        norm_q,
        norm_k,
        cos,
        sin,
        softmax_scale,
        pending,
    )
    if _FUSED_A2A_PACKED:
        outputs = (*outputs[0], *outputs[1])
    return [_owned_transport_tensor(tensor) for tensor in outputs]


def _fake_packed_raw_outputs(query, profile, world_size):
    config = AttentionA2AConfig(profile=profile)
    codecs = config.codecs
    b, h, s, d = query.shape
    shape = (b, s * world_size, h // world_size, d)
    from aiter.ops.mha_v4_quant import (
        mxfp4_k_raw_buffer_size,
        mxfp4_v_raw_buffer_size,
        mxfp6_v_raw_buffer_size,
    )
    from aiter.ops.triton.quant.mxfp6_fmha_pack import (
        fp6_k_raw_buffer_sizes,
    )

    sequence = shape[1]
    heads = shape[2]
    numel = b * sequence * heads * d
    qk_codec, _, v_codec = codecs

    if qk_codec in ("int8", "e4m3", "mxfp8"):
        q_raw = query.new_empty((numel,), dtype=torch.uint8)
        k_raw = query.new_empty((numel,), dtype=torch.uint8)
        if qk_codec in ("int8", "e4m3"):
            q_scales = query.new_empty((1,), dtype=torch.float32)
            k_scales = query.new_empty((1,), dtype=torch.float32)
        else:
            q_scales = query.new_empty((numel // 32,), dtype=torch.uint8)
            k_scales = query.new_empty((numel // 32,), dtype=torch.uint8)
    elif qk_codec in ("mxfp4", "mxfp6"):
        q_scales = query.new_empty((numel // 32,), dtype=torch.uint8)
        if qk_codec == "mxfp4":
            q_raw = query.new_empty((numel // 2,), dtype=torch.uint8)
            k_raw = query.new_empty(
                (mxfp4_k_raw_buffer_size(b, sequence, heads),),
                dtype=torch.uint8,
            )
            k_scales = query.new_empty((numel // 32,), dtype=torch.uint8)
        else:
            q_raw = query.new_empty((numel * 3 // 4,), dtype=torch.uint8)
            k_size, k_scale_size = fp6_k_raw_buffer_sizes(b, sequence, heads)
            k_raw = query.new_empty((k_size,), dtype=torch.uint8)
            k_scales = query.new_empty((k_scale_size,), dtype=torch.uint8)
    else:
        raise RuntimeError(f"unsupported packed Q/K codec {qk_codec!r}")

    if v_codec == "e4m3":
        v_raw = query.new_empty((numel,), dtype=torch.uint8)
        v_scales = query.new_empty((1,), dtype=torch.float32)
    elif v_codec == "e4m3_pc":
        v_raw = query.new_empty((numel,), dtype=torch.uint8)
        v_scales = query.new_empty((b, heads, d), dtype=torch.float32)
    elif v_codec in ("mxfp4", "mxfp6_p"):
        v_raw_size = (
            mxfp4_v_raw_buffer_size(b, sequence, heads)
            if v_codec == "mxfp4"
            else mxfp6_v_raw_buffer_size(b, sequence, heads)
        )
        v_raw = query.new_empty(
            (v_raw_size,),
            dtype=torch.uint8,
        )
        v_scales = query.new_empty(
            (b * heads * ((sequence + 127) // 128) * 512,),
            dtype=torch.uint8,
        )
    else:
        raise RuntimeError(f"unsupported packed V codec {v_codec!r}")
    return (q_raw, k_raw, v_raw), (q_scales, k_scales, v_scales)


@_fused_a2a_wait.register_fake
def _fused_a2a_wait_fake(
    query,
    key,
    value,
    profile,
    group_name,
    rank,
    world_size,
    norm_q,
    norm_k,
    cos,
    sin,
    softmax_scale,
    interleaved,
):
    del key, value, group_name, rank, norm_q, norm_k, cos, sin
    del softmax_scale, interleaved
    config = AttentionA2AConfig(profile=profile)
    b, h, s, d = query.shape
    if not config.enabled:
        return [query.new_empty((b, h // world_size, s * world_size, d)) for _ in range(3)]
    raw_payloads, raw_scales = _fake_packed_raw_outputs(query, profile, world_size)
    shape = (b, s * world_size, h // world_size, d)
    payloads, scales = _packed_output_views(
        raw_payloads,
        raw_scales,
        shape,
        codecs=config.codecs,
    )
    return [*payloads, *scales]


_register_ordered_effect(torch.ops.xfuser.fused_a2a_wait.default)


# Cached outputs are persistent symmetric buffers. A same-key launch is safe only
# after the prior attention consumer has finished reading them on its stream.

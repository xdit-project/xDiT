from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Tuple

import torch


_RUNTIME_FIELDS = (
    "patch_mode",
    "pipeline_patch_idx",
    "pp_patches_height",
    "pp_patches_start_idx_local",
    "pp_patches_start_end_idx_global",
    "pp_patches_token_num",
    "pp_patches_token_start_idx_local",
    "pp_patches_token_start_end_idx_global",
    "_xdit_pipefusion_layout_fingerprint",
    "attention_backend",
    "cross_attention_backend",
    "use_high_precision_gemm",
    "step_counter",
)
_ATTENTION_BYPASS_FIELD = "_xdit_compile_capture_bypass_attention"


def _map_tree(value: Any, fn) -> Any:
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_map_tree(item, fn) for item in value)
    if isinstance(value, list):
        return [_map_tree(item, fn) for item in value]
    if isinstance(value, dict):
        return {key: _map_tree(item, fn) for key, item in value.items()}
    return value


def _tree_signature(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return (
            "tensor",
            tuple(value.shape),
            tuple(value.stride()),
            value.dtype,
            value.layout,
        )
    if isinstance(value, tuple):
        return ("tuple", tuple(_tree_signature(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_tree_signature(item) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple((key, _tree_signature(item)) for key, item in sorted(value.items())),
        )
    try:
        hash(value)
        return ("value", value)
    except TypeError:
        return ("type", type(value))


def _dispatch_signature(runtime: "PipeFusionRuntimeSnapshot") -> Any:
    values = runtime.values

    def scalar(value):
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return value.item()
        try:
            hash(value)
            return value
        except TypeError:
            return repr(value)

    return tuple(
        (name, scalar(values.get(name)))
        for name in (
            "attention_backend",
            "cross_attention_backend",
            "use_high_precision_gemm",
            "step_counter",
        )
    )


@dataclass(frozen=True)
class PipeFusionRuntimeSnapshot:
    values: Mapping[str, Any]

    @classmethod
    def capture(cls, state) -> "PipeFusionRuntimeSnapshot":
        return cls({name: copy.deepcopy(getattr(state, name)) for name in _RUNTIME_FIELDS if hasattr(state, name)})

    @contextmanager
    def install(self, state) -> Iterator[None]:
        missing = object()
        previous = {}
        for name in self.values:
            value = getattr(state, name, missing)
            previous[name] = missing if value is missing else copy.deepcopy(value)
        try:
            for name, value in self.values.items():
                setattr(state, name, copy.deepcopy(value))
            yield
        finally:
            for name, value in previous.items():
                if value is missing:
                    if hasattr(state, name):
                        delattr(state, name)
                else:
                    setattr(state, name, value)


@dataclass
class PipeFusionCapturedCall:
    component_name: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    runtime: PipeFusionRuntimeSnapshot

    @classmethod
    def create(
        cls,
        component_name: str,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        runtime: PipeFusionRuntimeSnapshot,
    ) -> "PipeFusionCapturedCall":
        def offload(tensor: torch.Tensor):
            return tensor.detach().to("cpu", copy=True)

        return cls(
            component_name=component_name,
            args=_map_tree(args, offload),
            kwargs=_map_tree(kwargs, offload),
            runtime=runtime,
        )

    def signature(self) -> Any:
        return self.signature_for(
            self.component_name,
            self.args,
            self.kwargs,
            self.runtime,
        )

    @staticmethod
    def signature_for(
        component_name: str,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        runtime: PipeFusionRuntimeSnapshot,
    ) -> Any:
        return (
            component_name,
            runtime.values.get("patch_mode", False),
            runtime.values.get("pipeline_patch_idx", 0),
            _dispatch_signature(runtime),
            _tree_signature(args),
            _tree_signature(kwargs),
        )

    def restore_inputs(self, device: torch.device) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        def restore(tensor: torch.Tensor):
            return tensor.to(device=device, non_blocking=False)

        return _map_tree(self.args, restore), _map_tree(self.kwargs, restore)


class PipeFusionCompileCapture:
    """Capture real stage calls eagerly and replay them without pipeline P2P."""

    def __init__(self, components: Mapping[str, torch.nn.Module], runtime_state):
        self.components = dict(components)
        self.runtime_state = runtime_state
        self.calls: list[PipeFusionCapturedCall] = []
        self._signatures = set()
        self._handles = []
        self._original_forwards = {}

    def _capture(self, component_name: str, args, kwargs) -> None:
        runtime = PipeFusionRuntimeSnapshot.capture(self.runtime_state)
        signature = PipeFusionCapturedCall.signature_for(
            component_name,
            tuple(args),
            dict(kwargs),
            runtime,
        )
        if signature in self._signatures:
            return
        call = PipeFusionCapturedCall.create(
            component_name,
            tuple(args),
            dict(kwargs),
            runtime,
        )
        self._signatures.add(signature)
        self.calls.append(call)

    @contextmanager
    def hooks(self) -> Iterator["PipeFusionCompileCapture"]:
        missing = object()
        previous_bypass = getattr(self.runtime_state, _ATTENTION_BYPASS_FIELD, missing)
        try:
            # The eager pass exists only to discover stage-call signatures.
            # Running a real attention kernel here can trigger shape-specific
            # JIT/autotuning serially as the pipeline advances rank by rank.
            # A model may provide a shape-only stage forward so P2P propagates
            # representative tensors without executing any blocks. The
            # attention bypass remains as the generic fallback for models that
            # do not yet provide that hook.
            setattr(self.runtime_state, _ATTENTION_BYPASS_FIELD, True)
            for name, component in self.components.items():
                capture_forward = getattr(
                    component,
                    "pipefusion_compile_capture_forward",
                    None,
                )
                if capture_forward is not None:
                    self._original_forwards[name] = component.forward
                    component.forward = capture_forward
                handle = component.register_forward_pre_hook(
                    lambda _module, args, kwargs, name=name: self._capture(name, args, kwargs),
                    with_kwargs=True,
                )
                self._handles.append(handle)
            yield self
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            for name, forward in self._original_forwards.items():
                self.components[name].forward = forward
            self._original_forwards.clear()
            if previous_bypass is missing:
                delattr(self.runtime_state, _ATTENTION_BYPASS_FIELD)
            else:
                setattr(self.runtime_state, _ATTENTION_BYPASS_FIELD, previous_bypass)

    def replay(self, device: torch.device) -> None:
        with torch.inference_mode():
            for call in self.calls:
                component = self.components[call.component_name]
                args, kwargs = call.restore_inputs(device)
                with call.runtime.install(self.runtime_state):
                    component(*args, **kwargs)

    def clear(self) -> None:
        self.calls.clear()
        self._signatures.clear()


def pipefusion_recompile_limit(pp_degree: int, num_patches: Optional[int], local_blocks: int) -> int:
    """Return the Dynamo recompile limit one PipeFusion rank's compiled graphs need.

    Dynamo keeps compiled graphs in a cache attached to the Python code object it traces, not
    to a module instance. ``torch._dynamo.config.recompile_limit`` (8 by default) caps how many
    entries that cache may hold for one frame. Once the cap is reached, a call whose guards
    match no entry is not compiled and runs eagerly. Each block in a declared block list gets
    its own ``torch.compile`` wrapper, but blocks of one class execute the same code, so their
    specializations all count against one limit.

    Outside PipeFusion each block sees one input shape, and the default limit is enough. A
    PipeFusion stage calls each local block with the full latent during the synchronous warmup
    steps, then once per patch. Patch heights can differ, and the KV cache reads the patch mode
    and patch index from runtime state, so Dynamo specializes every patch separately. One rank
    can therefore need an entry per local block for the full latent and for each patch, plus a
    small margin:

        max(2 * num_patches + 4, local_blocks * (num_patches + 1) + 4)

    ``local_blocks`` counts the declared blocks of the compiled components on this rank. The
    first term is the floor for a stage with no declared block list, where only the bound stage
    ``forward`` is compiled. An unset patch count falls back to one patch per stage, as
    ``PipeFusionParallelConfig`` does.
    """
    num_patches = num_patches or pp_degree
    return max(2 * num_patches + 4, max(local_blocks, 1) * (num_patches + 1) + 4)


def record_pipefusion_stage_calls(
    stages: Mapping[str, torch.nn.Module],
    runtime_state,
    run_pipe: Callable[[int], None],
) -> PipeFusionCompileCapture:
    """Record each stage's eager calls from one short pipeline pass, before any wrapping.

    Replay needs each call signature once: a full-latent call from a synchronous step, then
    the per-patch calls from an asynchronous step. More synchronous steps would only repeat
    the full-latent call, so ``runtime_state.runtime_config.warmup_steps`` is capped at one for
    the pass and restored afterwards. ``run_pipe`` receives the step count to run.
    """
    runtime_config = runtime_state.runtime_config
    configured_warmup_steps = runtime_config.warmup_steps
    capture_warmup_steps = min(configured_warmup_steps, 1)
    capture = PipeFusionCompileCapture(stages, runtime_state)
    runtime_config.warmup_steps = capture_warmup_steps
    try:
        with capture.hooks():
            run_pipe(max(2, capture_warmup_steps + 1))
    finally:
        runtime_config.warmup_steps = configured_warmup_steps
    return capture


def replay_pipefusion_stage_calls(capture: PipeFusionCompileCapture, device: torch.device, replica_group) -> None:
    """Replay the recorded calls on every rank of the replica, then agree on the result.

    Replay is rank-local, so one rank can fail while the others succeed. Every rank gathers
    the status before continuing, and all of them raise if any failed. The recorded calls are
    dropped either way.
    """
    replica_group.barrier()
    local_error = None
    try:
        capture.replay(device)
        torch.cuda.synchronize()
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    finally:
        capture.clear()
    failures = replica_group.all_gather_object(local_error)
    if any(failure is not None for failure in failures):
        details = "; ".join(f"rank {rank}: {failure}" for rank, failure in enumerate(failures) if failure is not None)
        raise RuntimeError(f"PipeFusion rank-local compile replay failed collectively: {details}")

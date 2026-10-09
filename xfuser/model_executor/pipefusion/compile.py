from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Tuple

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
        return cls(
            {
                name: copy.deepcopy(getattr(state, name))
                for name in _RUNTIME_FIELDS
                if hasattr(state, name)
            }
        )

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

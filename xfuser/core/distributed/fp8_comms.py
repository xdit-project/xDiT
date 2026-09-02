"""FP8 Ulysses all-to-all communication.

Encapsulates the whole fp8-comms feature -- per-layer calibration state, the
calibration lifecycle, the per-step observation hooks, and the quantized
all-to-all helpers -- so that runtime_state / base_model / usp / transformer_wan
only make thin calls into it. Modelled on VAEManager (runner_models/vae_manager.py).

This module is a leaf: it imports nothing from usp / attention_backend /
runtime_state / base_model at module scope. Those are imported lazily inside the
methods that need them (once, at startup, or via the already-established lazy
pattern in the compiled attention path), which keeps the import graph acyclic.
"""
import copy
import os
from typing import NamedTuple, Optional

import torch
import torch.distributed as dist

import xfuser.envs as envs
from xfuser.logger import init_logger
from xfuser.config.config import DEFAULT_FP8_COMMS_SAFETY_FACTOR

if torch.cuda.is_available() or envs._is_npu():
    from yunchang.globals import PROCESS_GROUP
else:
    PROCESS_GROUP = None

logger = init_logger(__name__)

_FP8_LOG_SCALES = bool(os.environ.get("XFUSER_FP8_LOG_SCALES"))
_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
)


class Fp8CommsCall(NamedTuple):
    """The per-layer fp8-comms scales for one attention call.

    A single opaque argument threaded through USP so the attention signature does
    not need a separate parameter per scale. ``None`` at a call site means fp8 comms
    is off for that call. A NamedTuple (not a plain object) so torch.compile traces
    the field access without a graph break.
    """

    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    o_scale: torch.Tensor


class Fp8CommsModelState:
    """Per-transformer FP8 comms calibration state (one entry per self-attn layer)."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.q_running_max = torch.zeros(num_layers, dtype=torch.float32)
        self.k_running_max = torch.zeros(num_layers, dtype=torch.float32)
        self.v_running_max = torch.zeros(num_layers, dtype=torch.float32)
        self.o_running_max = torch.zeros(num_layers, dtype=torch.float32)
        self.synced = False

    def to_device_(self, device: torch.device):
        self.q_running_max = self.q_running_max.to(device)
        self.k_running_max = self.k_running_max.to(device)
        self.v_running_max = self.v_running_max.to(device)
        self.o_running_max = self.o_running_max.to(device)


class Fp8CommsState:
    """Holds all state for FP8 Ulysses all-to-all communication.

    Per-layer scales live on each attn1 module as compile-friendly buffers; this class
    holds per-model running amaxes during calibration only.
    """

    def __init__(
        self,
        fixed_scale: Optional[float] = None,
        safety_factor: float = DEFAULT_FP8_COMMS_SAFETY_FACTOR,
    ):
        self.fixed_scale = fixed_scale
        # Calibrated per-layer scale = amax / (FP8_MAX * safety_factor). A smaller
        # safety_factor enlarges the scale, so the calibrated amax maps further below
        # FP8_MAX and live values above the calibrated peak no longer clip.
        self.safety_factor = safety_factor
        self._models: dict[int, Fp8CommsModelState] = {}
        self.calibrated_model_ids: set = set()

    @classmethod
    def from_config(cls, config) -> "Optional[Fp8CommsState]":
        """Build the state from an EngineConfig, or None when fp8-comms is off / inapplicable."""
        runtime_config = config.runtime_config
        if not runtime_config.use_fp8_comms:
            return None
        ulysses_degree = config.parallel_config.sp_config.ulysses_degree or 1
        if ulysses_degree <= 1:
            logger.warning(
                "--use_fp8_comms is set but ulysses_degree <= 1. "
                "FP8 communication will not be applied."
            )
            return None
        scale = runtime_config.fp8_comms_scale
        safety_factor = runtime_config.fp8_comms_safety_factor
        if scale is not None:
            logger.warning(f"FP8 communication enabled with fixed scale {scale}.")
        else:
            logger.warning(
                "FP8 communication enabled with static per-layer scaling "
                f"(calibrated once before inference; safety_factor={safety_factor})."
            )
        return cls(fixed_scale=scale, safety_factor=safety_factor)

    # ---- per-model registration / state ------------------------------------

    def register_model(self, model, num_layers: int) -> None:
        """Register a transformer for per-layer FP8 comms calibration."""
        model_id = id(model)
        if model_id in self._models:
            return
        self._models[model_id] = Fp8CommsModelState(num_layers)
        if self.fixed_scale is not None:
            self.apply_fixed_scales_to_model(model)
            self._models[model_id].synced = True
            self.calibrated_model_ids.add(model_id)

    def get_model_state(self, model) -> Optional[Fp8CommsModelState]:
        return self._models.get(id(model))

    def apply_fixed_scales_to_model(self, model) -> None:
        """Broadcast a fixed scale to all self-attention layer buffers."""
        scale = float(self.fixed_scale)
        for block in model.blocks:
            block.attn1.fp8_q_scale.fill_(scale)
            block.attn1.fp8_k_scale.fill_(scale)
            block.attn1.fp8_v_scale.fill_(scale)
            block.attn1.fp8_o_scale.fill_(scale)

    def to_device_(self, device: torch.device):
        for model_state in self._models.values():
            model_state.to_device_(device)

    # ---- calibration amax accumulation (compiled-region safe) --------------

    def update_running_max(
        self,
        model,
        layer_idx: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """Update running amaxes in-place for one layer. Safe inside compiled region when unsynced."""
        model_state = self._models.get(id(model))
        if model_state is None or model_state.synced:
            return
        idx = layer_idx.reshape(-1).long()
        q_amax = q.abs().amax().reshape(1)
        k_amax = k.abs().amax().reshape(1)
        v_amax = v.abs().amax().reshape(1)
        model_state.q_running_max.index_copy_(
            0, idx, torch.maximum(model_state.q_running_max.index_select(0, idx), q_amax)
        )
        model_state.k_running_max.index_copy_(
            0, idx, torch.maximum(model_state.k_running_max.index_select(0, idx), k_amax)
        )
        model_state.v_running_max.index_copy_(
            0, idx, torch.maximum(model_state.v_running_max.index_select(0, idx), v_amax)
        )

    def update_o_running_max(
        self,
        model,
        layer_idx: torch.Tensor,
        o_descale: torch.Tensor,
    ):
        """Update o running amaxes in-place for one layer. Safe inside compiled region when unsynced."""
        model_state = self._models.get(id(model))
        if model_state is None or model_state.synced:
            return
        idx = layer_idx.reshape(-1).long()
        o_amax = o_descale.abs().amax().reshape(1)
        model_state.o_running_max.index_copy_(
            0, idx, torch.maximum(model_state.o_running_max.index_select(0, idx), o_amax)
        )

    def _scatter_scales_to_model(
        self,
        model,
        q_scales: torch.Tensor,
        k_scales: torch.Tensor,
        v_scales: torch.Tensor,
        o_scales: torch.Tensor,
    ):
        for i, block in enumerate(model.blocks):
            block.attn1.fp8_q_scale.copy_(q_scales[i : i + 1])
            block.attn1.fp8_k_scale.copy_(k_scales[i : i + 1])
            block.attn1.fp8_v_scale.copy_(v_scales[i : i + 1])
            block.attn1.fp8_o_scale.copy_(o_scales[i : i + 1])

    def sync(self, model) -> None:
        """All-reduce per-layer running amaxes and scatter scales into attn1 buffers.

        Call outside the compiled region after a calibration forward pass.
        """
        if self.fixed_scale is not None or model is None:
            return
        model_state = self.get_model_state(model)
        if model_state is None or model_state.synced:
            return
        if (
            model_state.q_running_max.max() == 0
            and model_state.k_running_max.max() == 0
            and model_state.v_running_max.max() == 0
        ):
            return
        from xfuser.core.distributed.attention_backend import AITER_FP8_DTYPE

        dtype_max = torch.finfo(AITER_FP8_DTYPE).max
        maxes = torch.stack(
            [
                model_state.q_running_max,
                model_state.k_running_max,
                model_state.v_running_max,
                model_state.o_running_max,
            ],
            dim=0,
        )
        dist.all_reduce(maxes, op=dist.ReduceOp.MAX, group=PROCESS_GROUP.ULYSSES_PG)
        # A smaller safety_factor enlarges the scale so live values above the
        # calibrated amax still map below FP8_MAX -> no clipping. Free for fp8.
        scales = maxes.clamp(min=1e-6) / (dtype_max * self.safety_factor)
        self._scatter_scales_to_model(model, scales[0], scales[1], scales[2], scales[3])
        model_state.q_running_max.zero_()
        model_state.k_running_max.zero_()
        model_state.v_running_max.zero_()
        model_state.o_running_max.zero_()
        model_state.synced = True
        self.calibrated_model_ids.add(id(model))
        if dist.get_rank() == 0:
            q_scales, k_scales, v_scales, o_scales = scales[0], scales[1], scales[2], scales[3]
            logger.info(
                f"[fp8_comms] {model.__class__.__name__} per-layer scales synced: "
                f"q=[{q_scales.min().item():.6f}, {q_scales.max().item():.6f}] "
                f"k=[{k_scales.min().item():.6f}, {k_scales.max().item():.6f}] "
                f"v=[{v_scales.min().item():.6f}, {v_scales.max().item():.6f}] "
                f"o=[{o_scales.min().item():.6f}, {o_scales.max().item():.6f}] "
                f"(amax q={maxes[0].max().item():.4f} k={maxes[1].max().item():.4f} "
                f"v={maxes[2].max().item():.4f} o={maxes[3].max().item():.4f})"
            )

    # ---- lifecycle hooks (called from base_model, thin) --------------------

    def register_models(self, pipe) -> None:
        """Register the pipe's transformer(s) and move running-max buffers to GPU."""
        for name in ("transformer", "transformer_2"):
            transformer = getattr(pipe, name, None)
            if transformer is not None and hasattr(transformer, "register_fp8_comms_state"):
                transformer.register_fp8_comms_state(self)
        if torch.cuda.is_available():
            self.to_device_(torch.device("cuda", torch.cuda.current_device()))

    def needs_calibration(self) -> bool:
        """Dynamic (calibrated) scaling needs a throwaway pass; fixed scale does not."""
        return self.fixed_scale is None

    def calibrate(
        self,
        pipe,
        input_args: dict,
        run_pipe_fn,
        split_prompts_fn=None,
        batch_size: Optional[int] = None,
    ) -> None:
        """Run one throwaway pipe call to calibrate per-layer FP8 comms scales.

        ``run_pipe_fn`` / ``split_prompts_fn`` are supplied by the runner so this class
        stays free of runner internals.
        """
        from xfuser.core.utils.runner_utils import log

        log("Calibrating FP8 comms scales (throwaway inference pass)...")
        calib_args = copy.deepcopy(input_args)
        if split_prompts_fn is not None:
            calib_args = split_prompts_fn(calib_args)
        if batch_size and isinstance(calib_args.get("prompt"), list):
            calib_args["prompt"] = calib_args["prompt"][:batch_size]
        run_pipe_fn(calib_args)
        for name in ("transformer", "transformer_2"):
            transformer = getattr(pipe, name, None)
            if transformer is not None:
                self.sync(transformer)
        log("FP8 comms calibration complete.")

    # ---- per-step observation hooks (called from the transformer forward) --

    def observe_qkv(self, attn, query, key, value, backend) -> bool:
        """During calibration, measure amaxes on the tensors USP will quantize.

        USP Hadamard-rotates Q,K before the fp8 all-to-all, so measure rotated copies
        (measurement-only; the flowing tensors are untouched). Returns whether this
        model's scales are already synced (i.e. fp8 comms may be applied this step).
        """
        fp8_owner = getattr(attn, "fp8_comms_owner", None)
        if fp8_owner is None or not hasattr(attn, "fp8_comms_layer_idx"):
            return False
        model_state = self.get_model_state(fp8_owner)
        if model_state is None:
            return False
        if not model_state.synced:
            from xfuser.core.distributed.attention_backend import rotate_qk_for_fp8_comms

            calib_query, calib_key = rotate_qk_for_fp8_comms(query, key, backend)
            self.update_running_max(
                fp8_owner, attn.fp8_comms_layer_idx, calib_query, calib_key, value
            )
        return model_state.synced

    def observe_output(self, attn, out) -> None:
        """During calibration, measure the attention-output amax for one layer."""
        fp8_owner = getattr(attn, "fp8_comms_owner", None)
        if fp8_owner is None or not hasattr(attn, "fp8_comms_layer_idx"):
            return
        self.update_o_running_max(fp8_owner, attn.fp8_comms_layer_idx, out)


# ---- attention-processor hooks (called from the transformer, thin) ---------
#
# Free functions (not methods) so callers never need a `fp8_comms is None` guard,
# and so the transformer never touches fp8 buffer names or the backend gate.


def install_fp8_comms_layer_state(transformer) -> None:
    """Register the per-layer fp8-comms buffers on each self-attention module.

    Owns the buffer contract (names/shapes + owner link). Buffers are non-persistent
    (kept out of the state_dict) and registered pre-compile so torch.compile captures
    them as graph inputs. Called after the model is loaded/moved to its device (via
    register_fp8_comms_state), so each buffer is placed on its module's device rather
    than defaulting to CPU.
    """
    for layer_idx, block in enumerate(transformer.blocks):
        attn = block.attn1
        device = next(attn.parameters()).device
        for name in ("fp8_q_scale", "fp8_k_scale", "fp8_v_scale", "fp8_o_scale"):
            attn.register_buffer(
                name, torch.ones(1, dtype=torch.float32, device=device), persistent=False
            )
        attn.register_buffer(
            "fp8_comms_layer_idx",
            torch.tensor([layer_idx], dtype=torch.long, device=device),
            persistent=False,
        )
        # Plain attribute (not an nn.Module child) so the transformer is not
        # registered as a submodule of attn (which would recurse forever).
        object.__setattr__(attn, "fp8_comms_owner", transformer)


def fp8_attention_kwargs(fp8_comms, attn, query, key, value, is_cross_attention, backend) -> dict:
    """Extras to splat into the attention call for fp8 comms, or ``{}`` when it does not apply.

    Also accumulates calibration amaxes as a side effect (via observe_qkv), which happens on
    every self-attn call regardless of the step's backend; fp8 comms is only *applied* once the
    scales are synced and the current backend supports pre-quantization.
    """
    if is_cross_attention or fp8_comms is None:
        return {}
    synced = fp8_comms.observe_qkv(attn, query, key, value, backend)
    if not synced:
        return {}
    from xfuser.core.distributed.attention_backend import SUPPORTS_PRE_QUANTIZATION_BACKENDS

    if backend not in SUPPORTS_PRE_QUANTIZATION_BACKENDS:
        return {}
    return {
        "fp8_comms": Fp8CommsCall(
            attn.fp8_q_scale,
            attn.fp8_k_scale,
            attn.fp8_v_scale,
            attn.fp8_o_scale,
        )
    }


def fp8_observe_output(fp8_comms, attn, out, is_cross_attention) -> None:
    """Measure the attention-output amax for calibration (no-op when fp8 comms is off/cross-attn)."""
    if is_cross_attention or fp8_comms is None:
        return
    fp8_comms.observe_output(attn, out)


# ---- quantized all-to-all helpers (called from USP, thin) ------------------


def _per_tensor_quant(x: torch.Tensor, scale_t: torch.Tensor):
    """Quantize x to FP8 using a fixed pre-allocated scale tensor. Returns (x_fp8, descale)."""
    import aiter

    fp8_dtype = aiter.dtypes.fp8
    return aiter.per_tensor_quant(
        x, scale=scale_t, quant_dtype=fp8_dtype, dtypeMax=torch.finfo(fp8_dtype).max
    )


def fp8_comms_input_all_to_all(query, key, value, q_scale, k_scale, v_scale, backend):
    """Rotate Q,K, quantize Q/K/V to FP8 using per-layer scales, and run interleaved
    input all-to-alls. Returns (query, key, value, attn_kwargs_update, qkv_amaxes)."""
    from xfuser.core.distributed.attention_backend import rotate_qk_for_fp8_comms
    from xfuser.model_executor.layers.usp import _ft_c_input_all_to_all

    query, key = rotate_qk_for_fp8_comms(query, key, backend)
    q_fp8, q_descale = _per_tensor_quant(query, q_scale)
    query = _ft_c_input_all_to_all(q_fp8)
    k_fp8, k_descale = _per_tensor_quant(key, k_scale)
    key = _ft_c_input_all_to_all(k_fp8)
    v_fp8, v_descale = _per_tensor_quant(value, v_scale)
    value = _ft_c_input_all_to_all(v_fp8)

    qkv_amaxes = (
        (q_descale.item(), k_descale.item(), v_descale.item()) if _FP8_LOG_SCALES else None
    )
    attn_kwargs_update = {
        "pre_quantized": True,
        "q_descale": q_descale,
        "k_descale": k_descale,
        "v_descale": v_descale,
    }
    return query, key, value, attn_kwargs_update, qkv_amaxes


def fp8_comms_output_all_to_all(out: torch.Tensor, o_scale: torch.Tensor, qkv_amaxes=None):
    """Quantize attention output to FP8, run output all-to-all, dequantize back."""
    from xfuser.model_executor.layers.usp import _ft_c_output_all_to_all

    if _FP8_LOG_SCALES and qkv_amaxes is not None:
        out_amax = out.abs().amax().item()
        rank = dist.get_rank()
        q_amax, k_amax, v_amax = qkv_amaxes
        logger.info(
            f"[fp8_scales rank{rank}] q_amax={q_amax:.4f} k_amax={k_amax:.4f} "
            f"v_amax={v_amax:.4f} out_amax={out_amax:.4f}"
        )
    if o_scale is None:
        raise RuntimeError("FP8 comms requires per-layer scale buffers fp8_o_scale")
    restore_dtype = out.dtype if out.dtype not in _FP8_DTYPES else torch.bfloat16
    if out.dtype not in _FP8_DTYPES:
        out_fp8, out_descale = _per_tensor_quant(out, o_scale)
    else:
        out_fp8, out_descale = out, o_scale
    return (_ft_c_output_all_to_all(out_fp8).float() * out_descale).to(restore_dtype)


# ---- config validation (module-level, mirrors validate_vae_config) ---------


def setup_fp8_comms(
    fp8_comms,
    pipe,
    input_args,
    run_pipe_fn,
    split_prompts_fn=None,
    batch_size=None,
) -> None:
    """Register the pipe's transformers and run calibration if needed.

    No-op when fp8 comms is off (``fp8_comms is None``). Owns the register -> calibrate
    sequence so the runner only makes one call; ``run_pipe_fn``/``split_prompts_fn`` are
    runner callbacks (calibration must run the pipe, which the runner owns).
    """
    if fp8_comms is None:
        return
    fp8_comms.register_models(pipe)
    if fp8_comms.needs_calibration():
        fp8_comms.calibrate(
            pipe,
            input_args,
            run_pipe_fn,
            split_prompts_fn=split_prompts_fn,
            batch_size=batch_size,
        )


def validate_fp8_comms_config(config, capabilities, settings) -> None:
    """Raise if --use_fp8_comms is requested but unsupported by the model/config; no-op if off."""
    if not config.use_fp8_comms:
        return
    from xfuser.core.distributed.attention_backend import (
        AttentionBackendType,
        SUPPORTS_PRE_QUANTIZATION_BACKENDS,
    )
    from xfuser.core.distributed.attention_schedule import AttentionSchedule

    def _parse(name, kind):
        if name is None:
            return None
        try:
            return AttentionBackendType[name.upper()]
        except KeyError:
            raise ValueError(f"Invalid {kind}: {name}")

    if not capabilities.use_fp8_comms:
        raise ValueError(f"Model {settings.model_name} does not support --use_fp8_comms.")
    if (config.ulysses_degree or 1) <= 1:
        raise ValueError("--use_fp8_comms requires ulysses_degree > 1.")
    effective_backends = set()
    if config.attention_backend:
        effective_backends.add(_parse(config.attention_backend, "attention backend"))
    if config.use_hybrid_attn_schedule:
        if config.hybrid_attn_schedule:
            effective_backends.update(
                AttentionSchedule.from_comma_delimited_string(
                    config.hybrid_attn_schedule
                ).backends
            )
        else:
            if config.hybrid_attn_low_precision_backend:
                effective_backends.add(
                    _parse(
                        config.hybrid_attn_low_precision_backend,
                        "hybrid low-precision attention backend",
                    )
                )
            if config.hybrid_attn_high_precision_backend:
                effective_backends.add(
                    _parse(
                        config.hybrid_attn_high_precision_backend,
                        "hybrid high-precision attention backend",
                    )
                )
    if not effective_backends & SUPPORTS_PRE_QUANTIZATION_BACKENDS:
        raise ValueError(
            f"--use_fp8_comms requires an attention backend that supports pre-quantization "
            f"({', '.join(b.name for b in SUPPORTS_PRE_QUANTIZATION_BACKENDS)}). "
            f"Set --attention_backend, --hybrid_attn_schedule, or "
            f"--hybrid_attn_low_precision_backend / --hybrid_attn_high_precision_backend "
            f"so at least one scheduled backend supports pre-quantization."
        )

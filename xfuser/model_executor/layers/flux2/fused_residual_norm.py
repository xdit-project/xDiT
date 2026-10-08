"""Fused gated-residual + modulated LayerNorm for FLUX.2 (Triton).

What it replaces
----------------
Every FLUX.2 block boundary -- both streams of a double block, and every single
block -- closes a residual and immediately re-normalises the result for the next
projection::

    x1 = x + gate * y
    h  = LayerNorm(x1) * (1 + scale) + shift

Inductor already fuses that pair into one kernel, but has to emit a
*non-persistent* reduction: D (4096 here) is too wide for one register tile at
the tile sizes it picks, so the row is streamed several times -- to reduce
``y``, to build ``x1`` while reducing it, and again to normalise ``x1`` into
``h``.  Forcing the persistent form through ``TORCHINDUCTOR_MULTI_KERNEL`` and
coordinate-descent tuning was tried and made things slower, not faster.

This kernel gives one row to one program and keeps both rows in registers, so
the traffic is exactly the four passes the dataflow needs: read ``y``, read
``x``, write ``x1``, write ``h``.  Measured against what Inductor emits, at the
three shapes FLUX.2-klein actually runs (bf16, MI355X):

    double-block image stream  [4096, 4096]   33.2 us -> 20.3 us   (+38.8%)
    double-block text stream   [ 512, 4096]   23.9 us -> 14.5 us   (+39.4%)
    single block               [4608, 4096]   34.6 us -> 23.4 us   (+32.5%)

Numerics
--------
``x1`` is rounded to bf16 *before* it is reduced, not just before it is stored,
because the unfused form round-trips it through bf16 memory between the residual
and the norm -- skipping that would feed the LayerNorm a value the reference
never sees.  Everything else is computed in fp32 and rounded once at each store,
which is what the compiled baseline this replaces also does.

Envelope / fallback
-------------------
Selection is from tensor properties alone -- no environment switch.  bf16 3-D
activations with a contiguous feature dim, matching shapes, broadcast
``(B, 1, D)`` modulation, and inference mode (the op has no autograd formula).
Anything else falls through to :func:`_reference`, the ops the model would
otherwise run, so selection changes the speed and never the result.
"""

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - only hit where triton is absent
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _gated_residual_layernorm_kernel(
        x_ptr, y_ptr, gate_ptr, scale_ptr, shift_ptr, x1_ptr, h_ptr,
        D: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK: tl.constexpr,
        ROWS_PER_MOD: tl.constexpr,
        EMIT_NORM: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < D
        # Modulation is (B, 1, D), broadcast over the sequence; rows of the
        # flattened (B*S, D) activation map to their batch by integer divide.
        mod_row = row // ROWS_PER_MOD

        y = tl.load(y_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + mod_row * D + cols, mask=mask, other=0.0).to(tl.float32)

        x1 = x + gate * y
        # Round before storing *and* before reducing: the unfused form
        # round-trips x1 through bf16 memory between the two, so the LayerNorm
        # must see the rounded row for the results to agree.
        x1_b = x1.to(tl.bfloat16)
        tl.store(x1_ptr + row * D + cols, x1_b, mask=mask)

        if EMIT_NORM:
            x1 = x1_b.to(tl.float32)
            mean = tl.sum(x1, 0) / D
            centered = tl.where(mask, x1 - mean, 0.0)
            rstd = tl.rsqrt(tl.sum(centered * centered, 0) / D + EPS)
            scale = tl.load(scale_ptr + mod_row * D + cols, mask=mask, other=0.0).to(tl.float32)
            shift = tl.load(shift_ptr + mod_row * D + cols, mask=mask, other=0.0).to(tl.float32)
            h = centered * rstd * (1.0 + scale) + shift
            tl.store(h_ptr + row * D + cols, h.to(tl.bfloat16), mask=mask)

    @torch.library.custom_op("xfuser::flux2_gated_residual_layernorm", mutates_args=())
    def _gated_residual_layernorm(
        x: torch.Tensor,      # [M, D] bf16 residual stream
        y: torch.Tensor,      # [M, D] bf16 attention / feed-forward output
        gate: torch.Tensor,   # [B, D] bf16
        scale: torch.Tensor,  # [B, D] bf16
        shift: torch.Tensor,  # [B, D] bf16
        eps: float,
        emit_norm: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Opaque launch boundary for torch.compile.

        The Triton launch lives inside the custom op so Dynamo treats it as a
        black box -- :func:`register_fake` propagates shapes during tracing and
        this body runs only with real tensors.
        """
        m, d = x.shape
        x1 = torch.empty_like(x)
        # The last block of a stack has nothing left to pre-normalise for; it
        # still has to return a tensor of the right shape for the schema, so
        # allocate an empty one rather than a second full row buffer.
        h = torch.empty_like(x) if emit_norm else x.new_empty((0, d))
        _gated_residual_layernorm_kernel[(m,)](
            x, y, gate, scale, shift, x1, h,
            D=d,
            EPS=float(eps),
            BLOCK=triton.next_power_of_2(d),
            ROWS_PER_MOD=m // gate.shape[0],
            EMIT_NORM=bool(emit_norm),
            # Swept 2/4/8/16 warps x 1/2 stages at the production shapes: this
            # is pure streaming plus two cheap reductions, so the narrowest
            # launch that still saturates the memory pipe wins.
            num_warps=4,
            num_stages=1,
        )
        return x1, h

    @_gated_residual_layernorm.register_fake
    def _(x, y, gate, scale, shift, eps, emit_norm):
        d = x.shape[1]
        return (
            torch.empty_like(x),
            torch.empty_like(x) if emit_norm else x.new_empty((0, d)),
        )


def _reference(x, y, gate, norm, scale, shift, emit_norm):
    """Verbatim copy of the ops the FLUX.2 block would otherwise run."""
    x1 = x + gate * y
    if not emit_norm:
        return x1, None
    return x1, norm(x1) * (1 + scale) + shift


def fused_gated_residual_layernorm(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    norm: torch.nn.Module,
    scale: torch.Tensor,
    shift: torch.Tensor,
    emit_norm: bool = True,
):
    """``(x + gate * y,  LayerNorm(that) * (1 + scale) + shift)``, fused.

    ``x`` / ``y`` are ``[B, S, D]``; ``gate`` / ``scale`` / ``shift`` are the
    ``[B, 1, D]`` modulation broadcast over the sequence.  ``norm`` is only read
    for its ``eps`` and to check it is the parameter-free LayerNorm FLUX.2 uses;
    with ``emit_norm=False`` the second output is None and only the residual is
    computed (the last block of a stack).  Falls back to :func:`_reference`
    whenever the fast envelope is not met.
    """
    _ref = lambda: _reference(x, y, gate, norm, scale, shift, emit_norm)  # noqa: E731

    if not _HAS_TRITON or torch.is_grad_enabled():
        return _ref()
    if not x.is_cuda or x.dtype != torch.bfloat16 or y.dtype != x.dtype:
        return _ref()
    if x.shape != y.shape or x.dim() != 3:
        return _ref()
    # FLUX.2's norms are nn.LayerNorm(dim, elementwise_affine=False): a weight
    # or bias here would be a different function than the kernel computes.
    if getattr(norm, "weight", None) is not None or getattr(norm, "bias", None) is not None:
        return _ref()

    b, s, d = x.shape
    for t in (gate, scale, shift):
        if t.shape != (b, 1, d) or t.dtype != torch.bfloat16:
            return _ref()
    # The kernel indexes rows linearly, so (B, S) must flatten without a copy
    # and the feature dim must be contiguous.
    for t in (x, y):
        if t.stride(-1) != 1 or t.stride(1) != d or t.stride(0) != s * d:
            return _ref()

    x1, h = torch.ops.xfuser.flux2_gated_residual_layernorm(
        x.reshape(b * s, d),
        y.reshape(b * s, d),
        gate.reshape(b, d),
        scale.reshape(b, d),
        shift.reshape(b, d),
        float(getattr(norm, "eps", 1e-6)),
        bool(emit_norm),
    )
    return x1.view(b, s, d), (h.view(b, s, d) if emit_norm else None)

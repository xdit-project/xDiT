"""Fused gated-residual + double RMSNorm for Z-Image (Triton).

What it replaces
----------------
Every modulated Z-Image block spends two of its kernels on the same shape --
a gated residual add whose result is immediately RMSNormed again for the next
projection::

    x1 = res + gate * RMSNorm(y,  w_post)      # close out attention / FFN
    h  =       RMSNorm(x1, w_pre) * scale      # open the next GEMM

Inductor already fuses that pair into a single kernel, but it has to emit a
*non-persistent* reduction: ``D = 3840`` is too wide for one register tile at
the tile sizes it picks, so the row is streamed three times -- once to reduce
``y``, once to build ``x1`` while reducing it, once to normalise ``x1`` into
``h``.  That is six row-passes of HBM traffic (``y`` twice, ``res`` once, ``x1``
written then read again, ``h`` once) for four passes of actual data movement.

This kernel gives one row to one program and keeps both rows in registers, so
the traffic is exactly the four passes the dataflow requires: read ``y``, read
``res``, write ``x1``, write ``h``.  Measured at the production shape
(``[8256, 3840]`` bf16, MI355X): 65.1 us -> 38.6 us, 5.84 -> 6.57 TB/s, which is
at the large-working-set bandwidth roof of the part.

Numerics
--------
Two roundings are load-bearing and are replayed.  ``ROUND_AFTER_NORM`` is
diffusers' own: RMSNorm rounds to the (bf16) weight dtype right after the
rstd-multiply.  And ``x1`` is rounded to bf16 *before* it is reduced, not just
before it is stored, because the unfused form round-trips it through bf16
memory between the two norms -- skipping that would feed the second RMSNorm a
value the reference never sees.

The remaining bf16 landing points of the eager expression (the products against
the norm weight and against the gate) are left in fp32, as the compiled baseline
this replaces also leaves them.  Replaying those too was measured: it costs
~2 us per launch and moves the end-to-end drift by nothing.  What dominates
instead is reduction *order* -- ``tl.sum`` over the row against ATen's tiled
``mean``.  Measured over a 30-block stack against eager: torch.compile's own
output already drifts 2.06e-2, this path 2.14e-2, so the fusion stays inside
the error the compiled baseline already carries.

Envelope / fallback
-------------------
Selection is from tensor properties alone -- no environment switch.  bf16
2-D activations, a contiguous feature dim, matching shapes, a broadcastable
``(B, 1, D)`` modulation whose row count divides the token count, and inference
mode (the op has no autograd formula).  Anything outside that falls through to
:func:`_reference`, the ops the model would otherwise run, so selection changes
the speed and never the result.
"""

from typing import Optional, Tuple

import torch

from xfuser.logger import init_logger

logger = init_logger(__name__)

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - only hit where triton is absent
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _gated_residual_norm_kernel(
        res_ptr, y_ptr, gate_ptr, w_post_ptr, w_pre_ptr, scale_ptr,
        x1_ptr, h_ptr,
        D: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK: tl.constexpr,
        ROWS_PER_MOD: tl.constexpr,
        ROUND_AFTER_NORM: tl.constexpr,
        HAS_W_POST: tl.constexpr,
        HAS_W_PRE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < D
        # The modulation is (B, 1, D) broadcast over the sequence; rows of the
        # flattened (B*S, D) activation map to their batch by integer divide.
        mod_row = row // ROWS_PER_MOD

        # ---- x1 = res + gate * RMSNorm(y, w_post) ----------------------------
        y = tl.load(y_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
        rstd1 = tl.rsqrt(tl.sum(y * y, 0) / D + EPS)
        yn = y * rstd1
        if ROUND_AFTER_NORM:
            yn = yn.to(tl.bfloat16).to(tl.float32)
        if HAS_W_POST:
            yn = yn * tl.load(w_post_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + mod_row * D + cols, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(res_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
        x1 = res + gate * yn

        # Round before storing *and* before reducing: the baseline round-trips
        # x1 through bf16 memory between the two norms, so the second RMSNorm
        # must see the rounded row for the results to agree.
        x1_b = x1.to(tl.bfloat16)
        tl.store(x1_ptr + row * D + cols, x1_b, mask=mask)
        x1 = x1_b.to(tl.float32)

        # ---- h = RMSNorm(x1, w_pre) * scale ----------------------------------
        rstd2 = tl.rsqrt(tl.sum(x1 * x1, 0) / D + EPS)
        hn = x1 * rstd2
        if ROUND_AFTER_NORM:
            hn = hn.to(tl.bfloat16).to(tl.float32)
        if HAS_W_PRE:
            hn = hn * tl.load(w_pre_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        hn = hn * tl.load(scale_ptr + mod_row * D + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(h_ptr + row * D + cols, hn.to(tl.bfloat16), mask=mask)

    @torch.library.custom_op(
        "xfuser::zimage_gated_residual_norm", mutates_args=()
    )
    def _gated_residual_norm(
        res: torch.Tensor,     # [M, D] bf16 residual stream
        y: torch.Tensor,       # [M, D] bf16 attention / FFN output
        gate: torch.Tensor,    # [B, D] bf16 (already tanh'd)
        w_post: Optional[torch.Tensor],  # [D] bf16 or None (weightless RMSNorm)
        w_pre: Optional[torch.Tensor],
        scale: torch.Tensor,   # [B, D] bf16 (already 1 + ...)
        eps: float,
        round_after_norm: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Opaque launch boundary for torch.compile.

        The Triton launch lives inside the custom op so Dynamo treats it as a
        black box -- :func:`register_fake` propagates shapes during tracing and
        this body runs only with real tensors.
        """
        m, d = res.shape
        x1 = torch.empty_like(res)
        h = torch.empty_like(res)
        dummy = res.new_empty(1)
        _gated_residual_norm_kernel[(m,)](
            res, y, gate,
            w_post if w_post is not None else dummy,
            w_pre if w_pre is not None else dummy,
            scale, x1, h,
            D=d,
            EPS=float(eps),
            BLOCK=triton.next_power_of_2(d),
            ROWS_PER_MOD=m // gate.shape[0],
            ROUND_AFTER_NORM=bool(round_after_norm),
            HAS_W_POST=w_post is not None,
            HAS_W_PRE=w_pre is not None,
            # 2 waves per row: the kernel is pure streaming + two cheap
            # reductions, so the narrowest launch that still saturates the
            # memory pipe wins -- wider ones only add reduction tree depth.
            # Swept 2/4/8/16 warps x 1/2 stages at the production shape.
            num_warps=2,
            num_stages=1,
        )
        return x1, h

    @_gated_residual_norm.register_fake
    def _(res, y, gate, w_post, w_pre, scale, eps, round_after_norm):
        return torch.empty_like(res), torch.empty_like(res)


def _reference(res, y, gate, norm_post, norm_pre, scale):
    """Verbatim copy of the ops the Z-Image block would otherwise run.

    Correctness oracle and fallback: running exactly what diffusers runs makes
    it numerically interchangeable with the fused path.
    """
    x1 = res + gate * norm_post(y)
    return x1, norm_pre(x1) * scale


def _plain_rmsnorm_weight(m: torch.nn.Module) -> Tuple[bool, Optional[torch.Tensor]]:
    """``(is_plain, weight)`` for an affine-or-weightless RMSNorm we reproduce."""
    if getattr(m, "bias", None) is not None:
        return False, None
    w = getattr(m, "weight", None)
    if w is None:
        return True, None
    if w.dim() != 1 or w.dtype != torch.bfloat16:
        return False, None
    return True, w


def fused_gated_residual_norm(
    res: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    norm_post: torch.nn.Module,
    norm_pre: torch.nn.Module,
    scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(res + gate * norm_post(y),  norm_pre(that) * scale)``, fused.

    ``res`` / ``y`` are ``[B, S, D]`` (the Z-Image residual stream); ``gate`` and
    ``scale`` are ``[B, 1, D]`` modulation broadcast over the sequence.  Falls
    back to :func:`_reference` whenever the fast envelope is not met.
    """
    _ref = lambda: _reference(res, y, gate, norm_post, norm_pre, scale)  # noqa: E731

    if not _HAS_TRITON:
        return _ref()
    # Inference-only: the custom op has no autograd formula.
    if torch.is_grad_enabled():
        return _ref()
    if not res.is_cuda or res.dtype != torch.bfloat16:
        return _ref()
    if res.shape != y.shape or y.dtype != res.dtype:
        return _ref()
    if res.dim() != 3 or gate.dim() != 3 or scale.dim() != 3:
        return _ref()

    b, s, d = res.shape
    # gate / scale are (B, 1, D) broadcast over S; anything else (per-token
    # modulation, a different feature width) is out of envelope.
    for t in (gate, scale):
        if t.shape != (b, 1, d) or t.dtype != torch.bfloat16:
            return _ref()
    # The kernel indexes rows linearly, so (B, S) must flatten without a copy
    # and the feature dim must be contiguous.
    if res.stride(-1) != 1 or y.stride(-1) != 1:
        return _ref()
    if res.stride(1) != d or y.stride(1) != d or res.stride(0) != s * d or y.stride(0) != s * d:
        return _ref()

    ok_post, w_post = _plain_rmsnorm_weight(norm_post)
    ok_pre, w_pre = _plain_rmsnorm_weight(norm_pre)
    if not (ok_post and ok_pre):
        return _ref()

    eps = float(getattr(norm_post, "eps", 1e-5))
    if float(getattr(norm_pre, "eps", eps)) != eps:
        return _ref()  # one eps per launch

    # diffusers' RMSNorm only rounds the normalised activation to the weight
    # dtype when that dtype is half; bf16 weights here always trigger it.
    round_after_norm = True

    x1, h = torch.ops.xfuser.zimage_gated_residual_norm(
        res.reshape(b * s, d),
        y.reshape(b * s, d),
        gate.reshape(b, d),
        w_post.contiguous() if w_post is not None else None,
        w_pre.contiguous() if w_pre is not None else None,
        scale.reshape(b, d),
        eps,
        round_after_norm,
    )
    return x1.view(b, s, d), h.view(b, s, d)

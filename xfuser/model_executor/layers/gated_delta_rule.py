"""Faster eager fallback for Qwen3.5's chunked gated delta rule (MOSS-Image text encoder).

Without ``fla`` installed, transformers runs ``torch_chunk_gated_delta_rule``, whose intra-chunk
forward substitution is a 63-step Python loop of tiny kernels (two clones each). Across 24 linear
attention layers and two prompts that is ~15k launches per inference. The loop computes
``(I - L)^-1`` for the strictly lower triangular ``L``, which one batched triangular solve does.
"""

import torch
import torch.nn.functional as F


def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _unit_lower_inverse(lhs, eye):
    return torch.linalg.solve_triangular(lhs, eye.expand_as(lhs), upper=False, unitriangular=True)


def chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query)
        key = _l2norm(key)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * (1 / (query.shape[-1] ** 0.5))

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size).cumsum(dim=-1)
    upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    decay_mask = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()
    eye = torch.eye(chunk_size, dtype=torch.float32, device=query.device)
    # I - L, with L = -(k_beta k^T * decay) strictly lower triangular.
    lhs = ((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(upper, 0) + eye
    attn = _unit_lower_inverse(lhs, eye)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.empty_like(value)
    strict_upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    intra = (query @ key.transpose(-1, -2) * decay_mask).masked_fill(strict_upper, 0)
    q_decay = query * g.exp().unsqueeze(-1)
    k_decay = key * (g[..., -1:] - g).exp().unsqueeze(-1)
    g_last = g[..., -1].exp()[..., None, None]
    for i in range(total_sequence_length // chunk_size):
        v_new = value[:, :, i] - k_cumdecay[:, :, i] @ state
        core_attn_out[:, :, i] = q_decay[:, :, i] @ state + intra[:, :, i] @ v_new
        state = state * g_last[:, :, i] + k_decay[:, :, i].transpose(-1, -2) @ v_new

    core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, v_head_dim)[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, (state if output_final_state else None)


def patch_qwen3_5_gated_delta():
    """Route transformers' Qwen3.5 (and Qwen3-Next) chunked delta rule to the batched version."""
    patched = False
    for name in ("qwen3_5", "qwen3_next"):
        try:
            module = __import__(f"transformers.models.{name}.modeling_{name}", fromlist=["_"])
        except ImportError:
            continue
        if hasattr(module, "torch_chunk_gated_delta_rule"):
            module.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
            patched = True
    return patched

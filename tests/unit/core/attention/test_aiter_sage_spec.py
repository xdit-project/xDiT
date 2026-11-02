"""AITER_SAGE refuses causal calls.

Its v1 kernel is never passed `causal`, so a causal call would silently get
full attention. The refusal is declared on the spec, so it holds without AITER.
"""

import torch

from xfuser.core.attention import registry
from xfuser.core.attention.spec import AttentionBackendType, AttnCall


def test_aiter_sage_refuses_causal_calls_and_serves_the_rest():
    spec = registry.get(AttentionBackendType.AITER_SAGE)
    query = torch.empty(1, 2, 16, 128)

    reason = spec.rejects(query, query, query, AttnCall(is_causal=True))

    assert reason is not None and "causal" in reason
    assert spec.rejects(query, query, query, AttnCall(is_causal=False)) is None

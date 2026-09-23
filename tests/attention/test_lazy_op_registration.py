"""Can a kernel module be imported lazily if it registers torch.library ops?

The split-backend design resolves a spec's implementation on first use. If that
import happens inside a torch.compile'd region -- which it would, since
attention is called from a compiled transformer forward -- then any custom op
the kernel module registers is declared mid-trace.

These tests establish whether that is tolerated, and therefore whether the
implementation may be resolved lazily or must be resolved when the backend is
selected, before any compiled call.

CPU only; Dynamo tracing is what is being tested, not the kernel.
"""

import importlib
import itertools
import sys

import pytest
import torch

_counter = itertools.count()

MODULE_SOURCE = '''
import torch
from torch.library import custom_op, register_fake


@custom_op("xfuser_probe::{name}", mutates_args=())
def _op(x: torch.Tensor) -> torch.Tensor:
    return x * 2


@register_fake("xfuser_probe::{name}")
def _op_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def call(x):
    return torch.ops.xfuser_probe.{name}(x)
'''


def _make_kernel_module(tmp_path) -> str:
    """A stand-in for backends/<family>/kernel.py: registers an op at import."""
    name = f"probe{next(_counter)}"
    module = f"lazy_kernel_{name}"
    (tmp_path / f"{module}.py").write_text(MODULE_SOURCE.format(name=name))
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))
    return module


def _reset_dynamo():
    torch._dynamo.reset()


def test_op_registered_before_compile_is_fine(tmp_path):
    """Baseline: resolve at selection time, i.e. before anything is traced."""
    module = importlib.import_module(_make_kernel_module(tmp_path))
    _reset_dynamo()

    @torch.compile(fullgraph=True, backend="eager")
    def run(x):
        return module.call(x)

    x = torch.ones(4)
    assert torch.allclose(run(x), x * 2)


@pytest.mark.xfail(
    strict=True,
    reason="Dynamo refuses to trace importlib._gcd_import, so a kernel module "
           "cannot be resolved from inside a compiled region. This is why the "
           "implementation is resolved when the backend is selected. Strict: if "
           "a torch upgrade lifts this, the lazy design becomes available.",
)
def test_op_registered_during_first_traced_call(tmp_path):
    """The lazy design: the kernel module is imported from inside the region
    being traced, so the op is registered mid-trace."""
    name = _make_kernel_module(tmp_path)
    _reset_dynamo()

    @torch.compile(fullgraph=True, backend="eager")
    def run(x):
        module = importlib.import_module(name)  # registration happens here
        return module.call(x)

    x = torch.ones(4)
    try:
        out = run(x)
    except Exception as exc:
        pytest.fail(
            "fullgraph tracing rejected mid-trace op registration -> the "
            "implementation must be resolved before the first compiled call. "
            f"{type(exc).__name__}: {exc}"
        )
    assert torch.allclose(out, x * 2)


def test_lazy_import_without_fullgraph(tmp_path):
    """If fullgraph rejects it, does it merely graph-break and still run?"""
    name = _make_kernel_module(tmp_path)
    _reset_dynamo()

    @torch.compile(backend="eager")
    def run(x):
        module = importlib.import_module(name)
        return module.call(x)

    x = torch.ones(4)
    assert torch.allclose(run(x), x * 2)


def test_lazy_import_through_inductor(tmp_path):
    """The realistic path: default backend, not just Dynamo."""
    name = _make_kernel_module(tmp_path)
    _reset_dynamo()

    @torch.compile
    def run(x):
        module = importlib.import_module(name)
        return module.call(x)

    x = torch.ones(4)
    assert torch.allclose(run(x), x * 2)


# --------------------------------------------------------------------------
# the dispatch path itself
# --------------------------------------------------------------------------

def test_spec_run_is_traceable_under_fullgraph():
    """Phase 5 put Spec.run on the hot path: it builds an AttnCall, walks the
    constraint chain, then calls impl. All of that now sits inside the region
    the transformer forward traces, where there used to be a dict lookup."""
    from xfuser.core.attention import registry
    from xfuser.core.attention.spec import AttentionBackendType, AttnCall

    spec = registry.get(AttentionBackendType.SDPA)
    _reset_dynamo()

    @torch.compile(fullgraph=True, backend="eager")
    def run(q, k, v):
        out, _ = spec.run(q, k, v, AttnCall())
        return out

    q = torch.randn(1, 2, 8, 16)
    expected, _ = spec.run(q, q, q, AttnCall())
    assert torch.allclose(run(q, q, q), expected)


@pytest.mark.xfail(
    strict=True,
    reason="A refusal raised inside a traced region surfaces as "
           "torch._dynamo.exc.Unsupported rather than the NotImplementedError "
           "itself; the real message survives in the debug context. Only "
           "reachable on a misconfigured run, so accepted. Strict: if torch "
           "starts propagating it cleanly, drop the caveat in Spec.run.",
)
def test_constraint_rejection_is_traceable_under_fullgraph():
    """A refusal raises from inside the traced region; make sure that is a
    clean exception rather than a Dynamo failure."""
    from xfuser.core.attention import registry
    from xfuser.core.attention.spec import AttentionBackendType, AttnCall

    spec = registry.get(AttentionBackendType.SDPA)  # declares NO_VARLEN
    _reset_dynamo()

    @torch.compile(fullgraph=True, backend="eager")
    def run(q, k, v, call):
        out, _ = spec.run(q, k, v, call)
        return out

    from xfuser.core.attention.spec import VarlenPacking

    q = torch.randn(1, 2, 8, 16)
    packed = AttnCall(
        varlen=VarlenPacking(
            indices_k=torch.tensor([0]),
            cu_seqlens_k=torch.tensor([0, 1]),
            max_seqlen_k=1,
        )
    )
    with pytest.raises(NotImplementedError, match="varlen"):
        run(q, q, q, packed)


def test_deep_constraint_chain_traces_clean_when_satisfied():
    """The MHA v4 chain is the deepest in the package -- four constraints
    including a tuple-membership test on head_dim. Standard usage walks it on
    every call, so it must fold away rather than break the graph."""
    from xfuser.core.attention.backends.aiter_mha_v4 import DENSE_CALLS, SPARGE_CALLS
    from xfuser.core.attention.spec import AttentionBackendType, AttnCall, Spec

    def impl(q, k, v, call):
        return q + 1, None

    q = torch.randn(1, 4, 32, 128)  # satisfies head_dim=128, MHA, self-attn

    for name, accepts in (("dense", DENSE_CALLS), ("sparge", SPARGE_CALLS)):
        spec = Spec(AttentionBackendType.AITER_BF16, impl=impl, accepts=accepts)
        assert spec.rejects(q, q, q, AttnCall()) is None, name

        _reset_dynamo()

        @torch.compile(fullgraph=True, backend="eager")
        def run(x):
            out, _ = spec.run(x, x, x, AttnCall())
            return out

        assert torch.allclose(run(q), q + 1), name


def test_repeated_dispatch_does_not_recompile():
    """Constraint results depend on shapes, not values, so a steady-state run
    should compile once and reuse."""
    from xfuser.core.attention import registry
    from xfuser.core.attention.spec import AttentionBackendType, AttnCall

    spec = registry.get(AttentionBackendType.SDPA)
    _reset_dynamo()

    compiles = 0

    def count(*_args, **_kwargs):
        nonlocal compiles
        compiles += 1
        return torch._dynamo.backends.debugging.eager(*_args, **_kwargs)

    @torch.compile(fullgraph=True, backend=count)
    def run(q):
        out, _ = spec.run(q, q, q, AttnCall())
        return out

    q = torch.randn(1, 2, 8, 16)
    for _ in range(5):
        run(q)
    assert compiles == 1, f"recompiled {compiles} times for identical shapes"

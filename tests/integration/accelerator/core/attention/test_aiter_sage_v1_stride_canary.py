"""Canary for the Sage v1 layout workaround in xfuser/core/attention/backends/aiter_sage/kernel.py.

AITER's Sage v1 kernel misreads Q/K/V in some layouts: its quantization reads
`K - mean(K)`, a new dense tensor, and V through K's original strides, and a
sequence slice of Q comes out as garbage too. `_v1_readable` copies the tensors
in those layouts. Each case feeds the raw kernel one of them and passes while
the kernel still gets it wrong; a failure means AITER fixed that layout, so the
matching part of the workaround can go.
"""

import json
import math
import os
import subprocess
import sys

import pytest
import torch

from xfuser.core.attention import registry
from xfuser.core.attention.spec import AttentionBackendType

# Run in a child process: a misread can also be a GPU memory fault, which kills the process instead of raising.
_CHILD = r"""
import json
import sys

import torch
from aiter.ops.triton.attention.fav3_sage import fav3_sage_wrapper_func as sage

heads, seq_len, head_dim, sp_pad = 4, 1024, 128, 4
generator = torch.Generator().manual_seed(0)


def randn(*shape):
    return torch.randn(*shape, generator=generator).to(device="cuda", dtype=torch.bfloat16)


def sliced():
    return randn(1, heads, seq_len + sp_pad, head_dim)[:, :, :seq_len]


layout = sys.argv[1]
query = sliced() if layout == "query-sequence-sliced" else randn(1, heads, seq_len, head_dim)
if layout == "kv-sequence-sliced":
    key, value = sliced(), sliced()
elif layout == "value-strides-differ":
    key, value = randn(1, heads, seq_len, head_dim), randn(1, seq_len, heads, head_dim).transpose(1, 2)
else:
    key, value = randn(1, heads, seq_len, head_dim), randn(1, heads, seq_len, head_dim)


def relative_error(query, key, value):
    output = sage(query, key, value, layout="bhsd").float()
    reference = torch.nn.functional.scaled_dot_product_attention(query.float(), key.float(), value.float())
    return ((output - reference).norm() / reference.norm()).item()


print(json.dumps({"contiguous": relative_error(query.contiguous(), key.contiguous(), value.contiguous())}), flush=True)
print(json.dumps({"strided": relative_error(query, key, value)}), flush=True)
"""

_KERNEL = "xfuser/core/attention/backends/aiter_sage/kernel.py"
LAYOUTS = {
    # K/V sliced along the sequence, as USP drops the pad: head stride larger than seq_len * head_dim.
    "kv-sequence-sliced": "the dense check on K in _v1_readable",
    # Contiguous K with V transposed from BSHD: either alone is dense, but V's strides are not K's.
    "value-strides-differ": "the stride comparison (_same_strides) in _v1_readable",
    # Q sliced along the sequence, as setup_sparge strips the pad without reordering.
    "query-sequence-sliced": "the check on Q in _v1_readable (and its query argument)",
}


@pytest.mark.parametrize("layout", LAYOUTS)
def test_sage_v1_kernel_still_misreads(layout):
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator")
    reason = registry.get(AttentionBackendType.AITER_SAGE).unavailable()
    if reason is not None:
        pytest.skip(reason)

    child = subprocess.run(
        [sys.executable, "-c", _CHILD, layout],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "HSA_DISABLE_COREDUMP_ON_EXCEPTION": "1"},
    )

    results = {}
    for line in child.stdout.splitlines():
        if line.startswith("{"):
            results.update(json.loads(line))
    assert "contiguous" in results, f"the kernel failed on contiguous inputs, so this cannot tell:\n{child.stderr}"
    if "strided" not in results:
        return  # The strided call crashed the process: still misread.
    strided, contiguous = results["strided"], results["contiguous"]
    # Measured on gfx950: 3x (V strides) up to inf (sliced K/V) the contiguous call's error.
    still_misread = not math.isfinite(strided) or strided > 2 * contiguous
    assert still_misread, (
        f"AITER's Sage v1 now reads {layout} inputs correctly (relative error {strided:.3e}, contiguous "
        f"{contiguous:.3e}). Remove {LAYOUTS[layout]} from {_KERNEL}, then this case."
    )

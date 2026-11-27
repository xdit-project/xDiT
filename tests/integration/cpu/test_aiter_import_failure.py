"""Modules with an optional AITER dependency must load when ``import aiter`` fails.

``import aiter`` can fail with an error that is not ImportError. For example, an
AITER install on a host without a visible GPU raises RuntimeError("No gfx arch
found ..."). The modules must then load without AITER, and Sol-Attn must report
the original error when a caller asks for it. Each case runs in a fresh
interpreter, because the pytest process can already have loaded these modules.
The interpreter also reports no GPU, as on such a host: with a GPU, xfuser.envs
loads packages such as yunchang, which import AITER themselves.
"""

import json
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.slow

FAIL_AITER_IMPORT = textwrap.dedent(
    """
    import importlib.abc, json, sys

    import torch

    class AiterWithoutGpu(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "aiter" or name.startswith("aiter."):
                raise RuntimeError("No gfx arch found in rocminfo output.")
            return None

    sys.meta_path.insert(0, AiterWithoutGpu())
    torch.cuda.is_available = lambda: False
    """
)


def _run(code: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", FAIL_AITER_IMPORT + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.splitlines()[-1])


def test_the_layers_load_without_aiter_when_its_import_raises_runtime_error():
    report = _run(
        """
        import xfuser.model_executor.layers
        from xfuser.model_executor.layers import fp8_linear, norms

        print(json.dumps({"has_aiter_norm": norms._HAS_AITER, "has_preshuffle": fp8_linear._HAS_PRESHUFFLE}))
        """
    )

    assert report == {"has_aiter_norm": False, "has_preshuffle": False}


def test_sol_attn_reports_the_aiter_import_error():
    report = _run(
        """
        from xfuser.core.attention.sparsity import sol

        try:
            sol.check_sol_attn_device()
        except sol.SolAttnUnsupported as error:
            message, cause = str(error), repr(error.__cause__)
        print(json.dumps({"available": sol.SOL_ATTN_AVAILABLE, "message": message, "cause": cause}))
        """
    )

    assert report["available"] is False
    assert "RuntimeError: No gfx arch found" in report["message"]
    assert report["cause"].startswith("RuntimeError(")

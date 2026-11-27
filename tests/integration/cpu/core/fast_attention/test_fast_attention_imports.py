"""xfuser.core.fast_attention must load without xfuser.model_executor.

xfuser.model_executor.base_wrapper imports xfuser.core.fast_attention. If that
package imports xfuser.model_executor when it loads, the import is circular
(#999). Each case runs in a fresh interpreter, because an earlier test can
already have loaded these modules into the pytest process.
"""

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.slow


def test_fast_attention_loads_without_the_model_executor():
    code = textwrap.dedent(
        """
        import sys
        import xfuser.core.fast_attention
        from xfuser.core.fast_attention import FastAttnMethod, get_fast_attn_enable, xFuserFastAttention
        from xfuser.core.fast_attention.utils import load_config_file, save_config_file
        print("model_executor_loaded", "xfuser.model_executor" in sys.modules)
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "model_executor_loaded False"

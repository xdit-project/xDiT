"""Unit tests for how the AITER FP8 quantizers reach diffusers and transformers.

Registering a quantizer mutates a process-global mapping in the host framework, and the transformers
half is built on an API that only exists in transformers 5. Both are therefore deferred to the call
sites that build one of our configs, so that merely importing a runner neither reaches into
transformers 5-only modules nor changes another library's behaviour. These tests pin that; they are
CPU-only.

Run with:
    pytest tests/core/test_fp8_quantizer_packaging.py -v
"""

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("diffusers.quantizers")

from xfuser.model_executor.quant import aiter_fp8_quantizer as quant

# Prelude for the probes below: reports (diffusers, transformers) auto-mapping membership.
_PROBE_PRELUDE = """
def registered():
    from diffusers.quantizers.auto import AUTO_QUANTIZER_MAPPING as d
    from transformers.quantizers.auto import AUTO_QUANTIZER_MAPPING as t
    return (any("aiter" in k for k in d), any("aiter" in k for k in t))
"""


def probe_registration(body):
    """Run body in a new interpreter and return its last line: registration is process-global and
    sticky, so a fresh process is the only honest way to ask what an import alone did.
    """
    result = subprocess.run(
        [sys.executable, "-c", _PROBE_PRELUDE + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        env={**os.environ, "RANK": "0", "WORLD_SIZE": "1"},
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return result.stdout.strip().splitlines()[-1]


@pytest.mark.slow
def test_importing_the_quantizer_registers_nothing():
    """A run that never quantizes must leave both frameworks' auto-mappings untouched."""
    assert probe_registration("""
        import xfuser.model_executor.quant  # noqa: F401
        print(registered())
    """) == "(False, False)"


@pytest.mark.slow
def test_dit_config_registers_only_the_diffusers_side():
    """The DiT config must not drag in the transformers quantizer, whose ops need transformers 5."""
    assert probe_registration("""
        from xfuser.model_executor.quant import AiterFp8BlockScaleConfig
        AiterFp8BlockScaleConfig(target_modules=["blocks"])
        print(registered())
    """) == "(True, False)"


@pytest.mark.slow
def test_text_encoder_config_registers_the_transformers_side():
    assert probe_registration("""
        from xfuser.model_executor.quant import AiterFp8BlockScaleTEConfig
        AiterFp8BlockScaleTEConfig(target_modules=["layers"])
        print(registered())
    """) == "(False, True)"


@pytest.mark.slow
def test_importing_the_torchao_policy_registers_nothing():
    """Swapping diffusers' torchao quantizer is global, so only a streaming load may do it."""
    assert probe_registration("""
        import xfuser.model_executor.quant.torchao_quantizer  # noqa: F401
        from diffusers.quantizers.auto import AUTO_QUANTIZER_MAPPING as d
        from diffusers.quantizers.quantization_config import QuantizationMethod
        installed = d[QuantizationMethod.TORCHAO]
        print(getattr(installed, "use_keep_in_fp32_modules", False))
    """) == "False"


@pytest.mark.slow
def test_the_torchao_policy_opts_into_the_models_fp32_modules():
    """Without this, a streamed TorchAO load demotes the modules a model pins to fp32, so it would
    load different weights than the post-load quantization walk for the same checkpoint."""
    assert probe_registration("""
        from diffusers.quantizers.auto import AUTO_QUANTIZER_MAPPING as d
        from diffusers.quantizers.quantization_config import QuantizationMethod
        from xfuser.model_executor.quant.torchao_quantizer import (
            register_torchao_fp32_policy,
        )
        original = d[QuantizationMethod.TORCHAO]
        register_torchao_fp32_policy()
        installed = d[QuantizationMethod.TORCHAO]
        print((
            installed.use_keep_in_fp32_modules,
            issubclass(installed, original),
        ))
    """) == "(True, True)"


def test_the_torchao_policy_leaves_an_opted_in_quantizer_alone(monkeypatch):
    """A no-op once diffusers opts in upstream, rather than a redundant subclass."""
    from diffusers.quantizers.auto import AUTO_QUANTIZER_MAPPING
    from diffusers.quantizers.quantization_config import QuantizationMethod
    from xfuser.model_executor.quant import torchao_quantizer

    class AlreadyOptedIn:
        use_keep_in_fp32_modules = True

    monkeypatch.setitem(
        AUTO_QUANTIZER_MAPPING, QuantizationMethod.TORCHAO, AlreadyOptedIn
    )
    torchao_quantizer.register_torchao_fp32_policy.cache_clear()
    try:
        torchao_quantizer.register_torchao_fp32_policy()
        assert (
            AUTO_QUANTIZER_MAPPING[QuantizationMethod.TORCHAO] is AlreadyOptedIn
        )
    finally:
        torchao_quantizer.register_torchao_fp32_policy.cache_clear()


def test_every_diffusers_side_streaming_quantizer_keeps_the_fp32_modules():
    """Diffusers gates `_keep_in_fp32_modules` on the active quantizer opting in, so a quantizer
    that does not silently loads the pinned modules in the compute dtype. Both routes into a
    diffusers model have to opt in, or streaming and the post-load walk load the same checkpoint
    differently."""
    assert quant.AiterFp8BlockScaleQuantizer.use_keep_in_fp32_modules is True


def test_missing_streaming_loader_gives_an_actionable_error(monkeypatch):
    """On a transformers below the streaming loader, the failure has to name the requirement rather
    than surface as an ImportError from inside the load."""
    monkeypatch.setattr(quant, "_has_transformers_conversion_ops", lambda: False)
    quant._quantize_op_cls.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="transformers 5"):
            quant._quantize_op_cls()
    finally:
        quant._quantize_op_cls.cache_clear()


@pytest.mark.slow
def test_streaming_op_is_built_only_on_demand():
    """The op class is what needs transformers 5, so nothing may build it before a quantized load —
    not importing the quantizer, and not building the text-encoder config either. Asked in a fresh
    interpreter because the op class is memoized process-wide."""
    assert probe_registration("""
        from xfuser.model_executor.quant import AiterFp8BlockScaleTEConfig
        from xfuser.model_executor.quant import aiter_fp8_quantizer as q
        AiterFp8BlockScaleTEConfig(target_modules=["layers"])
        print(q._quantize_op_cls.cache_info().currsize)
    """) == "0"

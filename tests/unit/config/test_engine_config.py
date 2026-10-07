from types import SimpleNamespace

import pytest

from xfuser.config.config import EngineConfig


def _engine_config(pp_degree: int, warmup_steps: int) -> EngineConfig:
    return EngineConfig(
        model_config=None,
        runtime_config=SimpleNamespace(warmup_steps=warmup_steps),
        parallel_config=SimpleNamespace(pp_degree=pp_degree),
        fast_attn_config=SimpleNamespace(use_fast_attn=False),
    )


def test_pipefusion_rejects_zero_warmup_steps():
    with pytest.raises(ValueError, match="warmup_steps must be at least 1"):
        _engine_config(pp_degree=2, warmup_steps=0)


@pytest.mark.parametrize(("pp_degree", "warmup_steps"), [(2, 1), (1, 0)])
def test_warmup_steps_accepted_when_pipefusion_can_fill_its_cache(pp_degree, warmup_steps):
    assert _engine_config(pp_degree=pp_degree, warmup_steps=warmup_steps).runtime_config.warmup_steps == warmup_steps

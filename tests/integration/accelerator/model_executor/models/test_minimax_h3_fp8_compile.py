"""MiniMax-H3's FP8 AdaLN projection must compile for every timestep count a run produces.

A MiniMax-H3 step packs rows at up to four noise levels and hands the transformer only the distinct
ones, so the timestep's length changes between steps. FP4 runs keep ``adaln_proj.linear`` (the GEMM
the timestep embedding feeds) in torchao FP8 with a per-tensor activation scale, and torchao's
quantizer compares every activation dimension with 1. When the runner marked that length unbacked,
dynamo could not answer the comparison and ``--use_fp4_gemms --use_torch_compile`` failed with
``Could not guard on data-dependent expression Ne(u0, 1)``.
"""

import math

import pytest
import torch

from xfuser.core.utils.runner_utils import FP8_ACTIVATION_SCALE_FLOOR

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FP8 quantization needs a device")

# Wide enough to take the same tensorwise kernel the models use; a 128-wide layer selects a
# blockwise one instead.
_WIDTH = 1024


class _TimestepModulation(torch.nn.Module):
    """The timestep's path through MiniMax-H3: sinusoidal embedding, then the AdaLN projection."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(_WIDTH, _WIDTH)

    def forward(self, timestep, timestep_indices):
        half = _WIDTH // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=timestep.device) / half)
        args = timestep.float()[:, None] * 1000.0 * freqs[None]
        temb = torch.cat((args.cos(), args.sin()), dim=-1).to(self.linear.weight.dtype)
        return self.linear(temb)[timestep_indices]


def _fp8_modulation():
    pytest.importorskip("torchao")
    from torchao.quantization.granularity import PerTensor
    from torchao.quantization.quant_api import (
        Float8DynamicActivationFloat8WeightConfig,
        quantize_,
    )

    torch.manual_seed(0)
    module = _TimestepModulation().to(device="cuda", dtype=torch.bfloat16)
    quantize_(
        module.linear,
        Float8DynamicActivationFloat8WeightConfig(
            granularity=PerTensor(),
            set_inductor_config=False,
            activation_value_lb=FP8_ACTIVATION_SCALE_FLOOR,
        ),
    )
    return module


_TIMESTEP_COUNTS = ([0.9], [0.4, 0.9], [0.1, 0.4, 0.7, 0.9])


def _step_inputs(timesteps):
    timestep = torch.tensor(timesteps, device="cuda")
    # One index per packed row, so a fixed count, and every distinct timestep is addressed by some
    # row, as build_row_timesteps produces.
    timestep_indices = torch.arange(12, device="cuda") % len(timesteps)
    return timestep, timestep_indices


def test_fp8_adaln_compiles_once_for_every_timestep_count():
    from xfuser.model_executor.models.runner_models.minimax_h3 import _wrap_compiled_forward

    module = _fp8_modulation()
    torch._dynamo.reset()
    forward = _wrap_compiled_forward(
        module,
        module.forward,
        torch.compile(module.forward, fullgraph=True),
    )

    # The runner runs inference under inference_mode; FP8 _scaled_mm has no backward.
    with torch.inference_mode(), torch._dynamo.config.patch(error_on_recompile=True):
        for timesteps in _TIMESTEP_COUNTS:
            timestep, timestep_indices = _step_inputs(timesteps)

            compiled = forward(timestep, timestep_indices)

            # Inductor fuses the activation quantization and rounds it a bf16 ulp or two apart.
            torch.testing.assert_close(compiled, module(timestep, timestep_indices), atol=3e-2, rtol=1.6e-2)


def test_padding_the_timestep_leaves_the_fp8_rows_unchanged():
    """A repeated timestep row cannot move the per-tensor activation scale."""
    from xfuser.model_executor.models.runner_models.minimax_h3 import _pad_timestep

    module = _fp8_modulation()
    with torch.inference_mode():
        for timesteps in _TIMESTEP_COUNTS:
            timestep, timestep_indices = _step_inputs(timesteps)

            assert torch.equal(
                module(_pad_timestep(timestep), timestep_indices),
                module(timestep, timestep_indices),
            )

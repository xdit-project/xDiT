"""CPU-only tests for applying a compute dtype without dropping pinned fp32 precision.

The two libraries xFuser loads from disagree on when a model's fp32 pins apply, so the rule is
resolved per model rather than read off one attribute. Getting that wrong is silent either way: too
little preservation loads a lower-precision model than an ordinary load, too much diverges from one.
"""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for dtype policy tests")

from xfuser.core.utils.dtype_policy import (  # noqa: E402
    cast_preserving_fp32_modules,
    fp32_modules_for,
    keeps_fp32,
    pinned_fp32_parameters,
)


def _pinned_model():
    """A stand-in for a diffusers transformer, which pins at any compute dtype."""

    model = torch.nn.Module()
    model._keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm2"]
    model.blocks = torch.nn.ModuleList([torch.nn.Module()])
    block = model.blocks[0]
    block.norm2 = torch.nn.LayerNorm(4)
    block.attn = torch.nn.Linear(4, 4)
    block.scale_shift_table = torch.nn.Parameter(torch.zeros(1, 6, 4))
    model.time_embedder = torch.nn.Linear(4, 4)
    model.register_buffer("token_ids", torch.zeros(4, dtype=torch.long))
    model.register_buffer("rope_freqs", torch.zeros(4), persistent=False)
    model.register_buffer("position_scale", torch.zeros(4), persistent=True)
    return model


def _transformers_model(pinned=(), strict=()):
    """A stand-in for a transformers text encoder, whose two pin lists apply at different dtypes."""

    class PreTrainedModel(torch.nn.Module):
        pass

    PreTrainedModel.__module__ = "transformers.modeling_utils"

    class Encoder(PreTrainedModel):
        _keep_in_fp32_modules = list(pinned)
        _keep_in_fp32_modules_strict = list(strict)

        def __init__(self):
            super().__init__()
            self.wo = torch.nn.Linear(4, 4)
            self.attention = torch.nn.Linear(4, 4)

    return Encoder()


def test_the_cast_preserves_every_pinned_module():
    model = _pinned_model()

    cast_preserving_fp32_modules(model, torch.bfloat16)

    dtypes = {name: tensor.dtype for name, tensor in model.named_parameters()}
    dtypes.update({name: tensor.dtype for name, tensor in model.named_buffers()})
    assert dtypes["blocks.0.norm2.weight"] is torch.float32
    assert dtypes["blocks.0.norm2.bias"] is torch.float32
    assert dtypes["blocks.0.scale_shift_table"] is torch.float32
    assert dtypes["time_embedder.weight"] is torch.float32
    assert dtypes["blocks.0.attn.weight"] is torch.bfloat16
    assert dtypes["position_scale"] is torch.bfloat16


def test_the_cast_leaves_nonpersistent_buffers_and_integers():
    """Neither is in the state dict, so no loader ever casts them either."""

    model = _pinned_model()

    cast_preserving_fp32_modules(model, torch.bfloat16)

    assert model.rope_freqs.dtype is torch.float32
    assert model.token_ids.dtype is torch.long


def test_the_cast_never_rounds_a_pinned_tensor_on_the_way():
    """For a component whose weights are already real, an upcast afterwards could not recover the
    bits a blanket .to(bf16) discarded."""

    model = _pinned_model()
    with torch.no_grad():
        model.blocks[0].norm2.weight.fill_(1.0 + 2.0**-20)
    original = model.blocks[0].norm2.weight.clone()

    cast_preserving_fp32_modules(model, torch.bfloat16)

    assert torch.equal(model.blocks[0].norm2.weight, original)
    assert not torch.equal(
        model.blocks[0].norm2.weight, original.to(torch.bfloat16).float()
    )


def test_a_model_without_a_policy_is_cast_whole():
    model = torch.nn.Linear(4, 4)

    cast_preserving_fp32_modules(model, torch.bfloat16)

    assert model.weight.dtype is torch.bfloat16


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("blocks.0.norm2.weight", True),
        ("norm2", True),
        ("blocks.0.norm2_extra.weight", False),
        ("blocks.0.attn.to_q.weight", False),
    ],
)
def test_pinning_matches_whole_path_segments(name, expected):
    """Substring matching would pin unrelated modules that merely share a prefix."""

    assert keeps_fp32(name, ("norm2", "time_embedder")) is expected


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_a_diffusers_model_pins_at_every_compute_dtype(dtype):
    assert fp32_modules_for(_pinned_model(), dtype) == (
        "time_embedder",
        "scale_shift_table",
        "norm2",
    )


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (torch.float16, ("wo",)),
        (torch.bfloat16, ()),
        (torch.float32, ()),
    ],
)
def test_a_transformers_model_pins_the_plain_list_at_fp16_only(dtype, expected):
    """transformers added _keep_in_fp32_modules to avoid fp16 overflow, so honouring it at bf16 would
    preserve precision its own loader drops."""

    assert fp32_modules_for(_transformers_model(pinned=["wo"]), dtype) == expected


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (torch.float16, ("wo",)),
        (torch.bfloat16, ("wo",)),
        (torch.float32, ()),
    ],
)
def test_a_transformers_model_pins_the_strict_list_at_bf16_too(dtype, expected):
    assert fp32_modules_for(_transformers_model(strict=["wo"]), dtype) == expected


def test_pinned_parameters_are_reported_by_identity_for_fsdp():
    """FSDP takes the parameters themselves as the set to leave out of its units."""

    model = _pinned_model()

    pinned = pinned_fp32_parameters(model, fp32_modules_for(model, torch.bfloat16))

    assert pinned == {
        model.blocks[0].norm2.weight,
        model.blocks[0].norm2.bias,
        model.blocks[0].scale_shift_table,
        model.time_embedder.weight,
        model.time_embedder.bias,
    }
    assert model.blocks[0].attn.weight not in pinned


def test_pinned_parameters_can_be_asked_of_a_submodule():
    """A block's own class carries no policy, but a caller sharding block by block has to ask about
    the block, and has to ask again after a fill rebinds its parameter slots."""

    model = _pinned_model()
    fp32_modules = fp32_modules_for(model, torch.bfloat16)
    block = model.blocks[0]

    assert pinned_fp32_parameters(block, fp32_modules) == {
        block.norm2.weight,
        block.norm2.bias,
        block.scale_shift_table,
    }

    replacement = torch.nn.Parameter(torch.ones(4))
    block.norm2.weight = replacement
    assert replacement in pinned_fp32_parameters(block, fp32_modules)


def test_no_parameters_are_pinned_without_a_policy():
    model = torch.nn.Linear(4, 4)

    assert pinned_fp32_parameters(model, fp32_modules_for(model, torch.bfloat16)) == set()

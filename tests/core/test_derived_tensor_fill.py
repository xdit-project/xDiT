"""Filling a live tensor that no single checkpoint tensor holds.

Ideogram 4's FP8 checkpoint fuses the three attention projections into one stored
weight and keeps each weight's scale beside it, so three live tensors are thirds
of one stored tensor and every weight needs two. The eager path did that
conversion on a whole state dict in memory, which is the thing the blockwise fill
exists to avoid, so the conversion travels with the checkpoint map instead.
"""

import torch

from xfuser.model_executor.models.runner_models.ideogram4 import (
    _dequantize_fp8,
    _dequantize_fp8_chunk,
    _fp8_transformer_manifest,
)
from xfuser.model_executor.models.runner_models.loading.checkpoint import (
    CheckpointManifest,
    DerivedTensor,
)
from xfuser.model_executor.models.runner_models.loading.meta_load import (
    _BlockwiseDiskFiller,
)


class _Shard:
    """Stands in for an open safetensors handle."""

    def __init__(self, tensors):
        self._tensors = tensors

    def get_tensor(self, name):
        return self._tensors[name]


def _filler(manifest, shard):
    filler = _BlockwiseDiskFiller.__new__(_BlockwiseDiskFiller)
    filler.weight_map = manifest.weight_map
    filler.checkpoint_keys = manifest.checkpoint_keys
    filler.derived = manifest.derived
    filler._handle = lambda _path: shard
    return filler


def test_a_fused_weight_becomes_one_projection_at_a_time():
    """Each of q, k and v is a third of the stored weight and a third of its scale."""
    weight = torch.arange(12, dtype=torch.float32).reshape(6, 2).to(torch.float8_e4m3fn)
    scale = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    manifest = CheckpointManifest(
        weight_map={name: "shard" for name in ("to_q", "to_k", "to_v")},
        derived={
            name: DerivedTensor(
                sources=("qkv.weight", "qkv.weight_scale"),
                build=lambda w, s, index=index: _dequantize_fp8_chunk(w, s, index=index),
            )
            for index, name in enumerate(("to_q", "to_k", "to_v"))
        },
    )
    filler = _filler(
        manifest, _Shard({"qkv.weight": weight, "qkv.weight_scale": scale})
    )

    thirds = [filler._tensor_for(name, "shard") for name in ("to_q", "to_k", "to_v")]

    assert [tuple(third.shape) for third in thirds] == [(2, 2)] * 3
    expected = _dequantize_fp8(weight, scale)
    assert torch.equal(torch.cat(thirds), expected)


def test_a_quantized_weight_is_read_with_the_scale_stored_beside_it():
    """A weight read without its scale would be off by a per-row factor."""
    weight = torch.full((2, 2), 2.0).to(torch.float8_e4m3fn)
    scale = torch.tensor([1.0, 4.0])
    manifest = CheckpointManifest(
        weight_map={"linear.weight": "shard"},
        derived={
            "linear.weight": DerivedTensor(
                sources=("linear.weight", "linear.weight_scale"),
                build=_dequantize_fp8,
            )
        },
    )
    filler = _filler(
        manifest, _Shard({"linear.weight": weight, "linear.weight_scale": scale})
    )

    filled = filler._tensor_for("linear.weight", "shard")

    assert filled.dtype is torch.bfloat16
    assert torch.equal(filled, torch.tensor([[2.0, 2.0], [8.0, 8.0]], dtype=torch.bfloat16))


def test_a_tensor_with_no_derivation_is_still_copied_straight_through():
    """Most of the checkpoint is stored the way the model wants it."""
    bias = torch.tensor([1.0, 2.0])
    manifest = CheckpointManifest(weight_map={"linear.bias": "shard"})
    filler = _filler(manifest, _Shard({"linear.bias": bias}))

    assert torch.equal(filler._tensor_for("linear.bias", "shard"), bias)


def test_a_renamed_tensor_is_read_under_its_stored_name():
    manifest = CheckpointManifest(
        weight_map={"attention.to_out.0.weight": "shard"},
        checkpoint_keys={"attention.to_out.0.weight": "attention.o.weight"},
    )
    stored = torch.ones(2)
    filler = _filler(manifest, _Shard({"attention.o.weight": stored}))

    assert torch.equal(filler._tensor_for("attention.to_out.0.weight", "shard"), stored)


def test_the_ideogram_manifest_names_what_the_model_names(monkeypatch):
    """The mapping is built from the checkpoint's header, before any tensor is read."""
    stored = [
        "layers.0.attention.qkv.weight",
        "layers.0.attention.qkv.weight_scale",
        "layers.0.attention.o.weight",
        "layers.0.attention.o.weight_scale",
        "layers.0.attention.norm_q.weight",
    ]
    monkeypatch.setattr(
        "xfuser.model_executor.models.runner_models.ideogram4._shard_paths",
        lambda *args, **kwargs: ["shard"],
    )
    monkeypatch.setattr(
        "xfuser.model_executor.models.runner_models.ideogram4._stored_tensor_names",
        lambda _path: stored,
    )

    manifest = _fp8_transformer_manifest("ideogram-ai/ideogram-4-fp8", "transformer")

    assert set(manifest.weight_map) == {
        "layers.0.attention.to_q.weight",
        "layers.0.attention.to_k.weight",
        "layers.0.attention.to_v.weight",
        "layers.0.attention.to_out.0.weight",
        "layers.0.attention.norm_q.weight",
    }
    # The scales are read as part of the weights they scale, never on their own
    assert not any(name.endswith("_scale") for name in manifest.weight_map)
    assert "layers.0.attention.norm_q.weight" not in manifest.derived


def test_a_denoiser_is_recognised_by_the_declaration_not_by_its_name():
    """Ideogram 4's second denoiser is `unconditional_transformer`.

    Routed by name it took the text-encoder branch, which broadcasts a component
    whole from rank 0 — and rank 0's copy was itself still on meta, so every rank
    ended up with a denoiser of empty weights and the render came out black. Nothing
    raised, which is why this is asserted rather than left to a rendered image.
    """
    from types import SimpleNamespace

    from xfuser.model_executor.models.runner_models.loading.meta_load import (
        ModelLoader,
    )

    loader = ModelLoader.__new__(ModelLoader)
    loader.model = SimpleNamespace(
        load_declaration=SimpleNamespace(
            all_meta_transformers=("transformer", "unconditional_transformer")
        )
    )
    loader.load_declaration = loader.model.load_declaration

    assert loader._is_meta_denoiser("unconditional_transformer")
    assert loader._is_meta_denoiser("transformer")
    # The naming convention still holds for a runner that declares fewer components
    loader.model.load_declaration = SimpleNamespace(all_meta_transformers=())
    loader.load_declaration = loader.model.load_declaration
    assert loader._is_meta_denoiser("transformer_2")
    assert not loader._is_meta_denoiser("text_encoder")


def test_retiring_a_key_is_what_counts_it_as_used():
    """Only the reading rank reads, but every rank retires, and strictness is collective."""
    filler = _BlockwiseDiskFiller.__new__(_BlockwiseDiskFiller)
    filler.weight_map = {"blocks.0.weight": "shard"}
    filler._used_keys = set()

    filler._retire_keys(["blocks.0.weight"])

    assert filler._used_keys == {"blocks.0.weight"}

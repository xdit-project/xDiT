import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from xfuser.model_executor.models.runner_models import vae_manager
from xfuser.model_executor.models.runner_models.base_model import xFuserModel
from xfuser.model_executor.models.runner_models.qwen import (
    xFuserQwenImage21Model,
    xFuserQwenImageEditModel,
    xFuserQwenImageModel,
)
from xfuser.model_executor.models.runner_models.vae_manager import (
    VAEManager,
    fold_single_frame_causal_convs,
    pad_attention_heads,
)

qwenimage_vae = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_qwenimage")
CONVS = (qwenimage_vae.QwenImageCausalConv3d,)
BLOCKS = (qwenimage_vae.QwenImageAttentionBlock,)


def _causal_conv(*args, **kwargs):
    return qwenimage_vae.QwenImageCausalConv3d(*args, **kwargs).double()


class CausalConvs(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.spatial = _causal_conv(4, 6, 3, padding=1)
        self.pointwise = _causal_conv(4, 6, 1)
        self.temporal = _causal_conv(4, 8, (3, 1, 1), padding=(1, 0, 0))
        # Halves time: it is fed a cached frame with the new one, never one frame alone.
        self.time_down = _causal_conv(4, 4, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        self.plain = nn.Conv3d(4, 4, 3, padding=1).double()


FOLDED = ("spatial", "pointwise", "temporal")


def test_one_uncached_frame_takes_the_2d_form_of_a_causal_conv():
    convs = CausalConvs()
    reference = copy.deepcopy(convs)
    assert fold_single_frame_causal_convs(convs, CONVS) == len(FOLDED)

    frame = torch.randn(2, 4, 1, 9, 7, dtype=torch.float64)
    for name in FOLDED:
        folded, expected = getattr(convs, name)(frame), getattr(reference, name)(frame)
        assert folded.shape == expected.shape
        torch.testing.assert_close(folded, expected, rtol=0, atol=1e-12)


def test_several_frames_and_cached_frames_keep_the_3d_conv(monkeypatch):
    convs = CausalConvs()
    reference = copy.deepcopy(convs)
    fold_single_frame_causal_convs(convs, CONVS)
    monkeypatch.setattr(vae_manager.F, "conv2d", lambda *args, **kwargs: pytest.fail("took the 2D path"))

    video = torch.randn(1, 4, 3, 9, 7, dtype=torch.float64)
    frame, cached = video[:, :, 2:], video[:, :, :2]
    for name in ("spatial", "temporal"):
        assert torch.equal(getattr(convs, name)(video), getattr(reference, name)(video))
        assert torch.equal(getattr(convs, name)(frame, cached), getattr(reference, name)(frame, cached))


def test_fold_leaves_other_convs_alone_and_runs_once():
    convs = CausalConvs()
    assert fold_single_frame_causal_convs(convs, CONVS) == len(FOLDED)
    assert fold_single_frame_causal_convs(convs, CONVS) == 0
    assert "forward" not in vars(convs.time_down)
    assert "forward" not in vars(convs.plain)


def test_only_the_named_layer_classes_are_changed():
    convs, blocks = CausalConvs(), AttentionBlocks()
    assert fold_single_frame_causal_convs(convs, ()) == 0
    assert pad_attention_heads(blocks, ()) == 0
    assert not any("forward" in vars(module) for module in (*convs.modules(), *blocks.modules()))


def test_qwen_image_vae_decodes_and_encodes_one_frame_as_before():
    torch.manual_seed(0)
    vae = qwenimage_vae.AutoencoderKLQwenImage(base_dim=8, z_dim=4, num_res_blocks=1).double().eval()
    reference = copy.deepcopy(vae)
    assert fold_single_frame_causal_convs(vae, CONVS) > 0

    latents = torch.randn(1, 4, 1, 4, 4, dtype=torch.float64)
    image = torch.rand(1, 3, 1, 32, 32, dtype=torch.float64) * 2 - 1
    video_latents = torch.randn(1, 4, 2, 4, 4, dtype=torch.float64)
    with torch.no_grad():
        torch.testing.assert_close(vae.decode(latents).sample, reference.decode(latents).sample, rtol=0, atol=1e-10)
        torch.testing.assert_close(
            vae.encode(image).latent_dist.mode(), reference.encode(image).latent_dist.mode(), rtol=0, atol=1e-10
        )
        torch.testing.assert_close(
            vae.decode(video_latents).sample, reference.decode(video_latents).sample, rtol=0, atol=1e-10
        )


def _manager(monkeypatch, use_parallel_vae, enable_tiling, tile_parallel):
    monkeypatch.setattr(vae_manager, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(vae_manager.vae_tiling, "supports_tile_parallel", lambda vae: tile_parallel)
    config = SimpleNamespace(use_parallel_vae=use_parallel_vae, enable_tiling=enable_tiling)
    return VAEManager(config=config, capabilities=object(), settings=object())


# (use_parallel_vae, enable_tiling, tile_parallel) and whether the VAE then decodes with its own layers.
DECODE_PATHS = [
    ((False, False, True), True),
    ((False, True, True), True),
    # Each rank decodes whole tiles.
    ((True, True, True), True),
    # The parallel VAE shards rows: its adapter runs the decoder with its own convolutions.
    ((True, False, True), False),
    ((True, True, False), False),
]


@pytest.mark.parametrize(("path", "own_layers"), DECODE_PATHS)
def test_manager_folds_the_decoder_where_the_vae_decodes_with_its_own_layers(monkeypatch, path, own_layers):
    vae = SimpleNamespace(decoder=CausalConvs(), encoder=CausalConvs())
    manager = _manager(monkeypatch, *path)
    manager.fold_single_frame_convs([vae], CONVS)
    folded = len(FOLDED) if own_layers else 0
    assert sum("forward" in vars(module) for module in vae.decoder.modules()) == folded
    # An edited image follows the last bits of its input latents: the encoder stays as it is.
    assert not any("forward" in vars(module) for module in vae.encoder.modules())


@pytest.mark.parametrize("runner", [xFuserQwenImageModel, xFuserQwenImageEditModel])
def test_qwen_image_runners_prepare_their_vae_with_the_other_options(runner):
    events = []
    model = object.__new__(runner)
    model.config = SimpleNamespace()
    model._decoding_vaes = lambda: ["vae"]
    model._enable_cpu_offload = lambda: None
    model._vae_manager = SimpleNamespace(
        fold_single_frame_convs=lambda vaes, types: events.append(("fold", vaes, types)),
        pad_attention_heads=lambda vaes, types: events.append(("pad", vaes, types)),
        enable_options=lambda vaes: events.append(("options", vaes)),
    )

    model._enable_options()

    assert events == [("fold", ["vae"], CONVS), ("pad", ["vae"], BLOCKS), ("options", ["vae"])]


class AttentionBlocks(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.wide = qwenimage_vae.QwenImageAttentionBlock(384).double()
        self.narrow = qwenimage_vae.QwenImageAttentionBlock(64).double()


def test_a_padded_attention_head_attends_as_before():
    blocks = AttentionBlocks()
    reference = copy.deepcopy(blocks)
    assert pad_attention_heads(blocks, BLOCKS) == 1
    assert pad_attention_heads(blocks, BLOCKS) == 0
    assert "forward" not in vars(blocks.narrow)

    # Two frames: the block attends within each frame.
    x = torch.randn(1, 384, 2, 5, 3, dtype=torch.float64)
    with torch.no_grad():
        padded, expected = blocks.wide(x), reference.wide(x)
    assert padded.shape == expected.shape
    torch.testing.assert_close(padded, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize(("path", "own_layers"), DECODE_PATHS)
def test_manager_pads_attention_heads_where_the_vae_decodes_with_its_own_layers(monkeypatch, path, own_layers):
    monkeypatch.setattr(vae_manager, "device_arch", lambda: "gfx950:sramecc+:xnack-")
    blocks = AttentionBlocks()
    manager = _manager(monkeypatch, *path)
    manager.pad_attention_heads([blocks], BLOCKS)
    assert sum("forward" in vars(module) for module in blocks.modules()) == int(own_layers)


# Measured on gfx950 only.
@pytest.mark.parametrize("arch", ["gfx942:sramecc+:xnack-", ""])
def test_manager_pads_attention_heads_on_gfx950_only(monkeypatch, arch):
    monkeypatch.setattr(vae_manager, "device_arch", lambda: arch)
    blocks = AttentionBlocks()
    manager = _manager(monkeypatch, use_parallel_vae=False, enable_tiling=False, tile_parallel=True)
    manager.pad_attention_heads([blocks], BLOCKS)
    assert not any("forward" in vars(module) for module in blocks.modules())


def test_qwen_image_21_keeps_its_2d_vae_as_it_is():
    assert "_enable_options" not in vars(xFuserQwenImage21Model)
    assert xFuserQwenImage21Model._enable_options is xFuserModel._enable_options

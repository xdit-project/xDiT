from types import SimpleNamespace

import torch

from xfuser.model_executor.models.transformers import transformer_z_image
from xfuser.model_executor.models.transformers.transformer_z_image import (
    xFuserZImagePipeFusionTransformerWrapper,
    z_image_pipefusion_patch_major,
)


def test_z_image_pipefusion_reorders_and_pads_image_kv_by_height_stripe(monkeypatch):
    monkeypatch.setattr(transformer_z_image, "SEQ_MULTI_OF", 4)
    tensor = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1, 1)

    result = z_image_pipefusion_patch_major(
        tensor,
        grid=(2, 3),
        patch_rows=(1, 1),
    )

    assert result.flatten().tolist() == [0, 1, 2, 2, 3, 4, 5, 5]


def test_z_image_attention_metadata_reaches_layers_inside_cache_wrappers(monkeypatch):
    attention = torch.nn.Identity()
    block = torch.nn.Module()
    block.attention = attention
    layers = torch.nn.ModuleList([torch.nn.Sequential(block)])
    transformer = SimpleNamespace()
    monkeypatch.setattr(
        transformer_z_image,
        "get_runtime_state",
        lambda: SimpleNamespace(pp_patches_height=(4, 4)),
    )

    xFuserZImagePipeFusionTransformerWrapper._set_image_attention_metadata(
        transformer,
        layers,
        image_tokens=8,
        x_size=[(1, 8, 4)],
        patch_size=2,
    )

    assert attention._xfuser_z_image_tokens == 8
    assert attention._xfuser_z_image_grid == (4, 2)
    assert attention._xfuser_z_image_patch_rows == (2, 2)


def test_z_image_patch_geometry_slices_rope_to_padded_hidden_lengths():
    image = torch.zeros(8, 4)
    caption = torch.zeros(32, 4)
    transformer = SimpleNamespace(
        patchify_and_embed=lambda *_args: (
            [image],
            [caption],
            [(1, 8, 4)],
            [torch.ones(8, dtype=torch.bool)],
            [torch.ones(32, dtype=torch.bool)],
            [torch.zeros(10, 3, dtype=torch.long)],
            [torch.zeros(53, 3, dtype=torch.long)],
        ),
        rope_embedder=lambda position_ids: torch.zeros(len(position_ids), 2),
    )

    *_, image_freqs, caption_freqs = xFuserZImagePipeFusionTransformerWrapper._prepare_patch_geometry(
        transformer,
        [image],
        [caption],
        patch_size=2,
        f_patch_size=1,
        patch_start_height=0,
    )

    assert image_freqs.shape[1] == len(image)
    assert caption_freqs.shape[1] == len(caption)


def test_z_image_context_refiner_is_reused_for_the_request():
    calls = []

    def embed(items, *_args):
        calls.append("embed")
        return items[0]

    def refine(hidden, _mask, _freqs):
        calls.append("refine")
        return hidden + 1

    transformer = SimpleNamespace(
        _pipefusion_context_cache=None,
        _embed_padded=embed,
        cap_embedder=None,
        cap_pad_token=None,
        context_refiner=(refine,),
    )
    mask = torch.ones(1, 2, dtype=torch.bool)
    freqs = torch.zeros(1, 2, 2)

    first = xFuserZImagePipeFusionTransformerWrapper._refine_context(
        transformer,
        [torch.ones(1, 2)],
        [mask[0]],
        mask,
        freqs,
    )
    second = xFuserZImagePipeFusionTransformerWrapper._refine_context(
        transformer,
        [torch.full((1, 2), 9.0)],
        [mask[0]],
        mask,
        freqs,
    )

    assert calls == ["embed", "refine"]
    assert first is second
    assert first[0].tolist() == [[2.0, 2.0]]


def test_z_image_compile_capture_propagates_list_and_stage_shapes(monkeypatch):
    transformer = SimpleNamespace(
        config=SimpleNamespace(dim=24, out_channels=4),
        _padded_tokens=xFuserZImagePipeFusionTransformerWrapper._padded_tokens,
    )
    images = [
        torch.zeros(4, 1, 4, 4),
        torch.zeros(4, 1, 4, 4),
    ]
    captions = [torch.zeros(5, 8), torch.zeros(7, 8)]
    timestep = torch.zeros(2)

    monkeypatch.setattr(transformer_z_image, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(transformer_z_image, "is_pipeline_last_stage", lambda: False)
    first_output = xFuserZImagePipeFusionTransformerWrapper.pipefusion_compile_capture_forward(
        transformer,
        images,
        timestep,
        captions,
        return_dict=False,
    )[0]

    monkeypatch.setattr(transformer_z_image, "is_pipeline_first_stage", lambda: False)
    monkeypatch.setattr(transformer_z_image, "is_pipeline_last_stage", lambda: True)
    last_output = xFuserZImagePipeFusionTransformerWrapper.pipefusion_compile_capture_forward(
        transformer,
        images,
        timestep,
        captions,
        pipeline_hidden_states=torch.zeros(2, 64, 24),
        return_dict=False,
    )[0]

    assert first_output.shape == (2, 64, 24)
    assert [tuple(value.shape) for value in last_output] == [
        (4, 1, 4, 4),
        (4, 1, 4, 4),
    ]

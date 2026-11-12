"""SD3 PipeFusion patches must reuse patch 0's per-block text input.

During a PipeFusion step every patch after the first feeds each block the text
stream that patch 0 fed that block. Cache-DiT replaces ``transformer_blocks``
with a single module that runs the real blocks itself, so a substitution done
only by the transformer's own block loop reached the first block alone and the
later blocks received each patch's own running text.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers.models.transformers.transformer_sd3")

from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel

from xfuser.model_executor.models.transformers import base_transformer, transformer_sd3

NUM_LAYERS = 3
LATENT_WIDTH = 8
PATCH_ROWS = (4, 4)


class _BlocksRunner(torch.nn.Module):
    """Runs the blocks internally, as Cache-DiT's cached-blocks module does."""

    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, hidden_states, encoder_hidden_states, **kwargs):
        for block in self.blocks:
            encoder_hidden_states, hidden_states = block(hidden_states, encoder_hidden_states, **kwargs)
        return encoder_hidden_states, hidden_states


@pytest.fixture
def state(monkeypatch):
    runtime_state = SimpleNamespace(
        patch_mode=False,
        pipeline_patch_idx=0,
        backbone_patch_size=2,
        vae_scale_factor=1,
        input_config=SimpleNamespace(width=LATENT_WIDTH),
        pp_patches_height=list(PATCH_ROWS),
    )
    # Layer wrapping needs initialized process groups, so the base conversion
    # sees one process while the SD3 wrapper sees two pipeline stages.
    for name in (
        "get_pipeline_parallel_world_size",
        "get_sequence_parallel_world_size",
        "get_tensor_model_parallel_world_size",
    ):
        monkeypatch.setattr(base_transformer, name, lambda: 1)
    monkeypatch.setattr(base_transformer, "get_fast_attn_enable", lambda: False)
    monkeypatch.setattr(base_transformer, "get_runtime_state", lambda: runtime_state)
    monkeypatch.setattr(transformer_sd3, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(transformer_sd3, "get_runtime_state", lambda: runtime_state)
    monkeypatch.setattr(transformer_sd3, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(transformer_sd3, "is_pipeline_last_stage", lambda: True)
    return runtime_state


def _tiny_model():
    torch.manual_seed(0)
    return SD3Transformer2DModel(
        sample_size=LATENT_WIDTH,
        patch_size=2,
        in_channels=4,
        num_layers=NUM_LAYERS,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        caption_projection_dim=16,
        pooled_projection_dim=8,
        out_channels=4,
        pos_embed_max_size=16,
    ).eval()


def _record_text_inputs(blocks, received):
    def hook(index):
        def record(module, args, kwargs):
            received[-1][index] = args[1] if len(args) > 1 else kwargs["encoder_hidden_states"]

        return record

    for index, block in enumerate(blocks):
        block.register_forward_pre_hook(hook(index), with_kwargs=True)


@pytest.mark.parametrize("cache_dit_blocks", [False, True], ids=["block_list", "cache_dit_runner"])
def test_later_patch_blocks_receive_patch_zero_text_input(state, cache_dit_blocks):
    model = _tiny_model()
    real_blocks = list(model.transformer_blocks)
    received = []
    _record_text_inputs(real_blocks, received)

    wrapper = transformer_sd3.xFuserSD3Transformer2DWrapper(model)
    if cache_dit_blocks:
        model.transformer_blocks = torch.nn.ModuleList([_BlocksRunner(model.transformer_blocks)])
    wrapper.reset_pipefusion_state()

    generator = torch.Generator().manual_seed(0)
    text = torch.randn(1, 3, 16, generator=generator)
    pooled = torch.randn(1, 8, generator=generator)
    state.patch_mode = True
    with torch.no_grad():
        for patch_index, rows in enumerate(PATCH_ROWS):
            state.pipeline_patch_idx = patch_index
            received.append([None] * NUM_LAYERS)
            wrapper(
                hidden_states=torch.randn(1, 4, rows, LATENT_WIDTH, generator=generator),
                encoder_hidden_states=text,
                pooled_projections=pooled,
                timestep=torch.tensor([1.0]),
                return_dict=False,
            )

    patch_zero, patch_one = received
    assert all(tensor is not None for tensor in patch_zero)
    for index in range(NUM_LAYERS):
        assert patch_one[index] is patch_zero[index], f"block {index} received patch 1's own text"

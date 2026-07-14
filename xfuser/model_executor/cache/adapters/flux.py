"""
xDiT in-tree cache for Flux1 (FluxTransformer2DModel).
USP-safe: all_reduce in l1_distance synchronises skip decision across SP ranks.
"""
from torch import nn

from xfuser.model_executor.cache import utils


def apply_teacache(
    transformer,
    *,
    rel_l1_thresh=0.12,
    return_hidden_states_first=False,
    num_steps=8,
):
    """Apply TeaCache to a Flux1 transformer. USP-safe via all_reduce in l1_distance."""
    cached_blocks = utils.TeaCachedTransformerBlocks(
        transformer.transformer_blocks,
        transformer.single_transformer_blocks,
        transformer=transformer,
        rel_l1_thresh=rel_l1_thresh,
        return_hidden_states_first=return_hidden_states_first,
        num_steps=num_steps,
        name="flux",
    )
    # Permanently replace block lists; same pattern as flux2.py.
    # Original blocks are captured inside TeaCachedTransformerBlocks.
    transformer.transformer_blocks = nn.ModuleList([cached_blocks])
    transformer.single_transformer_blocks = nn.ModuleList()
    return transformer

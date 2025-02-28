"""
adapted from https://github.com/ali-vilab/TeaCache.git
adapted from https://github.com/chengzeyi/ParaAttention.git
"""
import functools
import unittest

import torch
from torch import nn

from xfuser.model_executor.cache import utils

def create_cached_transformer_blocks(transformer, rel_l1_thresh, return_hidden_first, return_hidden_only, num_steps, dist):
    cached_transformer_class = utils.FBCachedTransformerBlocks

    return cached_transformer_class(
        transformer.blocks,
        None,
        transformer=transformer,
        rel_l1_thresh=rel_l1_thresh,
        return_hidden_first=return_hidden_first,
        return_hidden_only=return_hidden_only,
        num_steps=num_steps,
        dist=dist,
    )


def apply_cache_on_transformer(
    transformer,
    *,
    rel_l1_thresh=0.6,
    dist="l1",
):
    cached_transformer_blocks = nn.ModuleList([
        utils.FBCachedTransformerBlocks(
        transformer.blocks,
        None,
        transformer=transformer,
        rel_l1_thresh=rel_l1_thresh,
        return_hidden_first=False,
        return_hidden_only=True,
        num_steps=-1,
        dist=dist,
    )])

    original_forward = transformer.forward

    @functools.wraps(original_forward)
    def new_forward(
        self,
        *args,
        **kwargs,
    ):
        with unittest.mock.patch.object(
            self,
            "blocks",
            cached_transformer_blocks,
        ):
            return original_forward(
                *args,
                **kwargs,
            )

    transformer.forward = new_forward.__get__(transformer)

    return transformer


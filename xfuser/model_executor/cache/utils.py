"""
adapted from https://github.com/ali-vilab/TeaCache.git
adapted from https://github.com/chengzeyi/ParaAttention.git
"""
import dataclasses
from typing import Dict, Optional, List
from xfuser.core.distributed import (
    get_sp_group,
    get_sequence_parallel_world_size,
)

import torch
from torch.nn import Module
from abc import ABC, abstractmethod


# --------- CacheContext --------- #
class CacheContext(Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("default_coef", torch.tensor([1.0, 0.0]).cuda())
        self.register_buffer("flux_coef", torch.tensor([498.651651, -283.781631, 55.8554382, -3.82021401, 0.264230861]).cuda())
        
        self.register_buffer("original_hidden", None, persistent=False)
        self.register_buffer("original_encoder", None, persistent=False)
        self.register_buffer("hidden_residual", None, persistent=False)
        self.register_buffer("encoder_residual", None, persistent=False)
        self.register_buffer("modulated_inputs", None, persistent=False)

    def get_coef(self, name: str) -> torch.Tensor:
        return getattr(self, f"{name}_coef")

#---------  CacheCallback  ---------#
@dataclasses.dataclass
class CacheState:
    transformer: Optional[torch.nn.Module] = None
    transformer_blocks: Optional[List[torch.nn.Module]] = None
    single_transformer_blocks: Optional[List[torch.nn.Module]] = None
    cache_context: Optional[CacheContext] = None
    rel_l1_thresh: float = 0.6
    return_hidden_first: bool = False
    return_hidden_only: bool = False
    use_cache: torch.Tensor = torch.tensor(False, dtype=torch.bool)
    num_steps: int = 8
    name: str = "default"


class CacheCallback:
    def on_init_end(self, state: CacheState, **kwargs): pass
    def on_forward_begin(self, state: CacheState, **kwargs): pass
    def on_forward_remaining_begin(self, state: CacheState, **kwargs): pass
    def on_forward_end(self, state: CacheState, **kwargs): pass


class CallbackHandler(CacheCallback):
    def __init__(self, callbacks: Optional[List[CacheCallback]] = None):
        self.callbacks = list(callbacks) if callbacks else []

    def trigger_event(self, event: str, state: CacheState):
        for cb in self.callbacks:
            getattr(cb, event)(state)

# --------- Vectorized Poly1D --------- #
class VectorizedPoly1D(Module):
    def __init__(self, coefficients: torch.Tensor):
        super().__init__()
        self.register_buffer("coefficients", coefficients)
        self.degree = len(coefficients) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(x)
        for i, coef in enumerate(self.coefficients):
            result += coef * (x ** (self.degree - i))
        return result


class CachedTransformerBlocks(torch.nn.Module, ABC):
    def __init__(
        self,
        transformer_blocks: List[Module],
        single_transformer_blocks: Optional[List[Module]] = None,
        *,
        transformer: Optional[Module] = None,
        rel_l1_thresh: float = 0.6,
        return_hidden_first: bool = True,
        return_hidden_only: bool = True,
        num_steps: int = -1,
        dist: str = "default",
        name: str = "default",
        callbacks: Optional[List[CacheCallback]] = None,
    ):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(transformer_blocks)
        self.single_transformer_blocks = torch.nn.ModuleList(single_transformer_blocks) if single_transformer_blocks else None
        self.transformer = transformer
        self.register_buffer("cnt", torch.tensor(0).cuda())
        self.register_buffer("accumulated_rel_l1_distance", torch.tensor([0.0]).cuda())
        self.register_buffer("use_cache", torch.tensor(False, dtype=torch.bool).cuda())

        self.cache_context = CacheContext()
        self.callback_handler = CallbackHandler(callbacks)

        self.rel_l1_thresh = torch.tensor(rel_l1_thresh).cuda()
        self.return_hidden_first = return_hidden_first
        self.return_hidden_only = return_hidden_only
        self.num_steps = num_steps
        self.name = name
        self.distance_functions = {
            "l1": self.l1_distance,
            "l2": self.l2_distance,
            "default": self.l1_distance,
        }
        self.cacl_dist = self._get_distance_function(dist)
        self.callback_handler.trigger_event("on_init_begin", self)

    def _get_distance_function(self, dist: str):
        return self.distance_functions.get(dist, self.l1_distance)  # 默认使用 L1 距离

    @property
    def is_parallelized(self) -> bool:
        return get_sequence_parallel_world_size() > 1

    def all_reduce(self, input_: torch.Tensor, op=torch.distributed.ReduceOp.AVG) -> torch.Tensor:
        try:
            return get_sp_group().all_reduce(input_, op=op) if self.is_parallelized else input_
        except:
            return input_

    def l1_distance(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        diff = (t1 - t2).abs().mean()
        norm = t1.abs().mean() + t2.abs().mean()
        diff, norm = self.all_reduce(diff.unsqueeze(0)), self.all_reduce(norm.unsqueeze(0))
        return (diff / norm).squeeze()

    def l2_distance(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        squared_diff = ((t1-t2)**2).sum() / (t1**2 + t2**2).sum()
        squared_diff = self.all_reduce(squared_diff.unsqueeze(0))
        l2_dist = torch.sqrt(squared_diff)
        return l2_dist.squeeze()

    @abstractmethod
    def are_two_tensor_similar(self, t1: torch.Tensor, t2: torch.Tensor, threshold: float) -> torch.Tensor: pass

    @abstractmethod
    def get_start_idx(self) -> int: pass

    @abstractmethod
    def get_modulated_inputs(self, hidden: torch.Tensor, encoder: torch.Tensor, *args, **kwargs): pass

    def process_blocks(self, start_idx: int, hidden: torch.Tensor, encoder: torch.Tensor, *args, **kwargs):
        for block in self.transformer_blocks[start_idx:]:
            if not self.return_hidden_only:
                hidden, encoder = block(hidden, encoder, *args, **kwargs)
                hidden, encoder = (hidden, encoder) if self.return_hidden_first else (encoder, hidden)
            else:
                hidden = block(hidden, encoder, *args, **kwargs)

        if self.single_transformer_blocks:
            if not self.return_hidden_only:
                hidden = torch.cat([encoder, hidden], dim=1)
            for block in self.single_transformer_blocks:
                hidden = block(hidden, *args, **kwargs)
            if not self.return_hidden_only:
                encoder, hidden = hidden.split([encoder.shape[1], hidden.shape[1] - encoder.shape[1]], dim=1)

        self.cache_context.hidden_residual = hidden - self.cache_context.original_hidden
        if not self.return_hidden_only:
            self.cache_context.encoder_residual = encoder - self.cache_context.original_encoder
        return (hidden, encoder) if not self.return_hidden_only else hidden

    def forward(self, hidden, encoder, *args, **kwargs):
        self.callback_handler.trigger_event("on_forward_begin", self)

        modulated, prev_modulated, orig_hidden, orig_encoder = \
            self.get_modulated_inputs(hidden, encoder, *args, **kwargs)

        self.cache_context.original_hidden = orig_hidden
        self.cache_context.original_encoder = orig_encoder

        self.use_cache = self.are_two_tensor_similar(prev_modulated, modulated, self.rel_l1_thresh) \
            if prev_modulated is not None else torch.tensor(False, dtype=torch.bool)

        self.callback_handler.trigger_event("on_forward_remaining_begin", self)
        if self.use_cache:
            hidden = hidden + self.cache_context.hidden_residual
            if not self.return_hidden_only:
                encoder = encoder + self.cache_context.encoder_residual
        else:
            if not self.return_hidden_only:
                hidden, encoder = self.process_blocks(self.get_start_idx(), orig_hidden, orig_encoder, *args, **kwargs)
            else:
                hidden = self.process_blocks(self.get_start_idx(), orig_hidden, encoder, *args, **kwargs)

        self.callback_handler.trigger_event("on_forward_end", self)
        return ((hidden, encoder) if self.return_hidden_first else (encoder, hidden)) if not self.return_hidden_only else hidden


class FBCachedTransformerBlocks(CachedTransformerBlocks):
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        rel_l1_thresh=0.6,
        return_hidden_first=True,
        return_hidden_only: bool = True,
        num_steps=-1,
        dist="default",
        name="default",
        callbacks: Optional[List[CacheCallback]] = None,
    ):
        super().__init__(transformer_blocks,
                       single_transformer_blocks=single_transformer_blocks,
                       transformer=transformer,
                       rel_l1_thresh=rel_l1_thresh,
                       num_steps=num_steps,
                       return_hidden_first=return_hidden_first,
                       return_hidden_only=return_hidden_only,
                       dist=dist,
                       name=name,
                       callbacks=callbacks)

    def get_start_idx(self) -> int:
        return 1

    def are_two_tensor_similar(self, t1: torch.Tensor, t2: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        similarity = self.cacl_dist(t1, t2)
        return similarity < threshold

    def get_modulated_inputs(self, hidden, encoder, *args, **kwargs):
        original_hidden = hidden
        first_transformer_block = self.transformer_blocks[0]
        if not self.return_hidden_only:
            hidden, encoder = first_transformer_block(hidden, encoder, *args, **kwargs)
        else:
            hidden = first_transformer_block(hidden, encoder, *args, **kwargs)
        first_hidden_residual = hidden - original_hidden
        prev_first_hidden_residual = self.cache_context.modulated_inputs
        if not self.use_cache:
           self.cache_context.modulated_inputs = first_hidden_residual

        return (first_hidden_residual, prev_first_hidden_residual, hidden, encoder) if not self.return_hidden_only else \
            (first_hidden_residual, prev_first_hidden_residual, hidden, None)


class TeaCachedTransformerBlocks(CachedTransformerBlocks):
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        rel_l1_thresh=0.6,
        return_hidden_first=True,
        return_hidden_only: bool = True,
        num_steps=-1,
        dist="default",
        name="default",
        callbacks: Optional[List[CacheCallback]] = None,
    ):
        super().__init__(transformer_blocks,
                       single_transformer_blocks=single_transformer_blocks,
                       transformer=transformer,
                       rel_l1_thresh=rel_l1_thresh,
                       num_steps=num_steps,
                       return_hidden_first=return_hidden_first,
                       return_hidden_only=return_hidden_only,
                       dist=dist,
                       name=name,
                       callbacks=callbacks)
        self.rescale_func = VectorizedPoly1D(self.cache_context.get_coef(self.name))

    def get_start_idx(self) -> int:
        return 0

    def are_two_tensor_similar(self, t1: torch.Tensor, t2: torch.Tensor, threshold: float) -> torch.Tensor:
        diff = self.cacl_dist(t1, t2)
        new_accum = self.accumulated_rel_l1_distance + self.rescale_func(diff)
        reset_mask = (self.cnt == 0) or (self.cnt == self.num_steps - 1)
        self.use_cache = torch.logical_and(new_accum < threshold, torch.logical_not(reset_mask))
        self.accumulated_rel_l1_distance[0] = torch.where(self.use_cache, new_accum[0], 0.0)
        self.cnt = torch.where(self.cnt + 1 < self.num_steps, self.cnt + 1, 0)

        return self.use_cache

    def get_modulated_inputs(self, hidden, encoder, *args, **kwargs):
        inp = hidden.clone()
        temb_ = kwargs.get("temb", None).clone()
        modulated, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.transformer_blocks[0].norm1(inp, emb=temb_)
        prev_modulated = self.cache_context.modulated_inputs
        self.cache_context.modulated_inputs = modulated
        return modulated, prev_modulated, hidden, encoder
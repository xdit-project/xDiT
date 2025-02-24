"""
adapted from https://github.com/ali-vilab/TeaCache.git
adapted from https://github.com/chengzeyi/ParaAttention.git
"""
import contextlib
import dataclasses
from collections import defaultdict
from typing import DefaultDict, Dict, Optional, List
from xfuser.core.distributed import (
    get_sp_group,
    get_sequence_parallel_world_size,
)

import torch
from abc import ABC, abstractmethod


#---------  CacheCallback  ---------#
@dataclasses.dataclass
class CacheState:
    transformer: None
    transformer_blocks: None
    single_transformer_blocks: None
    cache_context: None
    rel_l1_thresh: int = 0.6
    return_hidden_states_first: bool = True
    use_cache: bool = False
    num_steps: int = 8
    name: str = "default"


class CacheCallback:
    def on_init_end(self, state: CacheState, **kwargs):
        pass

    def on_forward_begin(self, state: CacheState, **kwargs):
        pass

    def on_forward_remaining_begin(self, state: CacheState, **kwargs):
        pass

    def on_forward_end(self, state: CacheState, **kwargs):
        pass


class CallbackHandler(CacheCallback):
    def __init__(self, callbacks):
        self.callbacks = []
        if callbacks is not None:
            for cb in callbacks:
                self.add_callback(cb)

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        self.callbacks.append(cb)

    def pop_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    def remove_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    def on_init_end(self, state: CacheState):
        return self.call_event("on_init_end", state)

    def on_forward_begin(self, state: CacheState):
        return self.call_event("on_forward_begin", state)

    def on_forward_remaining_begin(self, state: CacheState):
        return self.call_event("on_forward_remaining_begin", state)
    
    def on_forward_end(self, state: CacheState):
        return self.call_event("on_forward_end", state)

    def call_event(self, event, state, **kwargs):
        for callback in self.callbacks:
            getattr(callback, event)(
                state,
                **kwargs,
            )


#---------  CacheContext  ---------#
@dataclasses.dataclass
class CacheContext:
    buffers: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    coefficients: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.coefficients["default"] = torch.Tensor([1, 0]).cuda()
        self.coefficients["flux"] = torch.Tensor([4.98651651e+02, -2.83781631e+02,  5.58554382e+01, -3.82021401e+00, 2.64230861e-01]).cuda()

    def get_coef(self, name):
        return self.coefficients.get(name)

    def get_buffer(self, name):
        return self.buffers.get(name)

    def set_buffer(self, name, buffer):
        self.buffers[name] = buffer

    def clear_buffer(self):
        self.buffers.clear()


#---------  torch version of poly1d  ---------#
class TorchPoly1D:
    def __init__(self, coefficients):
        self.coefficients = torch.tensor(coefficients, dtype=torch.float32)
        self.degree = len(coefficients) - 1

    def __call__(self, x):
        result = torch.zeros_like(x)
        for i, coef in enumerate(self.coefficients):
            result += coef * (x ** (self.degree - i))
        return result


class CachedTransformerBlocks(torch.nn.Module, ABC):
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        rel_l1_thresh=0.6,
        return_hidden_states_first=True,
        num_steps=-1,
        name="default",
        callbacks: Optional[List[CacheCallback]] = None,
    ):
        super().__init__()
        self.state = CacheState(
            transformer=transformer,
            transformer_blocks=transformer_blocks,
            single_transformer_blocks=single_transformer_blocks,
            cache_context=CacheContext(),
        )
        self.state.rel_l1_thresh = rel_l1_thresh
        self.state.return_hidden_states_first = return_hidden_states_first
        self.state.use_cache = False
        self.state.num_steps=num_steps
        self.state.name=name
        self.callback_handler = CallbackHandler(callbacks)
        self.callback_handler.on_init_end(self.state)

    def is_parallelized(self):
        if get_sequence_parallel_world_size() > 1:
            return True
        return False

    def all_reduce(self, input_, op):
        if get_sequence_parallel_world_size() > 1:
            return get_sp_group().all_reduce(input_=input_, op=op)
        raise NotImplementedError("Cache method not support parrellism other than sp")

    def l1_distance_two_tensor(self, t1, t2):
        mean_diff = (t1 - t2).abs().mean()
        mean_t1 = t1.abs().mean()
        if self.is_parallelized():
            mean_diff = self.all_reduce(mean_diff.unsqueeze(0), op=torch._C._distributed_c10d.ReduceOp.AVG)[0]
            mean_t1 = self.all_reduce(mean_t1.unsqueeze(0), op=torch._C._distributed_c10d.ReduceOp.AVG)[0]
        diff = mean_diff / mean_t1
        return diff

    @abstractmethod
    def are_two_tensor_similar(self, t1, t2, threshold):
        pass

    def run_one_block_transformer(self, block, hidden_states, encoder_hidden_states, *args, **kwargs):
        hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, *args, **kwargs)
        return (
            (hidden_states, encoder_hidden_states)
            if self.state.return_hidden_states_first
            else (encoder_hidden_states, hidden_states)
        )

    @abstractmethod
    def get_start_idx(self):
        pass

    def get_remaining_block_result(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        original_hidden_states = self.state.cache_context.get_buffer("original_hidden_states")
        original_encoder_hidden_states = self.state.cache_context.get_buffer("original_encoder_hidden_states")
        start_idx = self.get_start_idx()
        if start_idx == -1:
            return (hidden_states, encoder_hidden_states)
        for block in self.state.transformer_blocks[start_idx:]:
            hidden_states, encoder_hidden_states = \
                self.run_one_block_transformer(block, hidden_states, encoder_hidden_states, *args, **kwargs)
        if self.state.single_transformer_blocks is not None:
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            for block in self.state.single_transformer_blocks:
                hidden_states = block(hidden_states, *args, **kwargs)
            encoder_hidden_states, hidden_states = hidden_states.split(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
        self.state.cache_context.set_buffer("hidden_states_residual", hidden_states - original_hidden_states)
        self.state.cache_context.set_buffer("encoder_hidden_states_residual",
                                      encoder_hidden_states - original_encoder_hidden_states)
        return (hidden_states, encoder_hidden_states)

    @abstractmethod
    def get_modulated_inputs(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        pass

    def forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        self.callback_handler.on_forward_begin(self.state)
        modulated_inputs, prev_modulated_inputs, original_hidden_states, original_encoder_hidden_states = \
            self.get_modulated_inputs(hidden_states, encoder_hidden_states, *args, **kwargs)

        self.state.cache_context.set_buffer("original_hidden_states", original_hidden_states)
        self.state.cache_context.set_buffer("original_encoder_hidden_states", original_encoder_hidden_states)

        self.state.use_cache = prev_modulated_inputs is not None and self.are_two_tensor_similar(
            t1=prev_modulated_inputs, t2=modulated_inputs, threshold=self.state.rel_l1_thresh)

        self.callback_handler.on_forward_remaining_begin(self.state)
        if self.state.use_cache:
            hidden_states += self.state.cache_context.get_buffer("hidden_states_residual")
            encoder_hidden_states += self.state.cache_context.get_buffer("encoder_hidden_states_residual")
        else:
            hidden_states, encoder_hidden_states = self.get_remaining_block_result(
                original_hidden_states, original_encoder_hidden_states, *args, **kwargs)

        self.callback_handler.on_forward_end(self.state)
        return (
            (hidden_states, encoder_hidden_states)
            if self.state.return_hidden_states_first
            else (encoder_hidden_states, hidden_states)
        )


class FBCachedTransformerBlocks(CachedTransformerBlocks):
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        rel_l1_thresh=0.6,
        return_hidden_states_first=True,
        num_steps=-1,
        name="default",
        callbacks: Optional[List[CacheCallback]] = None,
    ):
        super().__init__(transformer_blocks,
                       single_transformer_blocks=single_transformer_blocks,
                       transformer=transformer,
                       rel_l1_thresh=rel_l1_thresh,
                       num_steps=num_steps,
                       return_hidden_states_first=return_hidden_states_first,
                       name=name,
                       callbacks=callbacks)

    def get_start_idx(self):
        return 1

    def are_two_tensor_similar(self, t1, t2, threshold):
        return self.l1_distance_two_tensor(t1, t2) < threshold

    def get_modulated_inputs(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        original_hidden_states = hidden_states
        first_transformer_block = self.state.transformer_blocks[0]
        hidden_states, encoder_hidden_states = \
            self.run_one_block_transformer(first_transformer_block, hidden_states, encoder_hidden_states, *args, **kwargs)
        first_hidden_states_residual = hidden_states - original_hidden_states
        prev_first_hidden_states_residual = self.state.cache_context.get_buffer("modulated_inputs")
        self.state.cache_context.set_buffer("modulated_inputs", first_hidden_states_residual)

        return first_hidden_states_residual, prev_first_hidden_states_residual, hidden_states, encoder_hidden_states


class TeaCachedTransformerBlocks(CachedTransformerBlocks):
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        rel_l1_thresh=0.6,
        return_hidden_states_first=True,
        num_steps=-1,
        name="default",
        callbacks: Optional[List[CacheCallback]] = None,
    ):
        super().__init__(transformer_blocks,
                       single_transformer_blocks=single_transformer_blocks,
                       transformer=transformer,
                       rel_l1_thresh=rel_l1_thresh,
                       num_steps=num_steps,
                       return_hidden_states_first=return_hidden_states_first,
                       name=name,
                       callbacks=callbacks)
        object.__setattr__(self.state, 'cnt', 0)
        object.__setattr__(self.state, 'accumulated_rel_l1_distance', 0)
        object.__setattr__(self.state, 'rescale_func', TorchPoly1D(self.state.cache_context.get_coef(self.state.name)))

    def get_start_idx(self):
        return 0

    def are_two_tensor_similar(self, t1, t2, threshold):
        if self.state.cnt == 0 or self.state.cnt == self.state.num_steps-1:
            self.state.accumulated_rel_l1_distance = 0
            self.state.use_cache = False
        else:
            diff = self.l1_distance_two_tensor(t1, t2)
            self.state.accumulated_rel_l1_distance += self.state.rescale_func(diff)
            if self.state.accumulated_rel_l1_distance < threshold:
                self.state.use_cache = True
            else:
                self.state.use_cache = False
                self.state.accumulated_rel_l1_distance = 0
        self.state.cnt += 1
        if self.state.cnt == self.state.num_steps:
            self.state.cnt = 0
        return self.state.use_cache


    def get_modulated_inputs(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        inp = hidden_states.clone()
        temb_ = kwargs.get("temb", None)
        if temb_ is not None:
            temb_ = temb_.clone()
        else:
            raise ValueError("'temb' not found in kwargs")
        modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.state.transformer_blocks[0].norm1(inp, emb=temb_)
        previous_modulated_input = self.state.cache_context.get_buffer("modulated_inputs")
        self.state.cache_context.set_buffer("modulated_inputs", modulated_inp)
        return modulated_inp, previous_modulated_input, hidden_states, encoder_hidden_states
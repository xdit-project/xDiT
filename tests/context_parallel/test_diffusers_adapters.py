import functools
import logging
import os
import time
from typing import Tuple, Optional

import pytest
import torch
from diffusers import DiffusionPipeline
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from xfuser.config.config import (
    EngineConfig,
    FastAttnConfig,
    ParallelConfig,
    TensorParallelConfig,
    PipeFusionParallelConfig,
    SequenceParallelConfig,
    DataParallelConfig,
    ModelConfig,
    InputConfig,
    RuntimeConfig,
)
from xfuser.logger import init_logger
from xfuser.core.distributed import init_distributed_environment
from xfuser.core.distributed import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_pipeline_parallel_world_size,
    get_runtime_state,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
    get_world_group,
    initialize_runtime_state,
    is_dp_last_group,
)
from xfuser.model_executor.layers.attention_processor import xFuserFluxAttnProcessor2_0

os.environ["HF_HUB_CACHE"] = "/mnt/co-research/shared-models/hub"

logger = init_logger(__name__)

def parallelize_transformer(pipe: DiffusionPipeline):
    transformer = pipe.transformer
    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        *args,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        **kwargs,
    ):
        assert hidden_states.shape[0] % get_classifier_free_guidance_world_size() == 0, \
            f"Cannot split dim 0 of hidden_states ({hidden_states.shape[0]}) into {get_classifier_free_guidance_world_size()} parts."
        if encoder_hidden_states.shape[-2] % get_sequence_parallel_world_size() != 0:
            get_runtime_state().split_text_embed_in_sp = False
        else:
            get_runtime_state().split_text_embed_in_sp = True
        
        if isinstance(timestep, torch.Tensor) and timestep.ndim != 0 and timestep.shape[0] == hidden_states.shape[0]:
            timestep = torch.chunk(timestep, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
        hidden_states = torch.chunk(hidden_states, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
        hidden_states = torch.chunk(hidden_states, get_sequence_parallel_world_size(),dim=-2)[get_sequence_parallel_rank()]
        encoder_hidden_states = torch.chunk(encoder_hidden_states, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
        if get_runtime_state().split_text_embed_in_sp:
            encoder_hidden_states = torch.chunk(encoder_hidden_states, get_sequence_parallel_world_size(),dim=-2)[get_sequence_parallel_rank()]
        img_ids = torch.chunk(img_ids, get_sequence_parallel_world_size(),dim=-2)[get_sequence_parallel_rank()]
        if get_runtime_state().split_text_embed_in_sp:
            txt_ids = torch.chunk(txt_ids, get_sequence_parallel_world_size(),dim=-2)[get_sequence_parallel_rank()]
        
        for block in transformer.transformer_blocks + transformer.single_transformer_blocks:
            block.attn.processor = xFuserFluxAttnProcessor2_0()
        
        output = original_forward(
            hidden_states,
            encoder_hidden_states,
            *args,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            **kwargs,
        )

        return_dict = not isinstance(output, tuple)
        sample = output[0]
        sample = get_sp_group().all_gather(sample, dim=-2)
        sample = get_cfg_group().all_gather(sample, dim=0)
        if return_dict:
            return output.__class__(sample, *output[1:])
        return (sample, *output[1:])

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward

@pytest.mark.timeout(900)
class DiffusionPipelineTest(DTensorTestBase):
    @property
    def world_size(self):
        device_count = torch.cuda.device_count()
        if device_count <= 4:
            return device_count
        elif device_count < 6:
            return 4
        elif device_count < 8:
            return 6
        else:
            return 8

    def new_pipe(self, dtype, device, rank):
        raise NotImplementedError

    def call_pipe(self, pipe, *args, **kwargs):
        raise NotImplementedError
    
    def create_config(self, compile, dtype) -> Tuple[EngineConfig, InputConfig]:
        raise NotImplementedError

    @property
    def enable_vae_parallel(self):
        return False

    def _test_benchmark_pipe(self, dtype, device, parallelize, compile):
        torch.manual_seed(0)
    
        init_distributed_environment()
        engine_config, input_config = self.create_config(compile, dtype)

        pipe = self.new_pipe(dtype, device)

        parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{self.rank}")
        print(f"Parameter memory: {parameter_peak_memory / 1e9:.2f} GB")

        initialize_runtime_state(pipe, engine_config)
        get_runtime_state().set_input_parameters(
            height=input_config.height,
            width=input_config.width,
            batch_size=1,
            num_inference_steps=input_config.num_inference_steps,
            max_condition_sequence_length=512,
            split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
        )

        if parallelize:
            parallelize_transformer(pipe)

        if compile:
            if parallelize:
                torch._inductor.config.reorder_for_compute_comm_overlap = True
            # If cudagraphs is enabled and parallelize is True, the test will hang indefinitely
            # after the last iteration.
            pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

        for _ in range(3):
            begin = time.time()
            self.call_pipe(pipe)
            end = time.time()
            print(f"Time taken: {end - begin:.3f} seconds")


class FluxPipelineTest(DiffusionPipelineTest):
    def new_pipe(self, dtype, device):
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=dtype,
        ).to(f"{device}:{self.rank}")
        return pipe

    def call_pipe(self, pipe, *args, **kwargs):
        return pipe(
            "A cat holding a sign that says hello world",
            num_inference_steps=28,
            output_type="pil" if self.rank == 0 else "pt",
        )
    
    def create_config(self, compile, dtype) -> Tuple[EngineConfig, InputConfig]:
        model_config = ModelConfig(
            model="black-forest-labs/FLUX.1-dev",
        )

        runtime_config = RuntimeConfig(
            warmup_steps=3,
            # use_cuda_graph=False,
            use_parallel_vae=False,
            use_torch_compile=compile,
            use_onediff=not compile,
            # use_profiler=False,
            use_fp8_t5_encoder=False,
        )

        parallel_config = ParallelConfig(
            dp_config=DataParallelConfig(dit_parallel_size=self.world_size),
            sp_config=SequenceParallelConfig(
                ulysses_degree=self.world_size,
                ring_degree=1,
                dit_parallel_size=self.world_size,
            ),
            tp_config=TensorParallelConfig(dit_parallel_size=self.world_size),
            pp_config=PipeFusionParallelConfig(dit_parallel_size=self.world_size),
            world_size=self.world_size,
            dit_parallel_size=self.world_size,
            vae_parallel_size=0,
        )

        fast_attn_config = FastAttnConfig()

        engine_config = EngineConfig(
            model_config=model_config,
            runtime_config=runtime_config,
            parallel_config=parallel_config,
            fast_attn_config=fast_attn_config,
        )

        input_config = InputConfig(
            height=1024,
            width=1024,
            batch_size=1,
            prompt="A dark tree.",
            num_inference_steps=28,
            max_sequence_length=512,
            output_type="pil",
        )
        engine_config.runtime_config.dtype = dtype

        return engine_config, input_config


    @property
    def enable_vae_parallel(self):
        return True

    @pytest.mark.skipif("not torch.cuda.is_available()")
    @with_comms
    @parametrize("dtype", [torch.bfloat16])
    @parametrize("device", ["cuda"])
    @parametrize(
        "parallelize,compile",
        [
            [False, False],
            [False, True],
            [True, False],
            [True, True],
        ],
    )
    def test_benchmark_pipe(self, dtype, device, parallelize, compile):
        super()._test_benchmark_pipe(dtype, device, parallelize, compile)

instantiate_parametrized_tests(DiffusionPipelineTest)
instantiate_parametrized_tests(FluxPipelineTest)

if __name__ == "__main__":
    run_tests()
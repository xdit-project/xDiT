# Parallelize new models with CFG parallelism provided by xDiT

This tutorial focuses on utilizing CFG parallelism in the context of the CogVideoX text-to-video model. It provides step-by-step instructions on how to apply CFG parallelism to a new DiT model.

The diffusion process involves receiving Gaussian noise as input, iteratively predicting and denoising using the *CogVideoX Transformer*, and generating the output video. This process, typically executed on a single GPU within `diffusers`, is outlined in the following figure.

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/developer/single-gpu-cfg.png" 
    alt="single-gpu.png">
</div>

The Transformer's input comprises timesteps, a text sequence, and an image sequence. CogVideoX employs classifier-free guidance (CFG) to enhance video quality. During each iteration, the model not only feeds the timesteps, text sequence, and image sequence into the transformer but also generates an empty text sequence. This, along with the original timesteps and image sequence, is forwarded to the transformer, enabling the model to combine the two outputs for noise prediction at the iteration's end. Consequently, when a single prompt is passed to the model, the timesteps, text sequence, and image sequence each have a batch size of 2.

CFG parallelism, depicted in the following figure, leverages 2 GPUs to process the two batches. At the beginning of each iteration, CFG parallelism splits the input tensor by the batch dimension, distributes each part to a GPU. At the end of the iteration, the two GPUs communicate through the `all_gather` primitive.


<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/developer/multiple-gpus-cfg.png" 
    alt="multiple-gpus.png">
</div>

Note that, for DiT models with no CFG functionality, such as Flux and HunyuanVideo, CFG parallelism cannot be applied.

To accelerate CogVideoX inference using CFG parallelism, two modifications to the original diffusion process are required. Firstly, the xDiT environment should be initialized at the beginning of the program. This requires several function such as `init_distributed_environment`, `initialize_model_parallel`, and `get_world_group` provided by xDiT. Secondly, in `diffusers`, the CogVideoX model is encapsulated within the `CogVideoXTransformer3DModel` class located at [diffusers/models/transformers/cogvideox_transformer_3d.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/cogvideox_transformer_3d.py), and it is reqired to split and merge seqeunces before and after the `forward` function of `CogVideoXTransformer3DModel`.

## 1. Initialization

Begin by setting up the distributed environment with the following code snippet:

```python
from xfuser.core.distributed import init_distributed_environment
dist.init_process_group("nccl")
init_distributed_environment(
    rank=dist.get_rank(), 
    world_size=dist.get_world_size()
)
# Ensure world size is 2 for CFG parallelism
```

Specify the level of CFG parallelism:

```python
from xfuser.core.distributed import initialize_model_parallel
initialize_model_parallel(
    classifier_free_guidance_degree=2,
)
```

Ensure the model checkpoint is loaded on all GPUs by copying the pipe from the CPU to each GPU:

```python
from xfuser.core.distributed import get_world_group
local_rank = get_world_group().local_rank
device = torch.device(f"cuda:{local_rank}")
pipe.to(device)
```


## 2. Splitting and Merging Sequences

The `forward` function of `CogVideoXTransformer3DModel` orchestrates the inference process for a single step iteration, outlined below:

```python
class CogVideoXTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    )
```

To parallelize the inference process, we utilize `parallelize_transformer` on `pipe`. Within this function, a `new_forward` function is introduced with identical input and output parameters as the original function. The `new_forward` function performs the following steps:

- Splits the timesteps, text sequence, and image sequence based on the batch size dimension, allocating each batch to a GPU.
- Executes the original forward process on the two GPUs.
- Merges the predicted noise using all_gather.

The code snippet below demonstrates the utilization of `@functools.wraps` to decorate the new_forward function, ensuring that essential details such as the function name, docstring, and argument list are inherited from original_forward. As forward is a method of a class object, the `__get__` function is employed to set transformer as the initial argument for new_forward, subsequently assigning new_forward to transformer.forward.

```python
def parallelize_transformer(pipe: DiffusionPipeline):
    transformer = pipe.transformer
    original_forward = transformer.forward

    
    # definition of the new forward
    @functools.wraps(transformer.__class__.forward)
    def new_forward(...)
    
    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward

parallelize_transformer(pipe)
```
The input parameters, timestep, hidden_state, and encoder_hidden_states, represent the timesteps, the input video sequence, and the input text sequence, respectively. These tensors require division. Their shapes are outlined below:

- timesteps (batch_size)
- hidden_state (batch_size, temporal_length, channels, height, width)
- encoder_hidden_states (batch_size, text_length, hidden_state)

where the batch size is 2. xDiT provides helper functions for CFG parallelism, offering functionalities such as `get_classifier_free_guidance_rank()` and `get_classifier_free_guidance_world_size()` to access the number of GPUs and their respective ranks. The `get_cfg_group()` function facilitates CFG parallelism, incorporating an `all_gather()` operation to merge sequences after `forward`. The new forward function is outlined as follows:

```python
from xfuser.core.distributed import (
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_cfg_group,
)

@functools.wraps(transformer.__class__.forward)
def new_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    timestep: torch.LongTensor = None,
    timestep_cond: Optional[torch.Tensor] = None,
    ofs: Optional[Union[int, float, torch.LongTensor]] = None,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    # Step 1: split tensors
    timestep = torch.chunk(timestep, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
    hidden_states = torch.chunk(hidden_states, get_classifier_free_guidance_world_size(), dim=0)[get_classifier_free_guidance_rank()]
    encoder_hidden_states = torch.chunk(encoder_hidden_states, get_classifier_free_guidance_world_size(), dim=0)[get_classifier_free_guidance_rank()]
    
    # Step 2: perform the original forward
    output = original_forward(
        hidden_states,
        encoder_hidden_states,
        timestep=timestep,
        timestep_cond=timestep_cond,
        ofs=ofs,
        image_rotary_emb=image_rotary_emb,
        **kwargs,
    )

    return_dict = not isinstance(output, tuple)
    sample = output[0]
    # Step 3: merge the output from two GPUs
    sample = get_cfg_group().all_gather(sample, dim=0)
    
    if return_dict:
        return output.__class__(sample, *output[1:])
    
    return (sample, *output[1:])
```

A complete example script can be found in [adding_model_cfg.py](adding_model_cfg.py).
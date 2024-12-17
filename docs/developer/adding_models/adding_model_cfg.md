# Parallelize new models with CFG parallelism provided by xDiT

This tutorial focuses on using the CogVideoX text-to-video model as an example and provides instructions on applying CFG parallelism supported by xDiT to a new DiT model.

The diffusion process involves iterating through the input video as Gaussian noise and generating an output video. In each iteration, the DiT model predicts the noise in the video and performs denoising. The original diffusion process of CogVideoX is implemented in `diffusers`, which use a single GPU for video generation. The following figure provides an overview of the video generation process on a single GPU.

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/developer/diffusion-overview-single-gpu.png" 
    alt="diffusion-overview-single-gpu.png">
</div>

In contrast, CFG parallelism leverages 2 GPUs to expedite the generation process. As depicted in the figure below, at the start of each iteration, CFG parallelism divides the input tensor by the first dimension (the `batch_size` dimension), assigning each part to a GPU. At the iteration's end, the two GPUs communicate to consolidate results. To accelerate CogVideoX inference using CFG parallelism, the following modifications are required to the original process.

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/developer/diffusion-overview-multiple-gpus.png" 
    alt="diffusion-overview-multiple-gpus.png">
</div>

## 1. Initialization

To start, we need to set up the distributed environment with the following code snippet:

```python
from xfuser.core.distributed import init_distributed_environment
dist.init_process_group("nccl")
init_distributed_environment(
    rank=dist.get_rank(), 
    world_size=dist.get_world_size()
)
```

Next, we specify the level of CFG parallelism, which is 2.

```python
from xfuser.core.distributed import initialize_model_parallel
initialize_model_parallel(
    classifier_free_guidance_degree=2,
)
```

Ensure that the model checkpoint is loaded on all GPUs. `diffusers` place the model checkpoints into a `pipe`, so we copy the pipe from the CPU to each GPU:

```python
device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
pipe.to(device)
```


## 2. Splitting and Merging Sequences

The DiT model of CogVideoX is encapsulated in the CogVideoX Transformer class, where the `forward` function defines the inference process for a single step iteration, as shown below:

```python
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

In order to parallelize the inference process, we employ `parallelize_transformer` to `pipe`. Within the function, we introduce a `new_forward` function with the same input and output as the original function. `new_forward` integrates split and merge logic before and after the original forward function, respectively. As illustrated below, we utilize `@functools.wraps` to decorate the new_forward function, ensuring that the function name, docstring, argument list, etc., are inherited from `original_forward`. Additionally, we employ the `__get__` function to designate transformer as the initial argument for new_forward and then assign `new_forward` to `transformer.forward`.

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


Within the input parameters, `timestep`, `hidden_state` and `encoder_hidden_states` represent the timesteps, the input video sequence and the input text sequence. These three tensors need to be split. The shapes of these tensors are detailed in the table below:

- `timesteps` (batch_size)
- `hidden_state` (batch_size, temporal_length, channels, height, width)
- `encoder_hidden_states` (batch_size, text_length, hidden_state)

xDiT provides runtime states for sequence parallelism. For instance, `get_classifier_free_guidance_rank()` and `get_classifier_free_guidance_world_size()` can retrieve the number of GPUs for sequential parallelism and the rank of each GPU. `get_cfg_group()`can obtain the group for CFG parallelism, which includes an `all_gather()` function to merge sequences after the forward pass. The new forward function can be defined as follows:

```python
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
    timestep = torch.chunk(timestep, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
    hidden_states = torch.chunk(hidden_states, get_classifier_free_guidance_world_size(), dim=0)[get_classifier_free_guidance_rank()]
    encoder_hidden_states = torch.chunk(encoder_hidden_states, get_classifier_free_guidance_world_size(), dim=0)[get_classifier_free_guidance_rank()]
    
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
    sample = get_cfg_group().all_gather(sample, dim=0)
    
    if return_dict:
        return output.__class__(sample, *output[1:])
    
    return (sample, *output[1:])
```

A complete example script can be found in [adding_model_cfg.py](adding_model_cfg.py).
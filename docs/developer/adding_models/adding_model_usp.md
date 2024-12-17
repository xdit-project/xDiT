# Parallelize new models with USP provided by xDiT

This tutorial focuses on using the CogVideoX text-to-video model as an example and provides instructions on applying USP (Unified Sequence Parallelism) supported by xDiT to a new DiT model.

The diffusion process involves iterating through the input video as Gaussian noise and generating an output video. In each iteration, the DiT model predicts the noise in the video and performs denoising. The original diffusion process of CogVideoX is implemented in `diffusers`, which use a single GPU for video generation. The following figure provides an overview of the video generation process on a single GPU.

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/developer/diffusion-overview-single-gpu.png" 
    alt="diffusion-overview-single-gpu.png">
</div>

In contrast, USP leverages multiple GPUs to expedite the generation process. As depicted in the figure below, at the start of each iteration, USP divides the input sequence, assigning each GPU to process a subsequence. At the iteration's end, all GPUs communicate to consolidate results. To accelerate CogVideoX inference using USP, the following modifications are required to the original process.

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

Next, we specify the level of sequential parallelization. The number of GPUs should be equal to the degrees of ulysses attention and the degrees of ring attention:

```python
from xfuser.core.distributed import initialize_model_parallel
initialize_model_parallel(
    sequence_parallel_degree=dist.get_world_size(),
    ring_degree=<ring_degree>,
    ulysses_degree=<ulysses_degree>,
)
# restriction: dist.get_world_size() == <ring_degree> x <ulysses_degree>
```

Ensure that the model checkpoint is loaded on all GPUs. `diffusers` place the model checkpoints into a `pipe`, so we copy the pipe from the CPU to each GPU:

```python
device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
pipe.to(device)
```


## 2. Splitting and Merging Sequences

The DiT model of CogVideoX is encapsulated in the CogVideoXTransformer class, where the `forward` function defines the inference process for a single step iteration, as shown below:

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


Within the input parameters, `hidden_state` and `encoder_hidden_states` represent the input sequence corresponding to the video and text segments, while `image_rotary_emb` denotes the Relative positional encoding (RoPE) used in attention computation, which can be added to hidden_state. These three tensors need to be split. The shapes of these tensors are detailed in the table below:

- `hidden_state` (batch_size, temporal_length, channels, height, width)
- `encoder_hidden_states` (batch_size, text_length, hidden_state)
- `image_rotary_emb` (batch_size, rope_temporal_length, rope_height, rope_width, hidden_state)

In CogVideoX, `rope_height` and `rope_width` are half of `height` and `width` respectively. xDiT provides runtime states for sequence parallelism. For instance, `get_sequence_parallel_rank()` and `get_sequence_parallel_world_size()` can retrieve the number of GPUs for sequential parallelism and the rank of each GPU. `get_sp_group()`can obtain the group for sequence parallelism, which includes an `all_gather()` function to merge sequences after the forward pass. Suppose we partition the video sequence and `image_rotary_emb` by the height dimension, then the new forward function can be defined as follows:

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
    rope_h, rope_w = hidden_states.shape[-2] // 2, hidden_states.shape[-1] // 2
    hidden_states = torch.chunk(hidden_states, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]
    encoder_hidden_states = torch.chunk(encoder_hidden_states, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]
    
    if image_rotary_emb is not None:
        freqs_cos, freqs_sin = image_rotary_emb
        dim_thw = freqs_cos.shape[-1]
        freqs_cos = freqs_cos.reshape(-1, rope_h, rope_w, dim_thw)
        freqs_cos = torch.chunk(freqs_cos, get_sequence_parallel_world_size(), dim=-3)[get_sequence_parallel_rank()]
        freqs_cos = freqs_cos.reshape(-1, dim_thw)
        
        freqs_sin = freqs_sin.reshape(-1, rope_h, rope_w, dim_thw)
        freqs_sin = torch.chunk(freqs_sin, get_sequence_parallel_world_size(), dim=-3)[get_sequence_parallel_rank()]
        freqs_sin = freqs_sin.reshape(-1, dim_thw)
        
        image_rotary_emb = (freqs_cos, freqs_sin)
    
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
    sample = get_sp_group().all_gather(sample, dim=-2)
    
    if return_dict:
        return output.__class__(sample, *output[1:])
    
    return (sample, *output[1:])
```


## 3. Attention with Sequential Parallelism

Following the previous modifications, each GPU now receives a subsequence and processes self-attention within its designated subsequence. To ensure correct computation, we need to adapt the attention mechanism. The original attention calculation is handled within the `CogVideoXAttnProcessor2_0` class. Its forward function derives `Q`, `K`, and `V` from `hidden_state` and carries out attention computation. To address the need for a modified approach, we introduce a new class, `xDiTCogVideoAttentionProcessor`. The key variance in its forward function compared to the original lies in the attention computation step, where we substitute the original `F.scaled_dot_product_attention` with the `USP` method provided by xDiT. This method accurately computes attention based on the Ulysses degree and ring degree.

```diff
- hidden_states = F.scaled_dot_product_attention(
-     query, key, value, dropout_p=0.0, is_causal=False
- )
+ hidden_states = USP(
+     query, key, value, dropout_p=0.0, is_causal=False
+ )
```


Lastly, to incorporate our `xDiTCogVideoAttentionProcessor` into the forward operation instead of the original `CogVideoXAttnProcessor2_0`, we iterate through each transformer block and update its attention processor. Prior to `output = original_forward...`, we include the following line to ensure that each attention block utilizes our processor:

```
for block in transformer.transformer_blocks:
    block.attn1.processor = xDiTCogVideoAttentionProcessor()
```

## 4. Adapting Positional Embeddings for Parallelism in CogVideoX

In most DiT models, the previously discussed modifications suffice. However, CogVideoX introduces an extra layer of complexity by incorporating positional embeddings into the video sequence before it enters the initial transformer block. This functionality is implemented within the transformer.patch_embed object. Each GPU's subsequence requires a distinct positional embedding. To streamline this process, within the new forward function of transformer.patch_embed, we first execute an `all_gather` operation to ensure every GPU obtains the complete text and image sequences. Subsequently, each GPU applies the original `transformer.patch_embed` operation to the entire sequence. Finally, we segment the text and image sequences identically to before. 

In CogVideoX, the height and width of the embedded video sequence are halved compared to the original dimensions. Consequently, we can define the new patch embed function as follows.

```python
original_patch_embed_forward = transformer.patch_embed.forward

@functools.wraps(transformer.patch_embed.__class__.forward)
def new_patch_embed(
    self, text_embeds: torch.Tensor, image_embeds: torch.Tensor
):
    text_embeds = get_sp_group().all_gather(text_embeds.contiguous(), dim=-2)
    image_embeds = get_sp_group().all_gather(image_embeds.contiguous(), dim=-2)
    batch, embed_height, embed_width = image_embeds.shape[0], image_embeds.shape[-2] // 2, image_embeds.shape[-1] // 2
    text_len = text_embeds.shape[-2]
    
    output = original_patch_embed_forward(text_embeds, image_embeds)

    text_embeds = output[:,:text_len,:]
    image_embeds = output[:,text_len:,:].reshape(batch, -1, embed_height, embed_width, output.shape[-1])

    text_embeds = torch.chunk(text_embeds, get_sequence_parallel_world_size(),dim=-2)[get_sequence_parallel_rank()]
    image_embeds = torch.chunk(image_embeds, get_sequence_parallel_world_size(),dim=-3)[get_sequence_parallel_rank()]
    image_embeds = image_embeds.reshape(batch, -1, image_embeds.shape[-1])
    return torch.cat([text_embeds, image_embeds], dim=1)

new_patch_embed = new_patch_embed.__get__(transformer.patch_embed)
transformer.patch_embed.forward = new_patch_embed
```

A complete example script can be found in [adding_model_usp.py](adding_model_usp.py).
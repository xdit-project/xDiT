# Parallelize new models with USP provided by xDiT

This tutorial focuses on utilizing USP (Unified Sequence Parallelism) in the context of the CogVideoX text-to-video model. USP combines two sequence parallelism methods such as ulysses attention and ring attention. This tutorial provides step-by-step instructions on how to apply USP to a new DiT model.

The diffusion process involves receiving Gaussian noise as input, iteratively predicting and denoising using the *CogVideoX Transformer*, and generating the output video. This process, typically executed on a single GPU within `diffusers`, is outlined in the following figure.

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/developer/single-gpu-usp.png" 
    alt="single-gpu.png">
</div>

The Transformer's input comprises a text sequence, an image sequence, and an image rotary position embedding (RoPE). USP segments the input tensor along the sequence length dimension and assigns each segment to a GPU. Notably, the image sequence and the RoPE contains much more elements than the text sequence. Thus, two design choices emerge: 

- Split the text sequence, image sequence, and RoPE; or
- Split solely the image sequence and RoPE while each GPU retains a complete text sequence. 

The former choice is easy to implement and incur no unnecessary computation, but requires both text and image sequence length being divisible by the number of GPUs. Conversely, the latter choice, although more complex to implement, and involving redundant computation, is useful when the text sequence length is not divisible by the number of GPUs.

This tutorial focus on the first situaion. The second one is described in [Parallelize new models with USP provided by xDiT (text replica)](adding_model_usp_text_replica.md)

As depicted in the subsequent diagram, USP harnesses multiple GPUs (2 in this instance) to execute diffusion. At the beginning of each iteration, USP segments the input tensor along the sequence length dimension, assigning each segment to a GPU. As the input sequence gets partitioned, USP necessitates communication to facilitate attention computation within the transformer. Concluding the iteration, the two GPUs communicate via the `all_gather` primitive.

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/developer/multiple-gpus-usp.png" 
    alt="multiple-gpus.png">
</div>

To accelerate CogVideoX inference using USP, four modifications to the original diffusion process are required. 

Firstly, the xDiT environment should be initialized at the beginning of the program. This requires several function such as `init_distributed_environment`, `initialize_model_parallel`, and `get_world_group` provided by xDiT. 

Secondly, in `diffusers`, the CogVideoX model is encapsulated within the `CogVideoXTransformer3DModel` class located at [diffusers/models/transformers/cogvideox_transformer_3d.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/cogvideox_transformer_3d.py), and it is reqired to split and merge seqeunces before and after the `forward` function.

Thirdly, in `diffusers`, the attention computation is managed by the `CogVideoXAttnProcessor2_0` class in [`diffusers/models/attention_processor.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py). As shown at the top of the following figure, we need to replace the attention computation into the USP version provided by xDiT to achieve parallel computation. The bottom of the figure indicates that, the results obtained from both versions of attention computation are consistent.

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/developer/usp_1.png" 
    alt="usp_1.png">
</div>

Finally, as each GPU own a distict sequence segment, the patch embedding layer in [diffusers/models/embeddings.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py) need to be adapted.

## 1. Initialization

Begin by setting up the distributed environment with the following code snippet:

```python
from xfuser.core.distributed import init_distributed_environment
dist.init_process_group("nccl")
init_distributed_environment(
    rank=dist.get_rank(), 
    world_size=dist.get_world_size()
)
```

Subsequently, define the level of sequential parallelization. The count of GPUs should align with the product of the degrees of ulysses attention and ring attention:

```python
from xfuser.core.distributed import initialize_model_parallel
initialize_model_parallel(
    ring_degree=<ring_degree>,
    ulysses_degree=<ulysses_degree>,
)
# restriction: dist.get_world_size() == <ring_degree> x <ulysses_degree>
```

Ensure that the model checkpoint is loaded on all GPUs. `diffusers` place the model checkpoints into a `pipe`, so we copy the pipe from the CPU to each GPU:

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

- Splits the text sequence, the image sequence, and the RoPE based on the sequence length dimension, allocating each batch to a GPU.
- Executes the original forward process on all GPUs.
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

In the input parameters, hidden_state, encoder_hidden_states, and image_rotary_emb represent the image sequence, text sequence, and the RoPE, which are added to hidden_state respectively. These three tensors require splitting. The shapes of these tensors are outlined in the table below:

- `hidden_state` (batch_size, temporal_length, channels, height, width)
- `encoder_hidden_states` (batch_size, text_length, hidden_dim)
- `image_rotary_emb` (batch_size, rope_temporal_length, rope_height, rope_width, hidden_dim)

`temporal_length`, `height`, and `width` denote the size on the temporal, height, and width dimensions of `hidden_state`. `rope_temporal_length`, `rope_height`, and `rope_width` denote the size on the temporal, height, and width dimensions of `image_rotary_emb`. In CogVideoX, `rope_height` and `rope_width` are half of `height` and `width` respectively. xDiT provides helper functions for USP, offering functionalities such as `get_sequence_parallel_rank()` and `get_sequence_parallel_world_size()` to access the number of GPUs and their respective ranks. The `get_sp_group()` function facilitates USP, incorporating an `all_gather()` operation to merge sequences after `forward`. When partitioning the text sequence by the text_length dimension and the image sequence and RoPE by the height dimension, the new forward function can be defined as follows:

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

    # Step 1: split tensors
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
    sample = get_sp_group().all_gather(sample, dim=-2)
    
    if return_dict:
        return output.__class__(sample, *output[1:])
    
    return (sample, *output[1:])
```


## 3. Attention with USP

Within `CogVideoXAttnProcessor2_0`, the `__call__` function concatenates the text and image sequences into a unified sequence. Subsequently, the function extracts `Q`, `K`, and `V` from this sequence and conducts the attention computation.

As we distribute the text and image sequences to multiple GPUs, we need to adapt the attention mechanism to ensure correctness. Each GPU derives `Q`, `K`, and `V` from its text and image seqeunce segments, with the subsequent attention computation transitioning to the USP version provided by xDiT. We use a new class `xDiTCogVideoAttentionProcessor`. The significant modification in its `__call__` function, in contrast to the original approach, lies in the attention computation step, where the conventional `F.scaled_dot_product_attention` is replaced with `xFuserLongContextAttention()` from xDiT, which computes attention based on the Ulysses degree and ring degree.

```diff
def __init__(self):
    ...
+   self.hybrid_seq_parallel_attn = xFuserLongContextAttention()

def __call__(...):
    ...
-   hidden_states = F.scaled_dot_product_attention(
-       query, key, value, dropout_p=0.0, is_causal=False
-   )
-   hidden_states = hidden_states.transpose(1, 2).reshape(
-       batch_size, -1, attn.heads * head_dim
-   )
+   query = query.transpose(1, 2)
+   key = key.transpose(1, 2)
+   value = value.transpose(1, 2)
+   hidden_states = self.hybrid_seq_parallel_attn(
+       None, query, key, value, dropout_p=0.0, causal=False
+   )
+   hidden_states = hidden_states.reshape(
+       batch_size, -1, attn.heads * head_dim
+   )
```

Lastly, to incorporate our `xDiTCogVideoAttentionProcessor` into the forward operation instead of the original `CogVideoXAttnProcessor2_0`, we iterate through each transformer block and update its attention processor. Prior to `output = original_forward...`, we include the following line to ensure that each attention block utilizes our processor:

```python
for block in transformer.transformer_blocks:
    block.attn1.processor = xDiTCogVideoAttentionProcessor()
```

## 4. Adapting Positional Embeddings for Parallelism in CogVideoX

In most DiT models, the previously discussed modifications suffice. However, CogVideoX introduces an extra layer of complexity by incorporating positional embeddings into the video sequence before it enters the initial transformer block. Each GPU's subsequence requires a distinct positional embedding. To streamline this process, within the new forward function of transformer.patch_embed, performing the following steps:

- Merge the text and image sequence by the `all_gather` operation to ensure every GPU obtains the complete text and image sequences;
- Apply the original `transformer.patch_embed` operation to the entire sequence on each GPU; and
- Segment the text and image sequences identically to before. 

In CogVideoX, the height and width of the embedded video sequence are halved compared to the original dimensions. Consequently, we can define the new patch embed function as follows.

```python
original_patch_embed_forward = transformer.patch_embed.forward

@functools.wraps(transformer.patch_embed.__class__.forward)
def new_patch_embed(
    self, text_embeds: torch.Tensor, image_embeds: torch.Tensor
):
    # Step 1: merge the text and image sequence 
    text_embeds = get_sp_group().all_gather(text_embeds.contiguous(), dim=-2)
    image_embeds = get_sp_group().all_gather(image_embeds.contiguous(), dim=-2)
    batch, embed_height, embed_width = image_embeds.shape[0], image_embeds.shape[-2] // 2, image_embeds.shape[-1] // 2
    text_len = text_embeds.shape[-2]
    
    # Step 2: apply the original patch_embed
    output = original_patch_embed_forward(text_embeds, image_embeds)

    # Step 3: segment the text and image sequences
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
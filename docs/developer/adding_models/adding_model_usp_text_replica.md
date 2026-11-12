# Implement USP with replicated text

For layouts that cannot split text tokens, keep them on every rank.
Without PipeFusion, FLUX.1-dev splits text embeddings and `txt_ids` when the
sequence-parallel degree divides their length. Otherwise, it replicates them. Read [xFuserFlux1Transformer2DWrapper and xFuserFluxAttnProcessor](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/transformers/transformer_flux.py) alongside the [USP integration guide](adding_model_usp.md).

## Treat replicated tokens as joint attention

Each rank holds a different image-token shard and the same full text sequence. Ordinary distributed K/V would repeat the replicated text keys once per rank.
This would change the attention result.

The FLUX processor separates text Q/K/V from image Q/K/V and passes them to [USP](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/layers/usp.py) as `joint_query`, `joint_key`, and `joint_value`, with `joint_strategy="front"`. Main Q/K/V contain only the image shard. All tensors use `[batch, heads, sequence, head_dim]` layout. USP handles the replicated keys once and returns combined text/image attention.
The processor restores the block's expected layout and separates the streams.

Use `"front"` or `"rear"` to match the model's token ordering. Preserve text position IDs and rotary embeddings alongside the replicated text. Gather only the distributed output stream. Gathering replicated text as a sharded
stream would duplicate tokens.

## Current FLUX restrictions

- The non-PipeFusion wrapper refuses `--use_fp8_comms` when prompt tokens cannot be split evenly. Joint-token attention bypasses the FP8 all-to-all path.
- The same wrapper rejects AITER VSA with both padded image tokens and replicated text. Other supported layouts can use padding.
- The FLUX PipeFusion pipeline keeps text replicated across sequence ranks and carries its condition state between transformer stages. Its patch-aware attention and cache path differs from the non-PipeFusion transformer. See [PipeFusion integration](adding_model_pipefusion.md).

When adapting another model, test attention against a reference with both divisible and nondivisible text lengths, image padding, joint-token order, and supported backends. Other models can require different fallback behavior and restrictions.

<a id="1-initialization"></a>
<a id="2-splitting-and-merging-sequences"></a>
<a id="3-attention-with-usp"></a>
<a id="4-adapting-positional-embeddings-for-parallelism-in-cogvideox"></a>

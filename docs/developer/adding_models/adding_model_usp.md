# Implement USP for a model

Use FLUX.1-dev as a reference for integration with the original Diffusers pipeline. Its [runner adapter](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/runner_models/flux.py) loads `xFuserFlux1Transformer2DWrapper` when PipeFusion is disabled. The [transformer and attention processor](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/transformers/transformer_flux.py) contain the complete implementation.

## Partition the transformer input

The wrapper receives image tokens, text embeddings, and their position IDs. Its `forward()`:

1. Pads image tokens and `img_ids` for the sequence-parallel degree. It rejects ranks that would contain only padding.
2. Splits image tokens and IDs along their sequence dimensions.
3. Splits text embeddings and `txt_ids` when the sequence-parallel degree divides the text length. Otherwise, it uses [replicated text](adding_model_usp_text_replica.md).
4. Calls the original transformer with matching local tokens and position IDs.
5. Gathers the predicted image sequence.
6. Removes padding and returns the original output structure.

Preserve token order and positional information together. FLUX partitions packed image tokens. Other models can require a different partition of image or video tensors.

## Replace attention

`xFuserFluxAttnProcessor` computes projections and rotary embeddings, then calls [USP](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/layers/usp.py). USP expects Q/K/V shaped `[batch, heads, local_sequence, head_dim]` and returns attention in that layout. The FLUX processor transposes projections before the call and restores the block's expected layout afterward.

USP combines Ulysses all-to-all with ring attention. For the sharded-text path, FLUX concatenates local text and image projections and passes `combine_qkv_a2a=True`, the attention module as `attn_layer`, and padding metadata. `attention_kwargs["valid_kv_len"]` identifies the global count of valid main K/V tokens, excluding joint tokens. Mask padding keys and keep padded rows from affecting later blocks. Removing padding only at the final output is insufficient.

`attn_layer` supplies registered PipeFusion KV-cache and FP8 communication state. For these features, preserve `attn_layer`. An isolated distributed attention call can bypass this state.

## Declare and validate support

The runner sets input metadata and validates attention head divisibility, backend restrictions, and model capabilities. Use the model's supported padding strategy. The FLUX adapter declares Ulysses and ring support and initializes `split_text_embed_in_sp` for the selected path.

Compare local attention and full model outputs with the unmodified reference, including uneven image-token counts, text replication, masks, position IDs, precision, and output ordering. Existing [FLUX sequence guards](https://github.com/xdit-project/xDiT/blob/main/tests/unit/model_executor/models/transformers/test_flux_sequence_parallel_guards.py) and [padding tests](https://github.com/xdit-project/xDiT/blob/main/tests/unit/model_executor/models/transformers/test_flux_sp_pad_rows.py) illustrate focused coverage. Model-output validation requires accelerators.

For user-facing degrees and backend constraints, see [USP](../../methods/usp.md). For stage/patch integration, see [PipeFusion](adding_model_pipefusion.md).

<a id="1-initialization"></a>
<a id="2-splitting-and-merging-sequences"></a>
<a id="3-attention-with-usp"></a>
<a id="4-adapting-positional-embeddings-for-parallelism-in-cogvideox"></a>

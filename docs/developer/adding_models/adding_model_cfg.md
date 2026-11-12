# Implement CFG parallelism for a model

CFG parallelism assigns ranks to the positive and negative prediction branches. Use [Z-Image](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/runner_models/z_image.py)
as a current example: its runner loads Diffusers `ZImagePipeline` with
`xFuserZImageTransformer2DWrapper`, and declares `use_cfg_parallel=True`.
FLUX.1-dev's guidance embedding does not create two prediction branches.

## Split branch inputs

Diffusers' [Z-Image pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/z_image/pipeline_z_image.py)
prepares positive embeddings **before** negative embeddings. It duplicates latents
and timesteps. The [transformer wrapper](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/transformers/transformer_z_image.py)
splits `x`, `cap_feats`, and `t` by CFG rank at the start of `forward()`:
rank 0 receives the positive branch, and rank 1 receives the negative branch.

Preserve the pipeline's branch order. Each tensor for a branch must select that
same branch. Do not split scalar or broadcast inputs as batch tensors.

## Gather before guidance

The wrapper reconstructs sequence-parallel outputs, unpatchifies them, and uses
`get_cfg_group().all_gather(..., dim=0)` to restore the original batch order.
The unchanged Diffusers pipeline then computes
`pos + current_guidance_scale * (pos - neg)`, applies any configured normalization,
and negates the prediction before the scheduler step. Keep this model-specific
arithmetic in the pipeline. The wrapper gathers predictions.

## Validate the contract

Declare CFG support only after both branch paths work. Compare with the original
pipeline using the same batch, seed, guidance, and precision. Check prompt batches,
branch order, output shapes, and matching collectives. With Z-Image's wrapper,
CFG must remain active throughout the call. Custom pipeline settings such as
`cfg_truncation` can disable a branch during generation. Supporting those settings
requires additional branch handling.
Z-Image-Turbo does not declare CFG parallel support.

See [CFG parallelism](../../methods/cfg_parallel.md) for user commands and
[CFG + USP](adding_model_cfg_usp.md) for simultaneous sequence partitioning.

<a id="1-initialization"></a>
<a id="2-splitting-and-merging-sequences"></a>

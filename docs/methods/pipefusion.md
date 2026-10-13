## PipeFusion: Displaced Patch Pipeline Parallelism for Diffusion Models
[Chinese Blog 1](https://zhuanlan.zhihu.com/p/699612077); [Chinese Blog 2](https://zhuanlan.zhihu.com/p/706475158)

PipeFusion is the innovative method first proposed by us. 
It is a sequence-level pipeline parallel method, similar to [TeraPipe](https://proceedings.mlr.press/v139/li21y.html), demonstrates significant advantages in weakly interconnected network hardware such as PCIe/Ethernet. 

PipeFusion innovatively harnesses input temporal redundancy—the similarity between inputs and activations across diffusion steps, a diffusion-specific characteristics also employed in DistriFusion. PipeFusion not only reduces communication volume but also streamlines pipeline parallelism with TeraPipe, avoiding the load balancing issues inherent in LLM models with Causal Attention.
It significantly surpasses other methods in communication efficiency, particularly in multi-node setups connected via Ethernet and multi-GPU configurations linked with PCIe.

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/overview.png" alt="PipeFusion Image">
</div>

The above picture compares DistriFusion and PipeFusion.
(a) DistriFusion replicates DiT parameters on two devices. 
It splits an image into 2 patches and employs asynchronous allgather for activations of every layer.
(b) PipeFusion shards DiT parameters on two devices.
It splits an image into 4 patches and employs asynchronous P2P for activations across two devices.

We briefly explain the workflow of PipeFusion. It partitions an input image into $M$ non-overlapping patches.
The DiT network is partitioned into $N$ stages ($N$ < $L$), which are sequentially assigned to $N$ computational devices. 
Note that $M$ and $N$ can be unequal, which is different from the image-splitting approaches used in sequence parallelism and DistriFusion.
Each device processes the computation task for one patch of its assigned stage in a pipelined manner. 

The PipeFusion pipeline workflow when $M$ = $N$ =4 is shown in the following picture.

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/workflow.png" alt="Pipeline Image">
</div>

### Composition and validation boundaries

- Parallel VAE encoding/decoding is scheduled independently of the DiT pipeline
  and can compose with PipeFusion when the model advertises the corresponding
  capability. Tile-parallel decoding remains subject to the model's normal tile
  geometry and memory validation.
- `--cache_method pipefusion` enables patch-aware step caching for models with
  a DBCache block layout. Every pipeline patch retains its own residual
  buffers. Patch 0 chooses the adaptive compute/reuse path for the stage and
  later patches follow that path without waiting for a timestep-wide
  collective. Different pipeline stages still decide independently because
  they own different blocks. With composed sequence parallelism, only replicas
  of the same stage synchronize through their sequence-parallel group.
- Other cache method names remain rejected with PipeFusion. The PipeFusion
  method reuses the model's DBCache block layout and tuning parameters, so
  model-specific `--cache_config` overrides such as the residual threshold
  continue to apply without exposing Cache-DiT's WORLD-group synchronization.
  Models whose cache layout spans multiple transformers are not yet supported.
- A Cache-DiT SCM mask can provide a stage-coordinated static policy by setting
  `scm_policy` to `slow`, `medium`, `fast`, `ultra`, or xDiT's `alternating`
  policy and `steps_computation_policy` to `static` in `--cache_config`.
  Because every stage knows those decisions in advance, PipeFusion caches the
  complete local stage residual and bypasses the configured Fn prefix on
  scheduled reuse steps. `alternating` computes the configured warmup prefix,
  reuses isolated alternating steps, and computes the final two correction
  steps. Dynamic policies continue to compute Fn and use residual similarity.
- `--memory_efficient_replicated_load` has a PipeFusion-specific meaning on
  runners that support stage-local meta construction (currently the FLUX
  family): each rank constructs and streams only its local transformer blocks.
  Replicated pipeline components remain eager. SD3.5's composition wrapper
  cannot construct a stage-local transformer on meta and explicitly rejects
  this option.
- A stage-local transformer cannot also be FSDP-wrapped because pipeline ranks
  own different blocks. Configuration therefore requires
  `--fully_shard_components` whenever FSDP and PipeFusion are combined, and the
  loader rejects stage-local names. Replicated components such as a text
  encoder may be selected for targeted FSDP.
- Quantization is applied while stage-local blocks stream from the checkpoint,
  before final device placement. Checkpoint names are preserved across stage
  slicing so each local block still maps to its original global layer.
- FLUX.2 reference-image conditioning uses a mixed generated/reference patch
  layout. Static reference tokens participate in transformer attention while
  scheduler feedback updates and returns only generated tokens.

### `torch.compile` warmup

Pipeline stages reach their first transformer call sequentially, so ordinary
lazy compilation also runs sequentially and can exceed the P2P watchdog on
large models. PipeFusion avoids that startup bottleneck by:

1. running a short eager pass that captures each rank's real full-sequence and
   patch-stage inputs;
2. compiling only the blocks retained by each local stage;
3. replaying the captured stage calls locally on every rank at the same time;
4. clearing temporary KV/runtime state before a short end-to-end validation.

The replay never performs pipeline communication. Models with additional
mutable PipeFusion state must expose `reset_pipefusion_state()` on their stage
wrapper so capture data cannot leak into inference. The generic capture,
snapshot, and replay implementation lives in
`xfuser/model_executor/pipefusion/compile.py`.


We have evaluated the accuracy of PipeFusion, DistriFusion and the baseline as shown below. To conduct the FID experiment, follow the detailed instructions provided in the [documentation](../../docs/fid/FID.md).

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/image_quality.png" alt="image_quality">
</div>


For more details, please refer to the following paper.

```
@inproceedings{
    fang2025pipefusion,
    title={PipeFusion: Patch-level Pipeline Parallelism for Diffusion Transformers Inference},
    author={Jiarui Fang and Jinzhe Pan and Aoyu Li and Xibo Sun and WANG Jiannan},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://openreview.net/forum?id=5xwyxupsLL}
}
```

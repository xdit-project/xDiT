# Historical parallel-method comparison

The table and illustration below were part of the earlier project overview.
They use simplified assumptions and are preserved as research context rather than
current runtime or peak-memory guarantees.

The remarkable performance of xDiT is attributed to two key facets.
Firstly, it leverages parallelization techniques, pioneering innovations such as USP, PipeFusion, and hybrid parallelism, to scale DiTs inference to unprecedented scales.

Secondly, we employ compilation technologies to enhance execution on GPUs, integrating established solutions like `torch.compile` and `onediff` to optimize xDiT's performance.

<h3 id="parallel">1. Parallel Methods</h3>

For supported DiT backbones, Data, USP, PipeFusion, and optional CFG parallelism can be combined subject to model and configuration constraints. Tensor parallelism and DistriFusion also have legacy/reference implementations with separate interfaces; their availability should not be inferred from the unified runner's option list.
For the VAE module, xDiT offers a parallel implementation, [DistVAE](https://github.com/xdit-project/DistVAE), designed to prevent out-of-memory (OOM) issues.
The (<span style="color: red;">xDiT</span>) highlights the methods first proposed by use.

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/methods/xdit_method.png" alt="xdit methods">
</div>

The communication and memory costs associated with the aforementioned intra-image parallelism, except for the CFG and DP (they are inter-image parallel), in DiTs are detailed in the table below. (* denotes that communication can be overlapped with computation.)

As we can see, PipeFusion and Sequence Parallel achieve the lowest communication cost on different scales and hardware configurations, making them suitable foundational components for a hybrid approach.

𝒑: Number of pixels;\
𝒉𝒔: Model hidden size;\
𝑳: Number of model layers;\
𝑷: Total model parameters;\
𝑵: Number of parallel devices;\
𝑴: Number of patch splits;\
𝑸𝑶: Query and Output parameter count;\
𝑲𝑽: KV Activation parameter count;\
𝑨 = 𝑸 = 𝑶 = 𝑲 = 𝑽: Equal parameters for Attention, Query, Output, Key, and Value;

| | attn-KV | communication cost | param memory | activations memory | extra buff memory |
| :-: | :-: | :-: | :-: | :-: | :-: |
| Tensor Parallel | fresh | $4O(p \times hs)L$ | $\frac{1}{N}P$ | $\frac{2}{N}A = \frac{1}{N}QO$ | $\frac{2}{N}A = \frac{1}{N}KV$ |
| DistriFusion* | stale | $2O(p \times hs)L$ | $P$ | $\frac{2}{N}A = \frac{1}{N}QO$ | $2AL = (KV)L$ |
| Ring Sequence Parallel* | fresh | $2O(p \times hs)L$ | $P$ | $\frac{2}{N}A = \frac{1}{N}QO$ | $\frac{2}{N}A = \frac{1}{N}KV$ |
| Ulysses Sequence Parallel | fresh | $\frac{4}{N}O(p \times hs)L$ | $P$ | $\frac{2}{N}A = \frac{1}{N}QO$ | $\frac{2}{N}A = \frac{1}{N}KV$ |
| PipeFusion* | stale- | $2O(p \times hs)$ | $\frac{1}{N}P$ | $\frac{2}{M}A = \frac{1}{M}QO$ | $\frac{2L}{N}A = \frac{1}{N}(KV)L$ |

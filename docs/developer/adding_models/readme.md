# Parallelize New Models with xDiT

xDiT was initially developed to accelerate the inference process of Diffusion Transformers (DiTs) within Huggingface `diffusers`. However, with the rapid emergence of various DiT models, you may find yourself needing to support new models that xDiT hasn't yet accommodated or models that are not officially supported by `diffusers` at all.

xDiT offers interfaces for multi-dimensional inference parallelization, including sequence parallelism, CFG parallelism, and PipeFusion. 

Among these parallelization methods, CFG parallelism stands out as the simplest to implement, requiring only additional split and merge operations over the sequence during each iteration. By leveraging CFG parallelism, a nearly 2x speedup can be achieved when conducting inference on two GPUs. Sequence parallelism, on the other hand, involves splitting the sequence during each iteration and necessitates additional communication to handle attention computation in a distributed environment. xDiT introduces USP (Unified Sequence Parallelism) combining two existing sequence parallelism method such as Ulysses Attention and Ring Attention. 

Moreover, CFG parallelism and USP can be performed samultaneously to achieve further speed enhancements.

PipeFusion is employed in situations where GPU memory constraints present challenges. This methodology, offered by xDiT, distributes the model parameters among multiple GPUs. While supporting models with PipeFusion is more complex compared to CFG parallelism and USP, it proves invaluable when dealing with GPUs of limited memory capacity.

For a detailed guide on leveraging CFG parallelism, USP, and PipeFusion using xDiT, refer to the following comprehensive tutorial.

[Parallelize new models with CFG parallelism provided by xDiT](adding_model_cfg.md)

[Parallelize new models with USP provided by xDiT](adding_model_usp.md)

[Parallelize new models with CFG parallelism and USP provided by xDiT](adding_model_cfg_usp.md)

[Parallelize new models with PipeFusion, USP, and CFG parallelism provided by xDiT](adding_model_pipefusion.md)
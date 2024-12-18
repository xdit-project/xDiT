# Parallelize New Models with xDiT

xDiT was initially developed to accelerate the inference process of Diffusion Transformers (DiTs) within Huggingface `diffusers`. However, with the rapid emergence of various DiT models, you may find yourself needing to support new models that xDiT hasn't yet accommodated or models that are not officially supported by `diffusers` at all.

xDiT offers interfaces for multiple parallelization methods, including CFG parallelism, sequence parallelism, and PipeFusion, shown as below. 

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/developer/api.jpg" 
    alt="api.jpg">
</div>

CFG parallelism is the simplest method to implement, requiring only additional split and merge operations over the batch_size dimension during each iteration. By leveraging CFG parallelism, a nearly 2x speedup can be achieved when conducting inference on two GPUs. Sequence parallelism, on the other hand, involves splitting the sequence during each iteration and necessitates additional communication to handle attention computation in a distributed environment. xDiT introduces USP (Unified Sequence Parallelism) combining two existing sequence parallelism method such as Ulysses Attention and Ring Attention. 

PipeFusion is employed in situations where GPU memory is insufficient or the communication bandwidth between GPUs is low. The method distributes the model parameters among multiple GPUs. Supporting models with PipeFusion is more complex compared to CFG parallelism and USP, but it is useful given machines of limited GPU memory capacity or limited bandwidth.

The parallelization methods mentioned above can be performed simultaneously to achieve further speed enhancements. For a detailed guide on leveraging CFG parallelism, USP, and PipeFusion using xDiT, refer to the following comprehensive tutorial.

[Parallelize new models with CFG parallelism provided by xDiT](adding_model_cfg.md)

[Parallelize new models with USP provided by xDiT](adding_model_usp.md)

[Parallelize new models with USP provided by xDiT (text replica)](adding_model_usp_text_replica.md)

[Parallelize new models with a hybrid of CFG parallelism and USP provided by xDiT](adding_model_cfg_usp.md)

[Parallelize new models with PipeFusion, USP, and CFG parallelism provided by xDiT](adding_model_pipefusion.md)
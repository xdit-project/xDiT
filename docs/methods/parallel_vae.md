## Patch Parallel VAE 

The [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) adopted by diffusers bring OOM to high-resolution images (8192px on A100). A critical issue is the CUDA memory spike, as documented in [diffusers/issues/5924](https://github.com/huggingface/diffusers/issues/5924).

To address this limitation, we developed [DistVAE](https://github.com/xdit-project/DistVAE), an solution that enables efficient processing of high-resolution images in parallel. Our approach incorporates two key strategies:

* Patch Parallel: We divide the feature maps in the latent space into multiple patches and perform sequence parallel VAE decoding across different devices. This technique reduces the peak memory required for intermediate activations to 1/$N$, where N is the number of devices utilized.
For the convolutional operator in VAE, we require the communication of the halo region data of the image as shown in the following figures.

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/methods/patchvaeconv.png" alt="hybrid process group config" width="60%">
</div>

* Chunked Input Processing: Similar to [MIT-patch-conv](https://hanlab.mit.edu/blog/patch-conv), we split the input feature map into chunks and feed them into convolution operator sequentially. This approach minimizes temporary memory consumption.

By synergizing these two methods, we have dramatically expanded the capabilities of VAE decoding. Our implementation successfully handles image resolutions up to 10240px - an impressive 11-fold increase compared to the default VAE implmentation.
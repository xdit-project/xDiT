import argparse

import torch
from diffusers import StableDiffusionXLPipeline
from torchprofile import profile_macs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, nargs="*", default=1024, help="Image size of generation")
    args = parser.parse_args()

    if isinstance(args.image_size, int):
        args.image_size = [args.image_size // 8, args.image_size // 8]
    elif len(args.image_size) == 1:
        args.image_size = [args.image_size[0] // 8, args.image_size[0] // 8]
    else:
        assert len(args.image_size) == 2
        args.image_size = [args.image_size[0] // 8, args.image_size[1] // 8]

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")

    unet = pipeline.unet

    latent_model_input = torch.randn(2, 4, *args.image_size, dtype=unet.dtype).to("cuda")
    t = torch.randn(1).to("cuda")
    prompt_embeds = torch.randn(2, 77, 2048, dtype=unet.dtype).to("cuda")
    add_text_embeds = torch.randn(2, 1280, dtype=unet.dtype).to("cuda")
    add_time_ids = torch.randint(0, 1024, (2, 6)).to("cuda")

    with torch.no_grad():
        macs = profile_macs(
            unet,
            args=(
                latent_model_input,
                t,
                prompt_embeds,
                None,
                None,
                None,
                None,
                {"text_embeds": add_text_embeds, "time_ids": add_time_ids},
            ),
        )
    print(f"MACs: {macs / 1e9:.3f}G")

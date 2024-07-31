import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("/data/models/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

image = pipe(
	"A cat holding a sign that says hello world",
	negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
).images[0]
image.save("a.png")
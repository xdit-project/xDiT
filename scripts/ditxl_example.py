import torch

from distrifuser.pipelines.dit import DistriDiTPipeline
from distrifuser.utils import DistriConfig

distri_config = DistriConfig(height=1024, width=1024, warmup_steps=4)
pipeline = DistriDiTPipeline.from_pretrained(
    distri_config=distri_config,
    pretrained_model_name_or_path="facebook/DiT-XL-2-256",
    # variant="fp16",
    # use_safetensors=True,
)

pipeline.set_progress_bar_config(disable=distri_config.rank != 0)
output = pipeline(
    # prompt="Emma Stone flying in the sky, cold color palette, muted colors, detailed, 8k",
    words=["panda"],
    generator=torch.Generator(device="cuda").manual_seed(42),
)
if distri_config.rank == 0:
    output.images[0].save("panda.png")

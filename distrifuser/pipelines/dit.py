import torch
from diffusers import DiTPipeline, Transformer2DModel

# from distrifuser.models.distri_sdxl_unet_pp import DistriSDXLUNetPP
# from distrifuser.models.distri_sdxl_unet_tp import DistriSDXLUNetTP
# from distrifuser.models.naive_patch_sdxl import NaivePatchSDXL
from distrifuser.utils import DistriConfig, PatchParallelismCommManager
from distrifuser.logger import init_logger

logger = init_logger(__name__)

class DistriDiTPipeline:
    def __init__(self, pipeline: DiTPipeline, module_config: DistriConfig):
        self.pipeline = pipeline
        self.distri_config = module_config

        self.static_inputs = None

        self.prepare()

    @staticmethod
    def from_pretrained(distri_config: DistriConfig, **kwargs):
        device = distri_config.device
        pretrained_model_name_or_path = kwargs.pop(
            "pretrained_model_name_or_path", "facebook/DiT-XL-2-256"
        )
        logger.info(f"Loading model from {pretrained_model_name_or_path}")
        torch_dtype = kwargs.pop("torch_dtype", torch.float16)
        transformer = Transformer2DModel.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, subfolder="transformer"
        ).to(device)

        # if distri_config.parallelism == "patch":
            # unet = DistriSDXLUNetPP(unet, distri_config)
        # elif distri_config.parallelism == "tensor":
            # unet = DistriSDXLUNetTP(unet, distri_config)
        # elif distri_config.parallelism == "naive_patch":
            # unet = NaivePatchSDXL(unet, distri_config)
        # else:
            # raise ValueError(f"Unknown parallelism: {distri_config.parallelism}")

        pipeline = DiTPipeline.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, transformer=transformer, **kwargs
        ).to(device)
        return DistriDiTPipeline(pipeline, distri_config)

    def set_progress_bar_config(self, **kwargs):
        pass

    @torch.no_grad()
    def __call__(self, words, *args, **kwargs):
        class_ids = self.pipeline.get_label_ids(words)
        return self.pipeline(class_labels=class_ids, *args, **kwargs)

    @torch.no_grad()
    def prepare(self, **kwargs):
        pass 

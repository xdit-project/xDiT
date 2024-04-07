import torch
from diffusers import PixArtAlphaPipeline
from diffusers.models.transformers.transformer_2d import Transformer2DModel

# from distrifuser.models.distri_sdxl_unet_tp import DistriSDXLUNetTP
from distrifuser.models import NaivePatchDiT, DistriDiTPP
from distrifuser.utils import DistriConfig, PatchParallelismCommManager
from distrifuser.logger import init_logger

logger = init_logger(__name__)

class DistriPixArtAlphaPipeline:
    def __init__(self, pipeline: PixArtAlphaPipeline, module_config: DistriConfig):
        self.pipeline = pipeline

        # assert module_config.do_classifier_free_guidance == False
        assert module_config.split_batch == False

        self.distri_config = module_config

        self.static_inputs = None

        self.prepare()

    @staticmethod
    def from_pretrained(distri_config: DistriConfig, **kwargs):
        device = distri_config.device
        pretrained_model_name_or_path = kwargs.pop(
            "pretrained_model_name_or_path", "PixArt-alpha/PixArt-XL-2-1024-MS"
        )
        torch_dtype = kwargs.pop("torch_dtype", torch.float16)
        transformer = Transformer2DModel.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, subfolder="transformer"
        ).to(device)

        if distri_config.parallelism == "patch":
            transformer = DistriDiTPP(transformer, distri_config)
        elif distri_config.parallelism == "naive_patch":
            logger.info("Using naive patch parallelism")
            transformer = NaivePatchDiT(transformer, distri_config)
        else:
            raise ValueError(f"Unknown parallelism: {distri_config.parallelism}")

        pipeline = PixArtAlphaPipeline.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, transformer=transformer, **kwargs
        ).to(device)
        return DistriPixArtAlphaPipeline(pipeline, distri_config)

    def set_progress_bar_config(self, **kwargs):
        pass

    @torch.no_grad()
    def __call__(self, prompt, *args, **kwargs):
        self.pipeline.transformer.set_counter(0)
        return self.pipeline(prompt=prompt, *args, **kwargs)

    @torch.no_grad()
    def prepare(self, **kwargs):
        distri_config = self.distri_config

        static_inputs = {}
        static_outputs = []
        pipeline = self.pipeline

        height = distri_config.height
        width = distri_config.width
        assert height % 8 == 0 and width % 8 == 0

        device = distri_config.device

        (prompt_embeds, prompt_attention_mask, 
        negative_prompt_embeds, negative_prompt_attention_mask) = pipeline.encode_prompt(
            prompt="",
            do_classifier_free_guidance=distri_config.do_classifier_free_guidance,
            device=device,
        )
        batch_size = 1

        if distri_config.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        num_channels_latents = pipeline.transformer.config.in_channels
        latents = pipeline.prepare_latents(
            batch_size, num_channels_latents, height, width, prompt_embeds.dtype, device, None
        )

        # 7. Prepare added time ids & embeddings


        t = torch.zeros([batch_size], device=device, dtype=torch.long)
        latent_model_input = torch.cat([latents] * 2) if distri_config.do_classifier_free_guidance else latents
        static_inputs["hidden_states"] = latent_model_input
        static_inputs["timestep"] = t
        static_inputs["encoder_hidden_states"] = prompt_embeds
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        if pipeline.transformer.config.sample_size == 128:
            resolution = torch.tensor([height, width]).repeat(batch_size * 1, 1)
            aspect_ratio = torch.tensor([float(height / width)]).repeat(batch_size * 1, 1)
            resolution = resolution.to(dtype=prompt_embeds.dtype, device=device)
            aspect_ratio = aspect_ratio.to(dtype=prompt_embeds.dtype, device=device)
            if distri_config.do_classifier_free_guidance:
                resolution = torch.cat([resolution, resolution], dim=0)
                aspect_ratio = torch.cat([aspect_ratio, aspect_ratio], dim=0)

            added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}
        static_inputs["added_cond_kwargs"] = added_cond_kwargs

        # Used to create communication buffer
        comm_manager = None
        if distri_config.n_device_per_batch > 1:
            comm_manager = PatchParallelismCommManager(distri_config)
            pipeline.transformer.set_comm_manager(comm_manager)

            # Only used for creating the communication buffer
            pipeline.transformer.set_counter(0)
            pipeline.transformer(**static_inputs, return_dict=False)
            if comm_manager.numel > 0:
                comm_manager.create_buffer()

        # Pre-run
        # pipeline.transformer.set_counter(0)
        # pipeline.transformer(**static_inputs, return_dict=False)

        # self.static_inputs = static_inputs
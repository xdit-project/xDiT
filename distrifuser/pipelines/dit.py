import torch
from diffusers import DiTPipeline
from distrifuser.models.distri_transformer_2d import DistriTransformer2DModel

# from distrifuser.models.distri_sdxl_unet_pp import DistriSDXLUNetPP
# from distrifuser.models.distri_sdxl_unet_tp import DistriSDXLUNetTP
from distrifuser.models.naive_patch_dit import NaivePatchDiT
from distrifuser.utils import DistriConfig, PatchParallelismCommManager
from distrifuser.logger import init_logger

logger = init_logger(__name__)

class DistriDiTPipeline:
    def __init__(self, pipeline: DiTPipeline, module_config: DistriConfig):
        self.pipeline = pipeline

        assert module_config.do_classifier_free_guidance == False
        assert module_config.split_batch == False

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
        transformer = DistriTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, subfolder="transformer"
        ).to(device)

        transformer = NaivePatchDiT(transformer, distri_config)
        # if distri_config.parallelism == "patch":
        #     # unet = DistriSDXLUNetPP(unet, distri_config)
        #     # raise ValueError("Patch parallelism is not supported for DiT")
        #     pass
        # elif distri_config.parallelism == "tensor":
        #     # unet = DistriSDXLUNetTP(unet, distri_config)
        #     raise ValueError("Tensor parallelism is not supported for DiT")
        # elif distri_config.parallelism == "naive_patch":
        #     unet = NaivePatchDiT(unet, distri_config)
        # else:
        #     raise ValueError(f"Unknown parallelism: {distri_config.parallelism}")

        pipeline = DiTPipeline.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, transformer=transformer, **kwargs
        ).to(device)
        return DistriDiTPipeline(pipeline, distri_config)

    def set_progress_bar_config(self, **kwargs):
        pass

    @torch.no_grad()
    def __call__(self, words, *args, **kwargs):
        class_ids = self.pipeline.get_label_ids(words)
        self.pipeline.transformer.set_counter(0)
        return self.pipeline(class_labels=class_ids, *args, **kwargs)

    @torch.no_grad()
    def prepare(self, **kwargs):
        distri_config = self.distri_config

        static_inputs = {}
        static_outputs = []
        pipeline = self.pipeline

        height = distri_config.height
        width = distri_config.width
        assert height % 8 == 0 and width % 8 == 0

        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)

        device = distri_config.device

        batch_size = 2 if distri_config.do_classifier_free_guidance else 1

        # 7. Prepare added time ids & embeddings

        t = torch.zeros([batch_size], device=device, dtype=torch.long)

        # static_inputs["hidden_states"] = latents
        # static_inputs["timestep"] = t
        # static_inputs["encoder_hidden_states"] = prompt_embeds
        # static_inputs["added_cond_kwargs"] = added_cond_kwargs

        # Used to create communication buffer
        comm_manager = None
        if distri_config.n_device_per_batch > 1:
            comm_manager = PatchParallelismCommManager(distri_config)
            pipeline.transformer.set_comm_manager(comm_manager)

            # Only used for creating the communication buffer
            pipeline.transformer.set_counter(0)
            # pipeline.transformer(**static_inputs, return_dict=False)
            if comm_manager.numel > 0:
                comm_manager.create_buffer()

        # Pre-run
        pipeline.transformer.set_counter(0)
        # pipeline.transformer(**static_inputs, return_dict=False)

        # self.static_inputs = static_inputs

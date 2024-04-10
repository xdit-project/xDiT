import torch
from diffusers import DiTPipeline

# lib impl
# from diffusers.models.transformers.transformer_2d import Transformer2DModel

# customized impl
from distrifuser.models.diffusers import Transformer2DModel


# from distrifuser.models.distri_sdxl_unet_tp import DistriSDXLUNetTP
from distrifuser.models import NaivePatchDiT, DistriDiTPP
from distrifuser.utils import DistriConfig, PatchParallelismCommManager
from distrifuser.logger import init_logger
from typing import Union, List

logger = init_logger(__name__)


class DistriDiTPipeline:
    def __init__(self, pipeline: DiTPipeline, module_config: DistriConfig):
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
            "pretrained_model_name_or_path", "facebook/DiT-XL-2-256"
        )
        torch_dtype = kwargs.pop("torch_dtype", torch.float16)
        transformer = Transformer2DModel.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            subfolder="transformer",
        ).to(device)

        logger.info(f"Using {distri_config.parallelism } parallelism")
        if distri_config.parallelism == "patch":
            transformer = DistriDiTPP(transformer, distri_config)
        elif distri_config.parallelism == "tensor":
            raise NotImplementedError("Tensor parallelism is not supported for DiT")
        elif distri_config.parallelism == "naive_patch":
            transformer = NaivePatchDiT(transformer, distri_config)
        else:
            raise ValueError(f"Unknown parallelism: {distri_config.parallelism}")

        pipeline = DiTPipeline.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            transformer=transformer,
            **kwargs,
        ).to(device)
        return DistriDiTPipeline(pipeline, distri_config)

    def set_progress_bar_config(self, **kwargs):
        self.pipeline.set_progress_bar_config(**kwargs)

    @torch.no_grad()
    def __call__(self, prompt: Union[List[str], str, List[int], int], *args, **kwargs):
        self.pipeline.transformer.set_counter(0)
        if isinstance(prompt, str):
            class_ids = self.pipeline.get_label_ids([prompt])
        elif isinstance(prompt, int):
            class_ids = [prompt]
        elif isinstance(prompt, list):
            if isinstance(prompt[0], str):
                class_ids = self.pipeline.get_label_ids(prompt)
            elif isinstance(prompt[0], int):
                class_ids = prompt
            else:
                raise ValueError("Invalid prompt type")
        else:
            raise ValueError("Invalid prompt type")
        return self.pipeline(class_labels=class_ids, *args, **kwargs)

    @torch.no_grad()
    def prepare(self, **kwargs):
        distri_config = self.distri_config

        static_inputs = {}
        static_outputs = []
        cuda_graphs = []
        pipeline = self.pipeline

        height = distri_config.height
        width = distri_config.width
        assert height % 8 == 0 and width % 8 == 0

        # original_size = (height, width)
        # target_size = (height, width)
        # crops_coords_top_left = (0, 0)

        device = distri_config.device
        batch_size = 1 if distri_config.batch_size is None else distri_config.batch_size

        # 7. Prepare added time ids & embeddings

        t = torch.zeros(2, device=device, dtype=torch.long)

        guidance_scale = 4.0
        latent_size = pipeline.transformer.config.sample_size
        latent_channels = pipeline.transformer.config.in_channels
        latents = torch.zeros(
            [batch_size, latent_channels, latent_size, latent_size],
            device=device,
            dtype=pipeline.transformer.dtype,
        )
        batch_size = 1
        class_labels = torch.tensor([0] * batch_size, device=device).reshape(-1)
        class_null = torch.tensor([1000] * batch_size, device=device)
        class_labels_input = (
            torch.cat([class_labels, class_null], 0)
            if guidance_scale > 1
            else class_labels
        )
        latent_model_input = (
            torch.cat([latents, latents], 0) if guidance_scale > 1 else latents
        )
        static_inputs["hidden_states"] = latent_model_input
        static_inputs["timestep"] = t
        static_inputs["class_labels"] = class_labels_input

        # Used to create communication buffer
        comm_manager = None
        if distri_config.n_device_per_batch > 1:
            comm_manager = PatchParallelismCommManager(distri_config)
            pipeline.transformer.set_comm_manager(comm_manager)

            # Only used for creating the communication buffer
            pipeline.transformer.set_counter(0)
            pipeline.transformer(**static_inputs, return_dict=False, record=True)
            if comm_manager.numel > 0:
                comm_manager.create_buffer()

        # Pre-run
        pipeline.transformer.set_counter(0)
        pipeline.transformer(**static_inputs, return_dict=False, record=True)

        if distri_config.use_cuda_graph:
            if comm_manager is not None:
                comm_manager.clear()
            if distri_config.parallelism == "naive_patch":
                counters = [0, 1]
            elif distri_config.parallelism == "patch":
                counters = [0, distri_config.warmup_steps + 1, distri_config.warmup_steps + 2]
            else:
                raise ValueError(f"Unknown parallelism: {distri_config.parallelism}")
            for counter in counters:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    pipeline.transformer.set_counter(counter)
                    output = pipeline.transformer(**static_inputs, return_dict=False, record=True)[0]
                    static_outputs.append(output)
                cuda_graphs.append(graph)
            pipeline.transformer.setup_cuda_graph(static_outputs, cuda_graphs)

        self.static_inputs = static_inputs

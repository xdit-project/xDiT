import logging
import os
import time
import torch
import torch.distributed
from diffusers.pipelines.consisid.consisid_utils import prepare_face_models, process_face_embeddings_infer
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download
from xfuser import xFuserConsisIDPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_runtime_state,
    is_dp_last_group,
)


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)

    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank

    assert engine_args.use_parallel_vae is False, "parallel VAE not implemented for ConsisID"

    # 1. Prepare all the Checkpoints
    if not os.path.exists(engine_config.model_config.model):
        print("Base Model not found, downloading from Hugging Face...")
        snapshot_download(repo_id="BestWishYsh/ConsisID-preview", local_dir=engine_config.model_config.model)
    else:
        print(f"Base Model already exists in {engine_config.model_config.model}, skipping download.")

    # 2. Load Pipeline.
    device = torch.device(f"cuda:{local_rank}")
    pipe = xFuserConsisIDPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    )
    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    elif args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} model CPU offload enabled")
    else:
        pipe = pipe.to(device)

    face_helper_1, face_helper_2, face_clip_model, face_main_model, eva_transform_mean, eva_transform_std = (
        prepare_face_models(engine_config.model_config.model, device=device, dtype=torch.bfloat16)
    )

    if args.enable_tiling:
        pipe.vae.enable_tiling()

    if args.enable_slicing:
        pipe.vae.enable_slicing()
    
    # 3. Prepare Model Input
    id_cond, id_vit_hidden, image, face_kps = process_face_embeddings_infer(
                 face_helper_1,
                 face_clip_model,
                 face_helper_2,
                 eva_transform_mean,
                 eva_transform_std,
                 face_main_model,
                 device,
                 torch.bfloat16,
                 input_config.img_file_path,
                 is_align_face=True,
             )

    # 4. Generate Identity-Preserving Video
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    output = pipe(
        image=image,
        prompt=input_config.prompt[0],
        id_vit_hidden=id_vit_hidden,
        id_cond=id_cond,
        kps_cond=face_kps,
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        num_inference_steps=input_config.num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        guidance_scale=6.0,
        use_dynamic_cfg=False,
    ).frames[0]

    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )
    if is_dp_last_group():
        resolution = f"{input_config.width}x{input_config.height}"
        output_filename = f"results/consisid_{parallel_info}_{resolution}.mp4"
        export_to_video(output, output_filename, fps=8)
        print(f"output saved to {output_filename}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(f"epoch time: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB")
    get_runtime_state().destory_distributed_env()


if __name__ == "__main__":
    main()

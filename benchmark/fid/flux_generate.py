import logging
import torch
import torch.distributed
import json, os
from xfuser import xFuserFluxPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_runtime_state
)
import gc


_NUM_FID_CANDIDATE = 30000
CFG = 1.5

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def main():
    parser = FlexibleArgumentParser(description='xFuser Arguments')
    parser.add_argument('--caption_file', type=str, default='captions_coco.json')
    parser.add_argument('--sample_images_folder', type=str, default='sample_images')
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16
    local_rank = get_world_group().local_rank

    pipe = xFuserFluxPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    )

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f'rank {local_rank} sequential CPU offload enabled')
    else:
        pipe = pipe.to(f'cuda:{local_rank}')

    pipe.prepare_run(input_config, steps=1)

    with open(args.caption_file) as f:
        raw_captions = json.load(f)

    raw_captions = raw_captions['images'][:_NUM_FID_CANDIDATE]
    captions = list(map(lambda x: x['sentences'][0]['raw'], raw_captions))
    filenames = list(map(lambda x: x['filename'], raw_captions))
    
    folder_path = args.sample_images_folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # run multiple prompts at a time to save time
    num_prompt_one_step = 120
    for j in range(0, _NUM_FID_CANDIDATE, num_prompt_one_step):
        output = pipe(
            height=256,
            width=256,
            prompt=captions[j:j+num_prompt_one_step],
            num_inference_steps=input_config.num_inference_steps,
            output_type=input_config.output_type,
            max_sequence_length=256,
            guidance_scale=CFG,
            generator=torch.Generator(device='cuda').manual_seed(input_config.seed),
        )
        if input_config.output_type == 'pil':
            if pipe.is_dp_last_group():
                for k, local_filename in enumerate(filenames[j:j+num_prompt_one_step]):
                    output.images[k].save(f'{folder_path}/{local_filename}')
        print(f'{j}-{j+num_prompt_one_step-1} generation finished!')
        flush()

    get_runtime_state().destroy_distributed_env()


if __name__ == '__main__':
    main()

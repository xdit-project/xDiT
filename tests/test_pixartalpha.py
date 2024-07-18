import time
import torch
import torch.distributed
from pipefuser.pipelines import PipeFuserPixArtAlphaPipeline
from pipefuser.config import EngineArgs, FlexibleArgumentParser
from pipefuser.distributed import get_world_group


def main():
    parser = FlexibleArgumentParser(description="PipeFuser PixArt Alpha Test Args")
    args = EngineArgs.add_cli_args(parser).parse_args()
    engine_args = EngineArgs.from_cli_args(args)
    engine_config = engine_args.create_engine_config()
    local_rank = get_world_group().local_rank
    pipe = PipeFuserPixArtAlphaPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        parallel_config=engine_config.parallel_config,
        runtime_config=engine_config.runtime_config,
        torch_dtype=torch.float16,
    ).to(f"cuda:{local_rank}")
    pipe.set_input_config(engine_config.input_config)
    pipe.prepare_run()

    start_time = time.time()
    output = pipe(
        prompt=engine_config.input_config.prompt,
        num_inference_steps=engine_config.input_config.num_inference_steps,
        output_type="pil",
        use_resolution_binning=engine_config.input_config.use_resolution_binning,
        num_pipeline_warmup_steps=engine_config.runtime_config.warmup_steps,
        generator=torch.Generator(device="cuda").manual_seed(engine_config.runtime_config.seed),
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device="cuda")

    if get_world_group().rank == get_world_group().world_size - 1:
        output.images[0].save("./results/new_split_batch.png")
        print(
            f"epoch time: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB"
        )


if __name__ == '__main__':
    main()
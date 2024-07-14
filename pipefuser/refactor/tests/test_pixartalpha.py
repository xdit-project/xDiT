import torch
import torch.distributed
from pipefuser.refactor.pipelines import PipeFuserPixArtAlphaPipeline
from pipefuser.refactor.config.args import EngineArgs, FlexibleArgumentParser
from pipefuser.refactor.distributed.parallel_state import get_world_group


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
    output = pipe(
        prompt=engine_config.input_config.prompt,
        generator=torch.Generator(device="cuda").manual_seed(engine_config.runtime_config.seed),
        num_inference_steps=engine_config.input_config.num_inference_steps,
        output_type="pil",
    )
    if get_world_group().rank == 1:
        output.images[0].save("./results/new_split_batch.png")


if __name__ == '__main__':
    main()
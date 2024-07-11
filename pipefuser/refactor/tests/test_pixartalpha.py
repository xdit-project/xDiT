import torch
from pipefuser.refactor.pipelines import PipeFuserPixArtAlphaPipeline
from pipefuser.refactor.config.config import EngineConfig
from pipefuser.refactor.config.args import EngineArgs, FlexibleArgumentParser


def main():
    parser = FlexibleArgumentParser(description="PipeFuser PixArt Alpha Test Args")
    args = EngineArgs.add_cli_args(parser).parse_args()
    engine_args = EngineArgs.from_cli_args(args)
    engine_config = engine_args.create_engine_config()
    pipe = PipeFuserPixArtAlphaPipeline.from_pretrained(
        engine_config.model_config.model,
        engine_config=engine_config
    )
    pipe(
        prompt="A beautiful sunset over the ocean.",
        generator=torch.Generator(device="cuda").manual_seed(engine_config.runtime_config.seed),
        num_inference_steps=engine_config.input_config.num_inference_steps,
        output_type="image",
    )


if __name__ == '__main__':
    main()
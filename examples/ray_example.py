import time
import os
import torch
import torch.distributed
from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.executor.gpu_executor import RayGPUExecutor

def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    executor = RayGPUExecutor(engine_config)
    executor.init_distributed_environment()
    executor.load_model(engine_config)
    executor.execute(input_config)

    # get_runtime_state().destory_distributed_env()


if __name__ == "__main__":
    main()

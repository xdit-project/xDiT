import json
import subprocess
import sys
import os
from pathlib import Path
import argparse
import shlex

WD: str = Path(__file__).parent.absolute()
#os.environ["PYTHONPATH"] = f"{WD}:{os.getenv('PYTHONPATH', '')}"
CONFIG_PATH: str = WD / "config.json"
HOST_SCRIPT: str = WD / "host.py"

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def build_command(config):
    cmd: str = (
        f"{sys.executable} -m torch.distributed.run --nproc_per_node={config['nproc_per_node']} {HOST_SCRIPT.as_posix()} "
        f"--model={config['model']} "
        f"--pipefusion_parallel_degree={config['pipefusion_parallel_degree']} "
        f"--ulysses_degree={config['ulysses_degree']} "
        f"--ring_degree={config['ring_degree']} "
        f"--height={config['height']} "
        f"--width={config['width']} "
        f"--max_queue_size={config.get('max_queue_size', 4)} "
        f"--use_torch_compile"
    )
    if config.get("use_cfg_parallel", False):
        cmd.append("--use_cfg_parallel")
    #return [arg for arg in cmd if arg]  # Remove any empty strings
    return cmd

def main():
    parser = argparse.ArgumentParser(description="Launch host with configuration")
    parser.add_argument(
        "--config",
        type=str,
        default=CONFIG_PATH.as_posix(),
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    cmd = build_command(config)
    cmd = shlex.split(cmd)
    print(cmd)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

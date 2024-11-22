import json
import subprocess
import sys
import argparse


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def build_command(config):
    cmd = [
        "torchrun",
        f"--nproc_per_node={config['nproc_per_node']}",
        "./http-service/host.py",
        f"--model={config['model']}",
        f"--pipefusion_parallel_degree={config['pipefusion_parallel_degree']}",
        f"--ulysses_degree={config['ulysses_degree']}",
        f"--ring_degree={config['ring_degree']}",
        f"--height={config['height']}",
        f"--width={config['width']}",
        f"--max_queue_size={config.get('max_queue_size', 4)}",
    ]
    if config.get("use_cfg_parallel", False):
        cmd.append("--use_cfg_parallel")
    return [arg for arg in cmd if arg]  # Remove any empty strings


def main():
    parser = argparse.ArgumentParser(description="Launch host with configuration")
    parser.add_argument(
        "--config",
        type=str,
        default="./http-service/config.json",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    cmd = build_command(config)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

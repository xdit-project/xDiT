import math
import os
import subprocess
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


def run_command(cmd):
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    output = ""
    for line in process.stdout:
        if "epoch time:" in line or "Running test for size" in line:
            print(line.strip())
        output += line
    process.wait()
    if process.returncode != 0:
        print(f"Command failed: {cmd}")
        print(output + "\n")
    # subprocess.run(cmd, shell=True, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run benchmark tests")
    parser.add_argument("--model_id", type=str, required=True, help="Path to the model")
    parser.add_argument(
        "--sizes", type=int, nargs="+", required=True, help="List of sizes to test"
    )
    parser.add_argument(
        "--script",
        type=str,
        required=True,
        help="Script to run (e.g., tests/test_pixartalpha.py)",
    )
    parser.add_argument(
        "--n_gpus", type=int, nargs="+", required=True, help="Number of GPUs to use"
    )
    parser.add_argument("--steps", type=int, default=20, help="Number of steps")
    args = parser.parse_args()
    MODEL_ID = args.model_id
    SIZES = args.sizes
    SCRIPT = args.script
    N_GPUS = args.n_gpus
    STEPS = args.steps

    for size in SIZES:
        for num_gpus in N_GPUS:
            for i in range(int(math.log2(num_gpus)) + 1):
                ulysses_degree = int(math.pow(2, i))
                ring_degree = num_gpus // ulysses_degree

                print(
                    f"Running test for size {size}, ulysses_degree {ulysses_degree}, ring_degree {ring_degree}",
                    flush=True,
                )
                cmd = (
                    f"torchrun --nproc_per_node={num_gpus} {SCRIPT} --prompt 'A small cat' --output_type 'latent' --model {MODEL_ID} "
                    f"--height {size} --width {size} --ulysses_degree {ulysses_degree} --ring_degree {ring_degree} --num_inference_steps {STEPS}"
                )

                run_command(cmd)


if __name__ == "__main__":
    main()

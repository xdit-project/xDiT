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
    parser.add_argument("--n_gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument(
        "--no_use_resolution_binning",
        action="store_true",
        help="Do not use resolution binning",
    )
    args = parser.parse_args()
    MODEL_ID = args.model_id
    SIZES = args.sizes
    SCRIPT = args.script
    N_GPUS = args.n_gpus
    RESOLUTION_BINNING = (
        "--no_use_resolution_binning" if args.no_use_resolution_binning else ""
    )

    visited = set()
    dp_degree = 1
    for size in SIZES:
        for cfg_degree in [1, 2]:
            model_parallel_degree = N_GPUS // cfg_degree
            for i in range(int(math.log2(model_parallel_degree)) + 1):
                pp_degree = int(math.pow(2, i))
                sp_degree = model_parallel_degree // pp_degree
                for i in range(int(math.log2(sp_degree)) + 1):
                    ulysses_degree = int(math.pow(2, i))
                    ring_degree = sp_degree // ulysses_degree
                    last_num_patch = None
                    for num_pipeline_patches in [
                        max(1, pp_degree // 2),
                        pp_degree,
                        pp_degree * 2,
                    ]:
                        for warmup_step in [0, 1, 2]:
                            # pp_degree = 1, no warmup and num_pipeline_patches
                            if pp_degree == 1 and (
                                warmup_step > 0 or num_pipeline_patches > 1
                            ):
                                continue

                            if (
                                size,
                                pp_degree,
                                ulysses_degree,
                                ring_degree,
                                num_pipeline_patches,
                                warmup_step,
                            ) in visited:
                                continue

                            visited.add(
                                (
                                    size,
                                    pp_degree,
                                    ulysses_degree,
                                    ring_degree,
                                    num_pipeline_patches,
                                    warmup_step,
                                )
                            )
                            # if num_pipeline_patches != last_num_patch:
                            if cfg_degree == 2:
                                print(
                                    f"Running test for size {size}, split batch, warmup_step {warmup_step}, pp_degree {pp_degree}, ulysses_degree {ulysses_degree}, ring_degree {ring_degree}, num_pipeline_patches {num_pipeline_patches}",
                                    flush=True,
                                )
                                cmd = (
                                    f"torchrun --nproc_per_node={N_GPUS} {SCRIPT} --prompt 'A small cat' --output_type 'latent' --model {MODEL_ID} "
                                    f"--height {size} --width {size} --warmup_steps {warmup_step} "
                                    f"{RESOLUTION_BINNING} --use_cfg_parallel --ulysses_degree {ulysses_degree} --ring_degree {ring_degree} "
                                    f"--pipefusion_parallel_degree {pp_degree} --num_pipeline_patch {num_pipeline_patches}"
                                )
                                run_command(cmd)
                            else:
                                print(
                                    f"Running test for size {size}, no split batch, warmup_step {warmup_step}, pp_degree {pp_degree}, ulysses_degree {ulysses_degree}, ring_degree {ring_degree}, num_pipeline_patches {num_pipeline_patches}",
                                    flush=True,
                                )
                                cmd = (
                                    f"torchrun --nproc_per_node={N_GPUS} {SCRIPT} --prompt 'A small cat' --output_type 'latent' --model {MODEL_ID} "
                                    f"--height {size} --width {size} --warmup_steps {warmup_step} "
                                    f"{RESOLUTION_BINNING} --ulysses_degree {ulysses_degree} --ring_degree {ring_degree} "
                                    f"--pipefusion_parallel_degree {pp_degree} --num_pipeline_patch {num_pipeline_patches} "
                                )

                                run_command(cmd)


if __name__ == "__main__":
    main()

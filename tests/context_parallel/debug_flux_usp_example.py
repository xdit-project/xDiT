import os
import sys
import subprocess
import shlex
from pathlib import Path

os.environ["HF_HUB_CACHE"] = "/mnt/co-research/shared-models/hub"

root_dir = Path(__file__).parents[2].absolute()
#os.environ["PYTHONPATH"] = f"{WD}:{os.getenv('PYTHONPATH', '')}"
examples_dir = root_dir / "examples"
flux_script = examples_dir / "flux_usp_example.py"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
n_gpus = 2

model_id = "black-forest-labs/FLUX.1-dev"
inference_steps = 28
warmup_steps = 3
max_sequence_length = 512
height = 1024
width = 1024
task_args = f"--max-sequence-length {max_sequence_length} --height {height} --width {width}"
pipefusion_parallel_degree = 1
ulysses_degree = 2
ring_degree = 1
parallel_args = (
    f"--pipefusion_parallel_degree {pipefusion_parallel_degree} "
    f"--ulysses_degree {ulysses_degree} "
    f"--ring_degree {ring_degree} "
)
compile_flag = "--use_torch_compile"

cmd: str = (
    f"{sys.executable} -m torch.distributed.run --nproc_per_node={n_gpus} {flux_script.as_posix()} "
    f"--model {model_id} "
    f"{parallel_args} "
    f"{task_args} "
    f"--num_inference_steps {inference_steps} "
    f"--warmup_steps {warmup_steps} "
    f"--prompt \"A dark tree.\" "
)
cmd = shlex.split(cmd)
print(cmd)
subprocess.run(cmd, check=True)
import os
import sys
import subprocess
import shlex
from pathlib import Path

wd: str = Path(__file__).parent.absolute()
#os.environ["PYTHONPATH"] = f"{WD}:{os.getenv('PYTHONPATH', '')}"
test_script: str = wd / "test_diffusers_adapters.py"
model_test: str = "FluxPipelineTest"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

cmd: str = (
    f"{sys.executable} -m pytest {test_script.as_posix()}::{model_test}"
)
cmd = shlex.split(cmd)
print(cmd)
subprocess.run(cmd, check=True)
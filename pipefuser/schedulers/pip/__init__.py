from packaging.version import Version
import diffusers

if Version('0.29.0') <= Version(diffusers.__version__):
    from .flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerPiP

from .ddim import DDIMSchedulerPiP
from .dpmsolver_multistep import DPMSolverMultistepSchedulerPiP

if Version('0.30.0.dev0') <= Version(diffusers.__version__):
    from .ddpm import DDPMSchedulerPiP

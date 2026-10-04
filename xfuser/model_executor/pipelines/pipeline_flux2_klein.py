"""xFuser pipeline wrapper for FLUX.2 klein, with PipeFusion support.

Kept apart from pipeline_flux2.py because Flux2KleinPipeline landed in diffusers 0.37,
one release after Flux2Pipeline: importing this module is the availability test for the
klein wrapper alone, so FLUX.2 dev PipeFusion still loads on 0.36.
"""

from diffusers import Flux2KleinPipeline

from .pipeline_flux2 import xFuserFlux2PipelineBase
from .register import xFuserPipelineWrapperRegister


@xFuserPipelineWrapperRegister.register(Flux2KleinPipeline)
class xFuserFlux2KleinPipeline(xFuserFlux2PipelineBase):
    """Klein differs only in the diffusers class it binds; the PipeFusion logic is shared."""

    _diffusers_cls = Flux2KleinPipeline

"""xDiT changes to the FLUX.3 modular pipeline.

Imported lazily by the FLUX.3 runners: the blocks exist only in diffusers builds with FLUX.3.
"""

from xfuser.core.utils.runner_utils import log


def prepare_flux3_pipeline(pipe, pretrained_model_name_or_path: str):
    """Rebuild a FLUX.3 modular pipeline without the prompt-upsampler step.

    Newer diffusers puts a prompt-upsampler step first, which runs a separate 9B model
    (black-forest-labs/flux-3-image-klein-pup) on every call. xDiT encodes prompts as
    given; upsample them before the run.

    ``pipe.blocks`` is a copy, so the edited blocks need a new pipeline of the same class.
    Call this before loading any component.
    """
    blocks = pipe.blocks
    if blocks.sub_blocks.pop("prompt_upsampler", None) is None:
        return pipe
    log("FLUX.3: prompt upsampler step removed; prompts are encoded as given (the model expects full scene captions).")
    return type(pipe)(blocks=blocks, pretrained_model_name_or_path=pretrained_model_name_or_path)

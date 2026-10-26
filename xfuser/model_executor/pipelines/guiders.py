"""CFG parallel for diffusers modular pipelines, as a guider.

Imported lazily: ``diffusers.guiders`` is only in diffusers builds with modular pipelines.
"""

from diffusers.guiders import ClassifierFreeGuidance

from xfuser.core.distributed import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)


class xFuserCFGParallelGuidance(ClassifierFreeGuidance):
    """ClassifierFreeGuidance with its branches split across the CFG-parallel group.

    Modular denoise loops ask the guider for the branch batches, run the denoiser once
    per batch, then hand the batches back to the guider to combine. This guider returns
    only this rank's branch (rank 0 conditional, rank 1 unconditional), so each rank runs
    one denoiser call per step, and all-gathers the predictions before the usual combine.
    The predictions share a shape across branches even when the branches' inputs do not
    (FLUX.3's prompt and negative prompt have different text lengths), so this computes
    exactly what the sequential loop does. The denoise blocks stay upstream's.

    Swap it in with ``pipe.update_components(guider=xFuserCFGParallelGuidance.from_config(pipe.guider.config))``.
    """

    _all_batches = None

    def _local_branch(self, batches: list) -> list:
        world_size = get_classifier_free_guidance_world_size()
        if world_size == 1:
            return batches
        if len(batches) != world_size:
            raise ValueError(
                f"CFG parallel splits {world_size} guidance branches, but step {self._step} has {len(batches)}; "
                "a guidance scale of 1 disables classifier-free guidance."
            )
        rank = get_classifier_free_guidance_rank()
        self._all_batches = batches
        # Other ranks prepare the branches before this one; count them, so is_conditional
        # describes this rank's branch while its denoiser call runs.
        self._count_prepared += rank
        return [batches[rank]]

    def prepare_inputs(self, data):
        return self._local_branch(super().prepare_inputs(data))

    def prepare_inputs_from_block_state(self, data, input_fields):
        return self._local_branch(super().prepare_inputs_from_block_state(data, input_fields))

    def __call__(self, data: list):
        batches, self._all_batches = self._all_batches, None
        if batches is None:
            return super().__call__(data)
        (local,) = data
        predictions = get_cfg_group().all_gather(local.noise_pred.contiguous(), separate_tensors=True)
        for batch, prediction in zip(batches, predictions):
            batch.noise_pred = prediction
        return super().__call__(batches)

"""Distributed and PipeFusion cache-context integration for Cache-DiT."""

import copy
import types

import torch
import torch.distributed as dist


def is_parallelized_flag() -> bool:
    """Return whether one cache decision spans replicated sequence ranks."""
    from xfuser.core.distributed import get_sequence_parallel_world_size

    return get_sequence_parallel_world_size() > 1


def install_pipefusion_static_fn_bypass(
    transformer: torch.nn.Module,
    cache_config,
) -> bool:
    """Make a static PipeFusion schedule cache the complete local stage.

    Cache-DiT normally computes the configured Fn prefix before consulting a
    static step mask. Its config requires Fn >= 1, so retain the validated
    configuration but make the runtime accessor return zero. All local blocks
    then belong to Mn: compute steps refresh the complete stage residual and
    scheduled cache steps apply it without executing Fn.
    """
    if getattr(cache_config, "steps_computation_policy", None) != "static" or not getattr(
        cache_config, "steps_computation_mask", None
    ):
        return False

    manager = getattr(transformer, "_context_manager", None)
    if manager is None:
        raise RuntimeError(
            "Cache-DiT did not expose _context_manager after enable_cache; "
            "cannot install static PipeFusion stage reuse."
        )
    if getattr(manager, "_xdit_static_fn_bypass_installed", False):
        return True

    @torch.compiler.disable
    def no_fn_blocks(self):
        return 0

    manager.Fn_compute_blocks = types.MethodType(no_fn_blocks, manager)
    manager._xdit_static_fn_bypass_installed = True
    return True


def _shift_patch_cache_config(cache_config, warmup_steps: int):
    """Align one patch context's schedule with diffusion steps after warmup.

    Cache-DiT indexes step schedules by the context's own counter. Patch contexts
    start at zero when PipeFusion switches to patch mode, so give them a copy of
    the config with schedule fields advanced by the full-sequence warmup. The
    copy is required because Cache-DiT keeps the config by reference. The first
    patch step still computes: its residual buffers are empty, since warmup ran
    on the base context.

    Forced refreshes are rejected: Cache-DiT rebuilds a refreshed context from
    this shifted config, so the schedule after the refresh would be offset.
    """
    if warmup_steps <= 0:
        return cache_config
    if cache_config.force_refresh_step_hint is not None:
        raise ValueError(
            "cache_config.force_refresh_step_hint is not supported with PipeFusion warmup (--warmup_steps > 0)."
        )
    mask = cache_config.steps_computation_mask
    warmup_horizon = cache_config.max_warmup_steps
    if mask is not None:
        first_continuous_compute_steps = 0
        for value in mask:
            if value != 1:
                break
            first_continuous_compute_steps += 1
        warmup_horizon = min(warmup_horizon, first_continuous_compute_steps)
    if (
        cache_config.warmup_interval != 1
        and warmup_steps < warmup_horizon
        and warmup_steps % cache_config.warmup_interval
    ):
        raise ValueError(
            "PipeFusion cache warmup is not phase-aligned: warmup_steps must be "
            "divisible by cache_config.warmup_interval while scheduled warmup steps remain."
        )
    shifted = copy.deepcopy(cache_config)
    if mask:
        sliced = list(mask[warmup_steps:]) or [1]
        sliced[0] = 1
        shifted.steps_computation_mask = sliced
    shifted.max_warmup_steps = max(cache_config.max_warmup_steps - warmup_steps, 1)
    if cache_config.num_inference_steps is not None:
        shifted.num_inference_steps = max(cache_config.num_inference_steps - warmup_steps, 1)
    return shifted


def install_cache_decision_sync(transformer: torch.nn.Module) -> None:
    """Synchronize one transformer's decisions over the correct replica group.

    Pipeline stages own different blocks and therefore decide independently.
    Within a stage, PipeFusion may let patch 0 choose the path reused by later
    patches. A PP-only stage needs no collective. If sequence parallelism is
    composed with PP, only replicas of that stage synchronize.
    """
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return

    manager = getattr(transformer, "_context_manager", None)
    if manager is None:
        raise RuntimeError(
            "Cache-DiT did not expose _context_manager after enable_cache; cannot synchronize cache decisions."
        )
    if getattr(manager, "_xdit_cache_decision_sync_installed", False):
        return

    original_can_cache = manager.can_cache

    @torch.compiler.disable
    def can_cache(self, *args, **kwargs):
        from xfuser.core.distributed import (
            get_pipeline_parallel_world_size,
            get_sequence_parallel_world_size,
        )

        pipefusion = get_pipeline_parallel_world_size() > 1
        sequence_parallel = get_sequence_parallel_world_size() > 1
        if pipefusion:
            # Cache-DiT's built-in parallelized path synchronizes over WORLD.
            # PipeFusion stages are not replicas, so synchronize below only
            # when this stage is itself sequence-sharded.
            kwargs["parallelized"] = False

        leader_key = None
        shared = False
        result = None
        if pipefusion and getattr(self, "_xdit_share_pipefusion_decisions", False):
            from xfuser.core.distributed import get_runtime_state

            state = get_runtime_state()
            if state.patch_mode:
                context_name = getattr(self.get_context(), "name", "")
                context_name = context_name.split(":pipefusion_patch_", 1)[0]
                leader_key = (
                    context_name,
                    self.get_current_step(),
                    kwargs.get("prefix", "Fn"),
                    (bool(self.is_separate_cfg_step()) if hasattr(self, "is_separate_cfg_step") else False),
                )
                decisions = getattr(self, "_xdit_pipefusion_leader_decisions", None)
                if decisions is None:
                    decisions = {}
                    self._xdit_pipefusion_leader_decisions = decisions
                if state.pipeline_patch_idx != 0 and leader_key in decisions:
                    result = decisions[leader_key]
                    shared = True

        if result is None:
            result = original_can_cache(*args, **kwargs)

        if pipefusion and not sequence_parallel:
            if leader_key is not None and not shared:
                self._xdit_pipefusion_leader_decisions[leader_key] = result
            return result

        if pipefusion:
            from xfuser.core.distributed import get_sp_group

            sync_group = get_sp_group()
        else:
            from xfuser.core.distributed import get_world_group

            sync_group = get_world_group()

        decision = torch.tensor(
            [1 if result else 0],
            device=torch.cuda.current_device(),
            dtype=torch.int32,
        )
        sync_group.broadcast(decision, src=0)
        agreed = bool(decision.item())
        if agreed and not getattr(self, "_xdit_cache_warmed", False):
            dist.barrier(group=sync_group.device_group)
            self._xdit_cache_warmed = True
        if leader_key is not None and not shared:
            self._xdit_pipefusion_leader_decisions[leader_key] = agreed
        return agreed

    manager.can_cache = types.MethodType(can_cache, manager)
    manager._xdit_share_pipefusion_decisions = bool(getattr(transformer, "_xdit_share_pipefusion_decisions", False))
    manager._xdit_cache_decision_sync_installed = True


def install_pipefusion_patch_contexts(transformer: torch.nn.Module) -> None:
    """Give every PipeFusion patch an independent Cache-DiT history."""
    from xfuser.core.distributed import get_runtime_state

    manager = getattr(transformer, "_context_manager", None)
    base_names = getattr(transformer, "_context_names", None)
    if manager is None or not base_names:
        raise RuntimeError(
            "Cache-DiT did not expose _context_manager/_context_names after "
            "enable_cache; cannot install PipeFusion patch contexts."
        )
    if getattr(manager, "_xdit_pipefusion_patch_contexts_installed", False):
        return

    patch_context_names = {name for name in base_names if isinstance(name, str)}
    if not patch_context_names:
        raise RuntimeError("Cache-DiT exposed no named PipeFusion contexts.")

    original_reset_context = manager.reset_context

    @torch.compiler.disable
    def reset_patch_contexts(
        self,
        cached_context,
        *args,
        _names=patch_context_names,
        _reset=original_reset_context,
        **kwargs,
    ):
        patch_kwargs = None
        state = None
        if isinstance(cached_context, str) and cached_context in _names:
            state = get_runtime_state()
            patch_kwargs = dict(kwargs)
            if "cache_config" in kwargs:
                patch_kwargs["cache_config"] = _shift_patch_cache_config(
                    kwargs["cache_config"],
                    state.runtime_config.warmup_steps,
                )
        context = _reset(cached_context, *args, **kwargs)
        if patch_kwargs is not None:
            decisions = getattr(self, "_xdit_pipefusion_leader_decisions", None)
            if decisions is not None:
                decisions.clear()
            configured_patch_count = getattr(
                getattr(
                    getattr(state, "parallel_config", None),
                    "pp_config",
                    None,
                ),
                "num_pipeline_patch",
                state.num_pipeline_patch,
            )
            for patch_index in range(max(state.num_pipeline_patch, configured_patch_count)):
                _reset(
                    f"{cached_context}:pipefusion_patch_{patch_index}",
                    *args,
                    **patch_kwargs,
                )
        return context

    manager.reset_context = types.MethodType(reset_patch_contexts, manager)
    original_set_context = manager.set_context

    @torch.compiler.disable
    def set_patch_context(
        self,
        cached_context,
        *args,
        _names=patch_context_names,
        _set=original_set_context,
        **kwargs,
    ):
        state = get_runtime_state()
        if state.patch_mode and isinstance(cached_context, str) and cached_context in _names:
            if state.pipeline_patch_idx >= state.num_pipeline_patch:
                raise RuntimeError("PipeFusion patch index exceeds the active patch layout.")
            cached_context = f"{cached_context}:pipefusion_patch_{state.pipeline_patch_idx}"
        return _set(cached_context, *args, **kwargs)

    manager.set_context = types.MethodType(set_patch_context, manager)
    manager._xdit_pipefusion_patch_contexts_installed = True

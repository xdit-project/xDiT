from __future__ import annotations

import torch
import torch.distributed as dist
import os
import gc
import logging
from typing import TYPE_CHECKING, Tuple


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
    )


# Allow single-GPU runs to work without torchrun, debugging
if not os.environ.get("RANK"):
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"


# ruff: noqa: E402
# Load only the argument parser at module level. Then main() parses the
# arguments, for example --help, before it loads the models and the runtime.
from xfuser.config.args import FlexibleArgumentParser, xFuserArgs

if TYPE_CHECKING:
    from xfuser.model_executor.models.runner_models.base_model import DiffusionOutput, xFuserModel


def _model_registry() -> dict:
    """Load the runner models and return their registry.

    The import loads the runtime: xfuser.envs (aiter, diffusers) and the
    import-time patches in xfuser.core.utils.runner_utils.
    """
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    return MODEL_REGISTRY


def _log(message: str) -> None:
    # runner_utils loads the runtime. Import it when the runner logs, not
    # when this module loads.
    from xfuser.core.utils.runner_utils import log

    log(message)


def _has_output_payload(output: DiffusionOutput | None) -> bool:
    return bool(output and (output.images or output.videos))


def _select_output_owner(output: DiffusionOutput | None) -> int:
    """Select one rank that actually owns output, falling back to the last rank."""
    if not dist.is_available() or not dist.is_initialized():
        return 0

    rank = dist.get_rank()
    if dist.get_world_size() == 1:
        return rank

    device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else torch.device("cpu")
    candidate = torch.tensor(
        [rank if _has_output_payload(output) else -1],
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(candidate, op=dist.ReduceOp.MAX)
    owner = int(candidate.item())
    return owner if owner >= 0 else dist.get_world_size() - 1


class xFuserModelRunner:
    """Run a registered diffusion model from a configuration dictionary.

    Construction validates configuration and initializes distributed state as
    needed, but model weights are loaded by ``initialize``. Multi-rank callers
    must launch each process with a matching parallel layout, usually torchrun.
    """

    def __init__(self, config: dict) -> None:
        # Keep the startup order from before the lazy imports: load the runtime
        # (the xfuser.envs checks and the import-time patches, for example the
        # torchao Float8 FSDP2 patches in xfuser.core.utils.runner_utils) before
        # create_config() initializes the distributed environment.
        _model_registry()

        xfuser_config = xFuserArgs.from_runner_args(config)
        # Runs the config through argument parsing and validation - not the cleanest solution but has to be done inside the runner
        engine_config, input_config = xfuser_config.create_config()

        self.config = xfuser_config
        self.model = self._select_model(xfuser_config.model, xfuser_config)
        self.is_initialized = False

    def _select_model(self, model_name: str, config: xFuserArgs) -> xFuserModel:
        """Select and instantiate model from registry"""
        model = _model_registry().get(model_name, None)
        if not model:
            raise ValueError(f"Model {model_name} not found in registry.")
        return model(config)

    def initialize(self, input_args: dict) -> None:
        """Load the model and runtime using preprocessed input arguments.

        Repeated calls are ignored once initialization has completed.
        """
        if self.is_initialized:
            _log("Model already initialized, skipping...")
            return

        _log(f"Initializing model: {self.config.model}")
        self.model.initialize(input_args)

        self.is_initialized = True
        _log("Model initialization complete.")

    def run(self, input_args: dict) -> Tuple[DiffusionOutput, list]:
        """Generate outputs after initialization and return outputs and timings.

        Timings are measured in seconds. Benchmark repetitions return the final
        repetition's output; explicit batching records a timing for each batch.
        """
        if not self.is_initialized:
            raise RuntimeError("ModelRunner not initialized. Call initialize() before run().")

        _log("Running model...")
        output, timings = self.model.run(input_args)
        return output, timings

    def preprocess_args(self, input_args: dict) -> dict:
        """Resolve model defaults, validate inputs, and load referenced images.

        Returns a new dictionary for ``initialize``, ``run``, or ``profile``.
        Dataset prompts replace supplied prompts when a dataset is configured.
        """
        return self.model.preprocess_args(input_args)

    def profile(self, input_args: dict) -> Tuple[DiffusionOutput, list, torch.profiler.profiler.profile]:
        """Profile the model execution"""
        output, timings, profile_object = self.model.profile(input_args)
        return output, timings, profile_object

    def print_args(self, args: dict) -> None:
        """Print the arguments from the Namespace"""
        _log("Model Runner Arguments:")
        for arg, value in args.items():
            _log(f"  {arg}: {value}")

    def cleanup(self) -> None:
        """Release the initialized pipeline, device caches, and distributed state."""
        from xfuser.core.distributed import get_runtime_state

        torch.compiler.reset()
        torch.cuda.synchronize()
        del self.model.pipe
        gc.collect()
        torch.cuda.empty_cache()
        get_runtime_state().destroy_distributed_env()
        _log("Cleaned up resources.")

    def save(
        self,
        output: DiffusionOutput = None,
        timings: list = None,
        profile: torch.profiler.profiler.profile = None,
        save_once: bool = True,
    ) -> None:
        """Save available outputs, timings, and per-rank profiling traces.

        By default only the selected output-owning rank saves outputs/timings.
        Files use the configured output directory; profile traces are compressed
        Chrome trace JSON files. Omitted payloads are not saved.
        """
        from xfuser.core.utils.runner_utils import is_last_process

        if profile:
            self.model.save_profile(profile)

        if save_once:  # By default, we save output/timing only from one process
            if dist.is_available() and dist.is_initialized():
                if dist.get_rank() != _select_output_owner(output):
                    return
            elif not is_last_process():
                return
        if _has_output_payload(output):
            self.model.save_output(output)  # Handle different output types
        if timings:
            self.model.save_timings(timings)


def main():
    setup_logging()
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    xfuser_args = xFuserArgs.add_runner_args(parser).parse_args()
    args = vars(xfuser_args)
    runner = xFuserModelRunner(args)
    runner.print_args(args)

    input_args = runner.preprocess_args(args)
    runner.initialize(input_args)

    if xfuser_args.profile:
        out, timing, profile = runner.profile(input_args)
        runner.save(profile=profile)
    else:
        output, timings = runner.run(input_args)
        runner.save(output=output, timings=timings)

    runner.cleanup()


if __name__ == "__main__":
    with torch.inference_mode():  # mostly redundant, but harmless
        main()

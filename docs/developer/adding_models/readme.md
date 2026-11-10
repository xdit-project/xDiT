# Add a model

Start with a runner adapter. Implement the parallel methods that the model needs.
Use FLUX.1-dev as an image-model reference and Z-Image as a CFG reference.

## Runner adapter

1. Subclass `xFuserModel`.
2. Implement `_load_model()` and `_run_pipe(input_args)`. Return a `DiffusionOutput` from `_run_pipe`.
3. Register supported checkpoint names with `@register_model(...)`. The runner-model package imports its modules to populate `MODEL_REGISTRY`.
4. Declare `ModelCapabilities`, `DefaultInputValues`, and `ModelSettings`. Capabilities describe validation rules; execution code must implement each feature.
5. Set dependency requirements and `LoadSupport` for the loading routes you implement.
6. Add model-specific validation for restrictions that shared checks cannot express.
7. Test defaults and each supported combination before enabling it.

Read the [adapter contract](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/runner_models/base_model.py), [FLUX adapter](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/runner_models/flux.py), and [adapter imports](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/runner_models/__init__.py). FLUX's `_load_model()` selects different transformer and pipeline implementations for PipeFusion.

## Parallel implementation

| Work | Guide | Current example |
| --- | --- | --- |
| Sequence and attention partitioning | [USP](adding_model_usp.md) | FLUX.1-dev |
| Replicated condition tokens | [Replicated text](adding_model_usp_text_replica.md) | FLUX.1-dev |
| Conditional/unconditional branch partitioning | [CFG](adding_model_cfg.md) | Z-Image |
| Combined branch and sequence groups | [CFG + USP](adding_model_cfg_usp.md) | Z-Image |
| Transformer stages and patch scheduling | [PipeFusion](adding_model_pipefusion.md) | FLUX.1-dev |

These guides explain implementation requirements and link to complete source.
See [framework design](../The_implement_design_of_xdit_framework.md) for runner adapters and component wrappers.
See the [runner guide](../../runner/runner.md) for user commands.

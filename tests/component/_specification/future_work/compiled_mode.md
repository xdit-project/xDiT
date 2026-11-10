# Future work: compiled mode

This document is for coding agents and for humans. It describes the limitation of version 1, the refactors of xDiT
that compiled mode needs, and the problems that the harness must solve.

## 1. Limitation

1. Version 1 runs only in eager mode. It forces `use_torch_compile` off, also when the YAML entry turns it on.
2. Production often runs compiled code. Inductor fuses operations and can choose other kernels. Thus the numbers of
   compiled code differ from those of eager code. For low-precision arithmetic, compiled code can even be more
   precise.
3. Eager results are valid only for eager code, and compiled results only for compiled code. Thus the harness must
   test a configuration in eager mode and in compiled mode separately.

## 2. Goal

1. In compiled mode, the question is how the component itself behaves when it is compiled.
2. The harness compiles only the tested component. The parent components that hold it stay eager.

## 3. Refactors that xDiT needs

### 3.1 Load without compile

1. **Today.** `initialize()` loads the model and compiles it in one step. `_compile_model`
   ([base_model.py:999](../../../../xfuser/model_executor/models/runner_models/base_model.py#L999-L999)) applies the
   wrapping and runs the warmup. A compiled init already includes a full generation for the warmup.
2. **Change.** Split the init into two steps:
   1. load the model without compile;
   2. apply the compile plan (section 3.2) as a separate step.
3. **Why the harness needs it.**
   1. The capture must run in eager mode, with the hooks outside compiled code (section 4.1).
   2. The harness decides what it compiles, with the settings of production.

### 3.2 The compile plan as data

1. **Today.** The compile settings are spread across code with side effects:
   1. `_compile_model` chooses the wrapping, handles the PipeFusion capture and replay, and runs the warmup;
   2. `_enable_compute_comm_overlap`
      ([base_model.py:953](../../../../xfuser/model_executor/models/runner_models/base_model.py#L953-L953)) changes
      global compiler settings;
   3. `_get_compile_mode` and `_get_compile_dynamic`
      ([base_model.py:913-918](../../../../xfuser/model_executor/models/runner_models/base_model.py#L913-L918))
      return the `torch.compile` arguments;
   4. model-specific overrides apply side effects. Example: Qwen swaps attention processors during the compile
      step.
2. **Change.** A declarative compile plan, which a generic step applies. Each entry holds:
   1. the target: a module path or the path of a block list;
   2. the kind of wrapping: replace the attribute with an `OptimizedModule`, compile each entry of a block list,
      or compile `forward` in place;
   3. the `torch.compile` arguments: `mode`, `dynamic`, `options`.
3. **Two additions to the plan:**
   1. **Global compiler settings.** `_enable_compute_comm_overlap` sets
      `torch._inductor.config.reorder_for_compute_comm_overlap` and its list of passes. Under PipeFusion it also
      raises `torch._dynamo.config.recompile_limit` and turns off `spmd_check`. These settings change the generated
      code. Thus the plan must hold them as compiler-configuration overrides. Otherwise a compile of the same unit
      by the harness runs with a different compiler configuration.
   2. **Side requirements.** Example: the CUDA-graph step marker that blockwise compile installs in
      `reduce-overhead` mode.
4. **The warmup moves out of the compile step.** This includes the PipeFusion capture-and-replay pass, which is a
   warmup strategy, not a compile.
5. **Model-specific overrides return plan entries** instead of side effects. The processor swap of Qwen moves to the
   build step.
6. **Why the harness needs it.** The harness can then compile a unit exactly as production does, and it can also
   compile units that production does not compile.

## 4. Problems that the harness must solve

### 4.1 Hooks and compiled code

1. The capture hooks must stay outside compiled code. A hook inside a compiled region breaks the graph or changes
   the compiled code.
2. Compile a separate wrapper for the replay: `compiled = torch.compile(component, ...)`, then
   `compiled(*args, **kwargs)`. The model itself stays eager. Thus no hook is ever inside compiled code.
3. Do not use `nn.Module.compile()` in place. It compiles `_call_impl`, which includes the hooks. An
   `OptimizedModule` runs the hooks of the module outside the compiled code.
4. Attention processors are compiled through their attention module, for example
   `pipe.transformer.blocks[0].attn1`, because a processor is not an `nn.Module`.

### 4.2 Compile settings of a sub-component

1. The settings come from the nearest plan entry that contains the sub-component: the entry whose target path is
   the longest prefix of the path of the sub-component.
2. Example: `transformer.blocks[3].attn1` falls under the entry `transformer` (whole compile) or
   `transformer.blocks` (blockwise compile).
3. The harness compiles the sub-component with the `torch.compile` arguments of that entry and under the compiler
   overrides of the plan.
4. If no entry contains the sub-component, production runs it eagerly. The harness can still compile it with
   settings from the test definition. The report then says "not a production compile path".

### 4.3 Graph boundaries

1. The compiler configuration then matches production, but the graph boundary does not.
2. In a whole-transformer compile, inductor fuses the operations of the attention module with their neighbors, and
   it treats the RoPE tables as intermediate values. Compiled alone, the RoPE tables are graph inputs.
3. Thus the kernels can differ from production. This is an accepted limit.

### 4.4 Global state as compile guards

1. Values that compiled code reads from global state become guards. Example: `attention_backend` comes from
   `get_runtime_state()`.
2. The harness must restore the runtime state before the first compiled call. A later change causes a recompile.

### 4.5 Shapes and recompiles

1. With `dynamic=None`, the second distinct shape makes dynamo recompile with dynamic dimensions. After that, all
   shapes run the dynamic kernels, also the first shape.
2. After `recompile_limit` recompiles of one function, dynamo runs the function eagerly without a message. The limit
   is 8 in the local torch version.
3. The harness compiles one static wrapper for each shape and calls `torch._dynamo.reset()` between shape groups.
4. Some models prepare compile for one shape. Example: LingBot-Video calls
   `prepare_for_compile(height, width, num_frames)`.

### 4.6 Checks in compiled mode

1. The fidelity check compares a compiled replay with an eager capture. They are not bitwise equal, because the
   kernels differ. The fidelity check of compiled mode therefore compares the eager replay with the eager capture,
   as in version 1.
2. The determinism check compares two compiled replays bitwise.
3. A compiled replay is compared with B, as an eager replay is.

### 4.7 Not verified

1. A compile of a single attention module that contains the Ulysses all-to-all. xDiT already compiles whole blocks
   that contain it (blockwise compile), so it probably works.

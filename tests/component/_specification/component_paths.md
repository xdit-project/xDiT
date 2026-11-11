# Component paths

A component path names one component of the model. It also selects calls of that component. Each listed path is
one case.

## 1. Syntax

### 1.1 Rules

1. A path is a sequence of elements, separated by dots. Example: `pipe.transformer.blocks[0].attn1`.
2. The path has no leading dot.
3. The first element names an attribute of the runner model object. See section 2.1.
4. `[i]` after an element selects item `i` of a container. `i` is an integer with Python index semantics, so
   negative values are permitted. Example: `blocks[-1]`.
5. For an `nn.ModuleDict`, `[key]` selects the item `key`.
6. `:i` after a called element selects call `i` of that element. `i` must be a non-negative integer. Example:
   `pipe.transformer:3.blocks[0]:0`.
7. An element without `:i` selects all of its calls.

Call numbers must not be negative. The capture then saves each call at once, because a call never needs a later
call to decide whether it is selected.

### 1.2 Grammar

```text
path      = element { "." element }
element   = name { index } [ call ]
name      = identifier
index     = "[" ( integer | identifier ) "]"
call      = ":" digits
integer   = [ "-" ] digits
```

### 1.3 When the harness detects errors

1. Syntax errors: during pytest collection. The test module then fails to collect.
2. Errors that need the model: after the init of A. See section 3.

## 2. Path elements

### 2.1 The root

1. The root of every path is the runner model object (`runner.model`).
2. The first element names an attribute of the runner model object, for example `pipe`, `second_pipe` or
   `refiner_pipe`.
   1. The harness does not hook the first element. It counts the runner runs as the calls of the first element
      (section 4.3).
   2. The harness does not check that the attribute holds a pipeline.
   3. Calls from other code can count under the first element (section 4.4).
3. `pipe` can be an xDiT pipeline wrapper. The wrapper forwards unknown attributes to the wrapped diffusers
   pipeline ([base_wrapper.py:24-30](../../../xfuser/model_executor/base_wrapper.py#L24-L30)), so
   `pipe.transformer` still resolves.

### 2.2 Element kinds

1. The first element. The harness does not hook it and counts its calls itself (section 4.3).
2. An `nn.Module`.
3. A container, followed by `[i]`.
4. A method of a module, as the last element only. Example: `pipe.vae.decode`.

### 2.3 Rules for kinds

1. `:i` on an `nn.ModuleList` or an `nn.ModuleDict` is an error. These containers are never called.
2. A diffusers attention processor cannot be listed. It is not an `nn.Module`, and Python looks up `__call__` on
   its type, so the harness cannot hook one instance. List the attention module that calls the processor instead,
   for example `pipe.transformer.blocks[0].attn1`. A path to a processor is an error with this advice.
3. For an `nn.Sequential`, `[i]` selects item `i`.
4. A method element must be the last element.

## 3. Resolution

1. The workers of A resolve the paths after the init of the model. The workers of B resolve them again after
   their init.
2. Resolution uses attribute access for names and Python indexing for `[i]`.
3. These are path errors:
   1. a missing attribute;
   2. an index out of range;
   3. `:i` on a container;
   4. a processor;
   5. a method element that is not the last element;
   6. one object listed two times;
   7. a first element whose attribute holds `None`, for example `refiner_pipe` of the LingBot-Video runner without
      `--task refine`.
4. "One object listed two times" compares the resolved objects. For a method, it compares the owner object and
   the method name. Examples:
   1. `pipe.transformer.blocks[0]` and `pipe.transformer.blocks[-40]` are the same object when the list has 40
      blocks.
   2. In the LTX-2.3 runner, `pipe.transformer` and `second_pipe.transformer` are the same object.

   Two entries with different call numbers on the same object are also an error.
5. A path error gives FAILED_TO_START to the affected cases, with the path error as the reason. The instance
   continues with the other cases.
6. A path must resolve in A and in B. If it fails in B, the case gets FAILED_TO_START with the path error of B.

## 4. Hooks

### 4.1 Which objects get hooks

1. Every called element on every listed path gets hooks, not only the tested component. The hooks of the ancestors
   give the call counters of the ancestors (section 5).
2. An object that is on several paths gets one set of hooks.

### 4.2 Kinds of hooks

1. **Modules** get a forward pre-hook and a forward hook. Both use `with_kwargs=True`, so they see the keyword
   arguments.
   1. The pre-hook starts a call. It saves the inputs before the module runs, because a module can change its
      inputs in place.
   2. The forward hook ends the call and saves the output.
2. **Methods** get a wrapper that the harness sets as an instance attribute with the method name. Example: the
   harness sets `pipe.vae.decode` on the `vae` object. The call `self.vae.decode(...)` in
   [pipeline_wan_i2v.py:272](../../../xfuser/model_executor/pipelines/pipeline_wan_i2v.py#L272-L272) then finds the
   wrapper first. Wrappers that xDiT puts around the method stay inside the captured call.
3. During replays, the harness disables all capture hooks.

### 4.3 The first element

1. The harness does not hook the first element. A pipeline is not an `nn.Module`, and `pipe(...)` looks up
   `__call__` on the type, so the harness cannot hook one pipeline instance. The rule applies to every first
   element, so all first elements work the same.
2. The harness starts each runner run itself and counts these runs as the calls of the first element. The forced
   overrides remove the extra runs of the run loop ([test_definition.md](test_definition.md), section 3.2). The
   code of the runner can still call a pipeline more than once in one run (section 4.4).
3. In version 1, each instance runs the runner once. The address of every call thus starts with the first element
   and call number 0, for example `pipe:0`.
4. The first element is in a call during the whole runner run. Thus a direct child of the first element never
   breaks the rules of section 5.4.

### 4.4 Calls from outside the first element

1. **Limitation.** The counter of a called element counts every call of its object during the runner run, whatever
   code makes the call. The harness does not detect, warn about or report calls from outside the first element.
   Thus an address does not tell which pipeline, or which call of a pipeline, made a call. There are three causes:
   1. Several pipelines of the runner model object hold the same object. Hooks are on objects, so the calls of all
      these pipelines count under the first element of the path. Examples:
      1. The LTX-2.3 runner builds `second_pipe` with the same `transformer` object
         ([ltx.py:131-135](../../../xfuser/model_executor/models/runner_models/ltx.py#L131-L135)), and
         `upsample_pipe` with the `vae` of `pipe`
         ([ltx.py:150](../../../xfuser/model_executor/models/runner_models/ltx.py#L150-L150)).
      2. With `--task refine`, the LingBot-Video `refiner_pipe` gets the `vae` and the `text_encoder` of `pipe`
         ([lingbot_video.py:217-223](../../../xfuser/model_executor/models/runner_models/lingbot_video.py#L217-L223)).
   2. The runner calls a pipeline several times in one run. Examples:
      1. The distilled LTX-2.5 runner calls `self.pipe(...)` for stage 1
         ([ltx.py:633](../../../xfuser/model_executor/models/runner_models/ltx.py#L633-L633)) and for stage 2
         ([ltx.py:660](../../../xfuser/model_executor/models/runner_models/ltx.py#L660-L660)). The calls of stage 2
         continue the counters of stage 1 under `pipe:0`.
      2. Batched mode (`batch_size` is set) runs the pipeline code of the runner once for each batch
         ([base_model.py:1310](../../../xfuser/model_executor/models/runner_models/base_model.py#L1310-L1310)).
   3. The runner calls a component directly, outside all pipeline calls. Examples:
      1. The LTX-2.5 runners call `self.pipe.audio_vae.decode(...)`
         ([ltx.py:700](../../../xfuser/model_executor/models/runner_models/ltx.py#L700-L700)).
      2. The LingBot-Video runner calls `vae.encode(...)` on the VAE that it shares with `pipe`
         ([lingbot_video.py:410](../../../xfuser/model_executor/models/runner_models/lingbot_video.py#L410-L410)).
2. **The results stay valid.** Each of these calls is a real call with its real inputs. B runs the same runner with
   the same run arguments, so it makes the same calls with the same addresses. The test compares A and B for these
   calls as for all other calls. Only the address can mislead.
3. **Responsibility of the user.** The user must know which code calls a listed object, and must select the paths
   and the call numbers to match. Example: in the LTX-2.3 runner, `pipe.transformer` and `second_pipe.transformer`
   are the same object, so only one of them can be listed (section 3). Its case selects the calls of `pipe` and the
   calls of `second_pipe`.
4. **Calls outside the path parent.** If other code calls a listed object outside the call of its path parent, the
   rules of section 5.4 still apply, and the instance ends with ERROR_A or ERROR_B.
5. **Future work.** [future_work/index.md](future_work/index.md), section 2.15.

## 5. Call addresses

### 5.1 Counters

1. Every called element on a path has a counter.
2. The counter restarts at each call of the nearest called ancestor on the path.
3. The address of a call is the tuple of the counters of all called elements on the path.
4. Elements that are not called have no counter. Examples: `blocks` (a `ModuleList`) and `vae` (the pipeline calls
   the method `vae.decode`, not the module).
5. Calls of objects that are not on the path do not count. Example: in `pipe.vae.decoder.up_blocks[0]`, the call
   of `vae.decode` is not on the path.

### 5.2 Written form

The address uses the path syntax with a call number on each called element. Example:
`pipe:0.transformer:3.blocks[0]:0` is the first call of block 0 during the fourth call of the transformer.

### 5.3 Selection

1. `:i` selects call `i` at that level, inside the selected calls of the ancestors.
2. No `:i` selects all calls at that level.
3. A case with several selected calls checks every call. It fails if any call fails. The failure message names the
   failing calls.
4. A selected call that does not occur fails its case.

### 5.4 Run-time rules for ancestors

1. An ancestor adds its counter to the address only while it is in a call.
2. If the in-call state of an ancestor changes between calls of the tested component, the harness raises an
   exception. Example: the tested component runs once inside a call of its path parent and once outside.
3. If a component runs outside the call of its path parent, the harness raises an exception. The user must give a
   correct path.
4. These exceptions end the instance with ERROR_A or ERROR_B ([processes.md](processes.md), section 6).

### 5.5 Equal addresses in A and B

A and B use the same run arguments. Thus they produce the same addresses. The test function checks this (see
[checks.md](checks.md), section 2).

## 6. Case IDs

1. The case ID is the configuration name, a colon, and the path string. Example:
   `test_component[wan2_2.quantgemm_quantattn.gfx950:pipe.transformer:3.blocks[0]:0]`.
2. A configuration name contains no `:` ([test_definition.md](test_definition.md), section 7). Thus the first `:`
   of a case ID separates the name from the path.
3. pytest `-k` accepts these characters. Its identifier grammar is `(\w|:|\+|-|\.|\[|\]|\\|/)+`
   ([expression.py:10](../../../../../venv/lib/python3.12/site-packages/_pytest/mark/expression.py#L10-L10),
   pytest 9.0.3).
4. `-k` matches substrings. For example, `-k transformer:3` also selects `transformer:30`. Use node IDs for an exact
   selection.
5. The harness can change the case IDs before pytest sees them, if a future path needs characters that `-k` does
   not accept.

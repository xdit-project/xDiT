# Component paths

A component path names one component of the model. It also selects calls of that component. Each listed path is
one case.

## 1. Syntax

### 1.1 Rules

1. A path is a sequence of elements, separated by dots. Example: `pipe.transformer.blocks[0].attn1`.
2. The path has no leading dot.
3. The first element must be `pipe` in version 1. See section 2.1.
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
2. The first element must be `pipe`.
3. A runner model object that holds another pipeline is UNSUPPORTED in version 1. Examples:
   1. LTX runners hold `second_pipe` and `upsample_pipe`
      ([ltx.py:155-156](../../../xfuser/model_executor/models/runner_models/ltx.py#L155-L156)), and one variant
      also holds `decode_pipe`
      ([ltx.py:565](../../../xfuser/model_executor/models/runner_models/ltx.py#L565-L565)).
   2. The LingBot-Video runner holds `refiner_pipe`
      ([lingbot_video.py:273](../../../xfuser/model_executor/models/runner_models/lingbot_video.py#L273-L273)).
4. Reason for item 3: such pipelines share components with `pipe`. For example, the LTX `second_pipe` uses the same
   `transformer` object. Hooks are on objects, so a call from the other pipeline would count as a call under
   `pipe`. The harness cannot tell the pipelines apart (section 4.3).
5. `pipe` can be an xDiT pipeline wrapper. The wrapper forwards unknown attributes to the wrapped diffusers
   pipeline ([base_wrapper.py:24-30](../../../xfuser/model_executor/base_wrapper.py#L24-L30)), so
   `pipe.transformer` still resolves.

### 2.2 Element kinds

1. `pipe`. The harness counts its calls (section 4.3).
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
   6. one object listed two times.
4. "One object listed two times" compares the resolved objects. For a method, it compares the owner object and
   the method name. Example: `pipe.transformer.blocks[0]` and `pipe.transformer.blocks[-40]` are the same object
   when the list has 40 blocks. Two entries with different call numbers on the same object are also an error.
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

### 4.3 The `pipe` element

1. `pipe` cannot be hooked. It is not an `nn.Module`, and `pipe(...)` looks up `__call__` on the type.
2. The harness starts each pipeline run itself. The forced overrides make each runner run call the pipeline exactly
   once ([test_definition.md](test_definition.md), section 3.2). Thus the harness counts its own pipeline runs as
   the calls of `pipe`.
3. In version 1, each instance runs the pipeline once. The address of every call thus starts with `pipe:0`.

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

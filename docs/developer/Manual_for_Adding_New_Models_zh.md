# 新模型添加手册

<aside>
💡 以pixart-alpha为例，并行版本模型支持的全过程如下
</aside>

<aside>
⚠️ 注意，在`__call__`以及`forward`中直接使用父类提供的函数进行修改时请小心，若父类提供的函数无法满足需求，则需要自行重载该函数从而使之与模型匹配
</aside>

# 0.命名规范

- xDiT的所有model，scheduler，layer的**文件名**应与来源库diffuser中的名称保持一致
- xDiT的Pipeline类wrapper，为了和diffusers pipeline完全适配，其名称应为`xFuser+原名` ，而不带有`Wrapper`后缀，如pixart-alpha在diffusers中的pipeline名为`PixArtAlphaPipeline`，则在xDiT中名称应为`xFuserPixArtAlphaPipeline`
- 除Pipeline类以外的其他类的wrapper名称应为`xFuser+原名+Wrapper` ，如pixart-alpha的backbone为`PixArtTransformer2D`，此backbone在xDiT中的wrapper名称应为`xFuserPixArtTransformer2DWrapper`

下图展示了xDiT项目中不同Class的调用关系。如果新加一个PixArt模型，增加的类用红框圈出。

![class_structure.png](../../assets/developer/class_structure.png)

# 1.pipeline class

pipelines文件目录位于`xfuser/model_executor/pipelines`，在该目录下新建一个与diffusers库中对应pipeline同名的文件(此处为pipeline_pixart_alpha.py)，在里面编写一个pipeline wrapper类`xFuserPixArtAlphaPipeline`。这个类需要使用装饰器`xFuserPipelineWrapperRegister.register`，该装饰器会将此pipeline wrapper与原始pipeline的对应关系进行注册，方便之后的自动并行化。

代码如下：

```python
@xFuserPipelineWrapperRegister.register(PixArtAlphaPipeline)
class xFuserPixArtAlphaPipeline(xFuserPipelineBaseWrapper):

   @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        engine_config: EngineConfig,
        **kwargs,
    ):
        pipeline = PixArtAlphaPipeline.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        return cls(pipeline, engine_config)

    @torch.no_grad()
    @xFuserPipelineBaseWrapper.enable_data_parallel
    @xFuserPipelineBaseWrapper.check_to_use_naive_forward
    def __call__(...
```

pipeline wrapper中，仅需要实现两个函数，`from_pretrained`用以将参数转发到原pipeline类(此处为diffuser中的`PixArtAlphaPipeline`)的`from_pretrained`，获取到一个原pipeline对象，并将其传进`cls.__init__`。随后逐层init的过程中会逐渐将其组分进行并行化，此种方式完全兼容diffusers接口。

将原始xFuserPixArtAlphaPipeline中的所有`@property`复制粘贴到`xFuserPixArtAlphaPipeline`中。

最大的改动是`__call__` 方法，首先使用`xFuserPipelineBaseWrapper`中的两个装饰器包装它，这是是必须的且顺序不可变，其作用如下：

- `enable_data_parallel`：启用数据并行(dp)，会在__call__之前自动读取dp相关配置与输入prompts，当promp多个时候，会分配到不同的dp rank上执行。如果输入prompt只有一个，则不发挥作用。
- `check_to_use_naive_forward`：进行并行条件检测。若仅enable了data parallel，则直接使用该装饰器对输入prompts进行朴素forward推理

<aside>
💡 装饰器的顺序不可交换，否则在使用朴素forward时data parallel将无法使用。

</aside>

然后对`__call__`中代码做修改。

## 1.1.__call__改动

`__call__`中代码逻辑是在diffusers库对应pipeline的`__call__`函数中沿袭而来的，需要现将pipeline的`__call__`函数复制到对应wrapper的`__call__`中，再进行进一步修改

<aside>
📝 例如对于pixart-alpha来说，需要先将`diffusers/pipelines/pixart_alpha/pipeline_pixart_alpha.py`中`PixArtAlphaPipeline`类的`__call__`复制为`xfuser/model_executor/pipelines/pipeline_pixart_alpha.py`中`xFuserPixArtAlphaPipeline`类的`__call__`

</aside>

1. encode input prompt之前，计算出batch size之后。使用本次推理的长宽和batch size调用`set_input_parameters`来对本次forward的输入信息进行设置，从而计算出各种运行时原数据，为正式forward做准备

    ```python
            ...
            # 2. Default height and width to transformer
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]
            device = self._execution_device

            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0
    **# -------------------------- ADDED BELOW ------------------------------**
            #* set runtime state input parameters
            get_runtime_state().set_input_parameters(
                height=height,
                width=width,
                batch_size=batch_size,
                num_inference_steps=num_inference_steps,
            )
    **# -------------------------- ADDED ABOVE ------------------------------**
            # 3. Encode input prompt
            (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self.encode_prompt(
                prompt,
                do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                clean_caption=clean_caption,
                max_sequence_length=max_sequence_length,
            )
            ...
    ```

2. 修改`do_classifier_free_guidance`的情况下的`prompt_embeds`&`prompt_attention_mask`划分，判定split batch的情况

    ```python
            ...
            # 3. Encode input prompt
            (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self.encode_prompt(
                prompt,
                do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                clean_caption=clean_caption,
                max_sequence_length=max_sequence_length,
            )

    **#! ---------------------------------------- MODIFIED BELOW ----------------------------------------**
            # * dealing with cfg degree
            if do_classifier_free_guidance:
                (
                    prompt_embeds,
                    prompt_attention_mask,
                ) = self._process_cfg_split_batch(
                    prompt_embeds,
                    prompt_attention_mask,
                    negative_prompt_embeds,
                    negative_prompt_attention_mask
                )

            #! ORIGIN
            # if do_classifier_free_guidance:
            #     prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            #     prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
    **#! ---------------------------------------- MODIFIED ABOVE ----------------------------------------**
    				...
    ```

3. 仍然是对classifier_free_guidance和split batch的特殊处理

    ```python
            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 6.1 Prepare micro-conditions.
            added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
            if self.transformer.config.sample_size == 128:
                resolution = torch.tensor([height, width]).repeat(
                    batch_size * num_images_per_prompt, 1
                )
                aspect_ratio = torch.tensor([float(height / width)]).repeat(
                    batch_size * num_images_per_prompt, 1
                )
                resolution = resolution.to(dtype=prompt_embeds.dtype, device=device)
                aspect_ratio = aspect_ratio.to(dtype=prompt_embeds.dtype, device=device)

    **#! ---------------------------------------- MODIFIED BELOW ----------------------------------------**
                if (
                    do_classifier_free_guidance
                    and get_classifier_free_guidance_world_size() == 1
                ):
                    resolution = torch.cat([resolution, resolution], dim=0)
                    aspect_ratio = torch.cat([aspect_ratio, aspect_ratio], dim=0)

                #! ORIGIN
                # if do_classifier_free_guidance:
                #     resolution = torch.cat([resolution, resolution], dim=0)
                #     aspect_ratio = torch.cat([aspect_ratio, aspect_ratio], dim=0)
    **#! ---------------------------------------- MODIFIED ABOVE ----------------------------------------**
    ```

4. 模型forward过程需要在前几个diffusion step使用同步流水线做与人，后面都使用异步流水线。复杂的通信逻辑已封装进`xFuserPipelineBaseWrapper`，直接调用即可
    - 若在基类中实现的`_sync_pipeline`与`_async_pipeline`函数与模型不适配，则需要在当前类中重载该函数，并参考基类中的代码单独实现。通常这种情况会出现在存在多余的通信逻辑时

    ```python
            # 7. Denoising loop
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    **#! ---------------------------------------- MODIFIED BELOW ----------------------------------------**
            num_pipeline_warmup_steps = get_runtime_state().runtime_config.warmup_steps

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                if (
                    get_pipeline_parallel_world_size() > 1
                    and len(timesteps) > num_pipeline_warmup_steps
                ):
                    # * warmup stage
                    latents = self._sync_pipeline(
                        latents=latents,
                        prompt_embeds=prompt_embeds,
                        prompt_attention_mask=prompt_attention_mask,
                        guidance_scale=guidance_scale,
                        timesteps=timesteps[:num_pipeline_warmup_steps],
                        num_warmup_steps=num_warmup_steps,
                        extra_step_kwargs=extra_step_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        progress_bar=progress_bar,
                        callback=callback,
                        callback_steps=callback_steps,
                    )
                    # * pipefusion stage
                    latents = self._async_pipeline(
                        latents=latents,
                        prompt_embeds=prompt_embeds,
                        prompt_attention_mask=prompt_attention_mask,
                        guidance_scale=guidance_scale,
                        timesteps=timesteps[num_pipeline_warmup_steps:],
                        num_warmup_steps=num_warmup_steps,
                        extra_step_kwargs=extra_step_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        progress_bar=progress_bar,
                        callback=callback,
                        callback_steps=callback_steps,
                    )
                else:
                    latents = self._sync_pipeline(
                        latents=latents,
                        prompt_embeds=prompt_embeds,
                        prompt_attention_mask=prompt_attention_mask,
                        guidance_scale=guidance_scale,
                        timesteps=timesteps,
                        num_warmup_steps=num_warmup_steps,
                        extra_step_kwargs=extra_step_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        progress_bar=progress_bar,
                        callback=callback,
                        callback_steps=callback_steps,
                        sync_only=True,
                    )
    **#! ---------------------------------------- MODIFIED ABOVE ----------------------------------------**
    ```

5. 输出处理，由于只有流水线最后一段持有生成的结果，设置为仅有每个dp group的最后一个rank返回数据，其他rank返回None

    ```python
            # 8. Decode latents (only rank 0)
    **#! ---------------------------------------- ADD BELOW ----------------------------------------**
            if is_dp_last_rank():
    **#! ---------------------------------------- ADD ABOVE ----------------------------------------**
                if not output_type == "latent":
                    image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
                    if use_resolution_binning:
                        image = self.image_processor.resize_and_crop_tensor(image, orig_width, orig_height)
                else:
                    image = latents

                if not output_type == "latent":
                    image = self.image_processor.postprocess(image, output_type=output_type)

                # Offload all models
                self.maybe_free_model_hooks()

                if not return_dict:
                    return (image,)

                return ImagePipelineOutput(images=image)
    **#! ---------------------------------------- ADD BELOW ----------------------------------------
            else:**
                return None
    **#! ---------------------------------------- ADD ABOVE ----------------------------------------**
    ```


至此，pipeline中的改动已完成，在pipeline的__call__层次主要处理了cfg的split batch情况。pipeline parallel相关的改动与通信被封装到了_sync_pipeline与_async_pipeline中，从而简化模型修改。但在基类中此函数无法满足模型需求是同样需要重载并手动更改以保证正确性。

# 2.transformer backbone class

transformer wrapper文件目录位于`xfuser/model_executor/models/transformers`，在其中新建diffuser中transformer backbone同名的文件即可。此例中transformer为`PixArtTransformer2DModel`，在diffusers中位于`pixart_transformer_2d.py`文件中，故该wrapper文件名为`xfuser/model_executor/models/transformers/pixart_transformer_2d.py`

transformer backbone模型同样需要经过一定的修改，但需修改处很少，且仅涉及到对特定pp_rank做的事情进行特判，需要使用`@xFuserTransformerWrappersRegister.register`装饰器。和实现两个函数，`__init__`与`__forward__` 我们后面分别介绍。

```python
@xFuserTransformerWrappersRegister.register(PixArtTransformer2DModel)
class xFuserPixArtTransformer2DWrapper(xFuserTransformerBaseWrapper):
    def __init__(...

    @xFuserBaseWrapper.forward_check_condition
    def forward(...
```

## 2.1.`__init__` 修改

`__init__`中需要指定transformer model中需要wrap哪些层，以及wrap时有没有什么额外参数。

```python
    def __init__(
        self,
        transformer: PixArtTransformer2DModel,
    ):
        super().__init__(
            transformer=transformer,
            submodule_classes_to_wrap=[nn.Conv2d, PatchEmbed],
            submodule_name_to_wrap=["attn1"],
        )
```

- 传入需要wrap的layer class(`submodule_classes_to_wrap`)或者其submodule name(`submodule_name_to_wrap`)即可。通常来说不需要改动

## 2.2.`__forward__` 修改

`__forward__`仍然只需要对diffusers/transformers中原始模型的forward做如下少许修改，读者请自行对比注释掉和新增的部分。

1. 更改获取height / width的方式，因为patch情况下无法直接通过hidden_state获取到准确的height & width。
2. 设置仅pp_rank为0时候进行pos_embed

    ```python
            # 1. Input
            batch_size = hidden_states.shape[0]
    **#! ---------------------------------------- MODIFIED BELOW ----------------------------------------**
            #* get height & width from runtime state
            height, width = self._get_patch_height_width()
            #* only pp rank 0 needs pos_embed (patchify)
            if is_pipeline_first_stage():
                hidden_states = self.pos_embed(hidden_states)

            #! ORIGIN
            # height, width = (
            #     hidden_states.shape[-2] // self.config.patch_size,
            #     hidden_states.shape[-1] // self.config.patch_size,
            # )
            # hidden_states = self.pos_embed(hidden_states)
    **#! ---------------------------------------- MODIFIED ABOVE ----------------------------------------**
    ```

3. 每个diffusion step结束需要进行unpatchify，将attention中使用的tokens形式的hidden state转化回到latent space下的图片，我们只让最后一个pp_rank做这个操作。

    ```python
            # 3. Output
            #* only the last pp rank needs unpatchify
    **#! ---------------------------------------- ADD BELOW ----------------------------------------**
            if is_pipeline_last_stage():
    **#! ---------------------------------------- ADD ABOVE ----------------------------------------**
                shift, scale = (
                    self.scale_shift_table[None] + embedded_timestep[:, None].to(self.scale_shift_table.device)
                ).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states)
                # Modulation
                hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
                hidden_states = self.proj_out(hidden_states)
                hidden_states = hidden_states.squeeze(1)

                # unpatchify
                hidden_states = hidden_states.reshape(
                    shape=(-1, height, width, self.config.patch_size, self.config.patch_size, self.out_channels)
                )
                hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
                output = hidden_states.reshape(
                    shape=(-1, self.out_channels, height * self.config.patch_size, width * self.config.patch_size)
                )
    **#! ---------------------------------------- ADD BELOW ----------------------------------------**
            else:
                output = hidden_states
    **#! ---------------------------------------- ADD ABOVE ----------------------------------------**
    ```


# 3.scheduler

scheduler为每一个diffusion step的结果进行采样，scheduler有很多种，比如DDIM，DPM等等。对于不同scheduler，我们都仅需对scheduler类的一个成员函数`step`进行修改。

在目前的框架中，已经实现了主流的scheduler的并行化改造，若使用未实现的scheduler，运行时会有

`ValueError: Scheduler is not supported by xFuser`报错，需要单独添加对scheduler的支持。

示例文件位于`xfuser/model_executor/schedulers/scheduling_dpmsolver_multistep.py`，文件中对应位置标有修改记号，可尝试直接将对应逻辑搬运到新增的scheduler即可。

该部分逻辑是对于patch情况下的model_output进行暂存，通过对一个完整tensor进行切片来更新对应的patch位置，从而做到与单设备运行时等价。

# 4.layers

DiT中需要并行改造的Layer（torch.nn.Module派生类）主要集中在Attention Layer，比如SelfAttention。如果使用U-Net卷积也需要并行改造，不过DiT中很少使用卷积。

Layer的改造需要处理Stale状态，比如Attention中的Stale KV，这部分逻辑复杂，和PipeFusion、Sequence Parallel的逻辑耦合。如果模型需要添加其他layer的情况，请参照`xfuser/model_executor/layers`目录中已有layers进行更改。

目前框架中已对`Conv2d`，`PatchEmbed` layer进行了支持，它们被用到transformer backbone中的`pos_embed`层与其内部需要用到的卷积操作。diffuser库中attention实现后端会有不同Attention的具体实现，称之为processor，工作量原因无法一次性完成对于所有processors的更改。目前已实现的processors有`AttnProcessor2_0`与`JointAttnProcessor2_0`的并行化版本

一个新的模型加入可能需要新的processor实现支持。若新加入模型的Attention processors不被支持，会出现运行时报错: `ValueError: Attention Processor class xxx is not supported by xFuser`。如果遇到此种情况，请尝试执行完成支持或在代码仓库中提issue，以便模型能尽快获得支持。

由于位于模型中不同位置不同layer的并行化方法不同，无法做到统一。若有任何问题，咨询xDiT maintainer。

<aside>
💡 上述所有修改标记均在xDiT项目pixart-alpha相关源码文件中存在，建议直接参照其中的修改标记进行新模型的适配

</aside>
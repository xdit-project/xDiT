import torch
from diffusers import FluxPipeline
from torch.profiler import profile, record_function, ProfilerActivity

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda:1")
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

def single_run(num_inference_steps=50):
    prompt = "A cat holding a sign that says hello world"
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=num_inference_steps,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    image.save("flux-dev.png")

# warmup
def warmup(times=3):
    for _ in range(times):
        single_run()

def run():
    single_run(num_inference_steps=30)
    num_inference_steps=10
    # Example PyTorch code to profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True,
                 with_stack=True,
                 with_flops=True,
                 with_modules=True,
                 record_shapes=True,
                 on_trace_ready=torch.profiler.tensorboard_trace_handler("./tensorboard/flux")
                 ) as prof:
        with record_function("flux_pipeline"):
            single_run(num_inference_steps=num_inference_steps)
    # prof.export_chrome_trace("test_trace_" + "flux" + f"_steps_{num_inference_steps}" + ".json")

def main():
    warmup()
    run()

if __name__ == "__main__":
    main()

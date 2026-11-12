# FID evaluation

Compute Fréchet Inception Distance between a reference image folder and generated
images. Run commands from the repository root after installing xDiT and
`clean-fid`:

```bash
pip install clean-fid
python benchmark/fid/compute_fid.py \
    --ref /path/to/reference_images --sample /path/to/generated_images
```

Evaluation defaults to the `cuda` device (CUDA or ROCm); add `--device cpu` for
CPU evaluation. Clean-FID may download its feature-extractor weights on first use.
Use the same dataset, sample selection, resolution, and preprocessing when
comparing runs. Record those settings alongside the model and generation options.
FID alone does not establish prompt fidelity or perceptual quality.

## Optional COCO caption workflow

The supplied generation scripts implement a specific 30,000-image, 256×256
experiment using FLUX.1-dev or PixArt-alpha. They use the **first 30,000 JSON
entries**, without filtering by dataset split. They are separate from the unified
runner and do not provide its dataset, caching, or quantization options.

1. Obtain the Karpathy [COCO caption JSON](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)
   (`dataset_coco.json`) and all COCO images referenced by those entries. This can
   include both [train2014](https://images.cocodataset.org/zips/train2014.zip) and
   [val2014](https://images.cocodataset.org/zips/val2014.zip); a validation-only
   directory is insufficient when selected entries refer to training images.
2. Put the selected image files together in one input directory, then prepare
   the references:

```bash
python benchmark/fid/process_ref_data.py \
    --json_file dataset_coco.json --num_samples 30000 \
    --input_dir /path/to/coco_images --output_dir /path/to/reference_images
```

The processor resizes to 256×256 by default. Check its logged success/failure
counts; it logs missing images and continues.

1. Generate matching samples with the dedicated FLUX script, with access to the
   gated FLUX.1-dev checkpoint and sufficient memory on four GPUs:

```bash
torchrun --nproc_per_node=4 benchmark/fid/flux_generate.py \
    --model black-forest-labs/FLUX.1-dev \
    --height 256 --width 256 \
    --pipefusion_parallel_degree 4 --warmup_steps 1 \
    --num_inference_steps 28 --output_type pil \
    --caption_file dataset_coco.json \
    --sample_images_folder /path/to/generated_images
```

This script hard-codes 256×256 images, guidance 1.5, maximum sequence length 256,
and batches of 120 captions. It uses the first sentence of each selected entry
and saves under that entry's filename. `--height`, `--width`, and
`--guidance_scale` do not change those hard-coded values. Check that both folders
contain exactly the selected 30,000 files before scoring.

`benchmark/fid/generate.sh` is an alternative editable launcher for these scripts;
its defaults use eight GPUs and a local PixArt checkpoint. Replace its paths,
model selection, and layout before use. Its output variable is spelled
`SAMPLE_IMAGES_FOLODER` in the script.

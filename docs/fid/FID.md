
### Procedure
#### Prerequisite
Firstly, Install the following additional dependencies before testing:
```
pip3 install datasets tensorflow scipy
```

#### Sample Batch Generation
Then you can use `scripts/generate.py` to generate images with COCO captions. An example command is as follow:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --rdzv-endpoint=localhost:8070 scripts/generate.py --pipeline pixart --scheduler dpm-solver --warmup_steps 4 --parallelism pipeline --no_cuda_graph --dataset coco --no_split_batch --guidance_scale 2.0 --pp_num_patch 8.0
```

After that, you can use `scripts/npz.py` to pack the generated images into a `.npz` file, where the `$GENERATED_IMAGES_FOLODER` is the path you saved the generated images, while `$IMAGES_NUM` is the total images count:
```
python3 scripts/npz.py --sample_dir $GENERATED_IMAGES_FOLODER --num $IMAGES_NUM
```

#### Reference Batch Generation
To get the COCO ref images, you can run the following commands:
```
python3 scripts/dump_coco.py
```
Then you could use `scripts/npz.py` to pack the reference images into a `.npz` file as well, where the `$REF_IMAGES_FOLODER` is the path you saved the reference images, while `$IMAGES_NUM` is the total images count:
```
python3 scripts/npz.py --sample_dir $REF_IMAGES_FOLODER --num $IMAGES_NUM
```

#### Evaluate the results
After you completing the above procedure, you'll get two .npz files `$SAMPLE_NPZ` and `$REF_NPZ` (replace them with the corresponding files). You can evalute the results with `scripts/evaluator` by running:
```
python3 scripts/evaluator.py --ref_batch $REF_NPZ --sample_batch $SAMPLE_NPZ
```
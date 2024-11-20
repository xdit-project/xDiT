### Procedure
#### Prerequisite
Firstly, Install the following additional dependencies before testing:
```
pip3 install clean-fid
```

#### Reference Batch Preparation
Download the COCO dataset from [here](https://huggingface.co/datasets/HuggingFaceM4/COCO), only the validation set and caption dataset are needed. Unzip the [val2014.zip](http://images.cocodataset.org/zips/val2014.zip) and [caption_datasets.zip](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) and you'll get the files in the following format:
```
val2014/
    COCO_val2014_000000xxxxxx.jpg
    ...
dataset_coco.json
dataset_flickr30k.json
dataset_flickr8k.json
```
Then run the following command to process the reference images:
```
python3 process_ref_data.py --coco_json dataset_coco.json --num_samples 30000 --input_dir $PATH_TO_VAL2014 --output_dir $REF_IMAGES_FOLODER
```

#### Sample Batch Generation
Run the following command to generate the sample images:
```
bash ./benchmark/fid/generate.sh
``` 
You can edit the `generate.sh` to change the model type, caption file, sample images folder, etc.

#### Evaluate the results using clean-fid
After you completing the above procedure, you'll get the reference images and generated images in the `$REF_IMAGES_FOLODER` and `$SAMPLE_IMAGES_FOLODER` (replace them with the corresponding folders). You can evalute the results with `compute_fid.py` by running:

```
python compute_fid.py --ref_path $REF_IMAGES_FOLODER --sample_path $SAMPLE_IMAGES_FOLODER
```

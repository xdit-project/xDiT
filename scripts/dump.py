import json
import os
from argparse import ArgumentParser

from datasets import load_dataset
from tqdm import tqdm, trange

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="imagenet", choices=["coco", "imagenet"])
    parser.add_argument("--output_root", type=str, default="./images")
    args = parser.parse_args()

    if args.dataset == "coco":
        dataset = load_dataset("HuggingFaceM4/COCO", name="2014_captions", split="validation")
    elif args.dataset == "imagenet":
        dataset = load_dataset("evanarlian/imagenet_1k_resized_256", split="val")
    dataset = dataset.shuffle(seed=42)

    if args.dataset == "coco":
        prompt_list = []
        for i in trange(len(dataset["sentences_raw"])):
            prompt = dataset["sentences_raw"][i][i % len(dataset["sentences_raw"][i])]
            prompt_list.append(prompt)

        os.makedirs(args.output_root, exist_ok=True)
        prompt_path = os.path.join(args.output_root, "prompts.json")
        with open(prompt_path, "w") as f:
            json.dump(prompt_list, f, indent=4)

    os.makedirs(os.path.join(args.output_root, "images"), exist_ok=True)

    # dataset = load_dataset("HuggingFaceM4/COCO", name="2014_captions", split="validation")
    for i, image in enumerate(tqdm(dataset["image"][:1000])):
        image.save(os.path.join(args.output_root, "images", f"{i:04}.png"))

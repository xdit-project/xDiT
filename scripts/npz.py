from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:04d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--sample_dir", type=str, required=True, help="Path to the directory containing .png samples.")
    args.add_argument("--num", type=int, default=10_000, help="Number of samples to include in the .npz file.")
    args = args.parse_args()
    create_npz_from_sample_folder(args.sample_dir, args.num)
"""Seed the HF cache with models that would unblock uncovered runners.

Ordered cheapest first, and each download is checked against free space before it starts so a large
repo cannot fill the disk and take the running validations down with it.
"""
import shutil
import sys
import time

from huggingface_hub import snapshot_download

HEADROOM_GB = 60

REPOS = [
    "black-forest-labs/FLUX.2-klein-4B",
    "Tongyi-MAI/Z-Image-Turbo",
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "black-forest-labs/FLUX.1-Kontext-dev",
]


def free_gb():
    return shutil.disk_usage("/.cache").free / 1e9


for repo in REPOS:
    start = time.time()
    print(f"[seed] {repo}: {free_gb():.0f} GB free", flush=True)
    if free_gb() < HEADROOM_GB:
        print(f"[seed] stopping before {repo}: under {HEADROOM_GB} GB headroom", flush=True)
        break
    try:
        snapshot_download(repo, allow_patterns=None, max_workers=8)
        print(f"[seed] {repo}: done in {time.time()-start:.0f}s, {free_gb():.0f} GB free", flush=True)
    except Exception as error:
        print(f"[seed] {repo}: FAILED {type(error).__name__}: {error}", flush=True)
print("[seed] finished", flush=True)

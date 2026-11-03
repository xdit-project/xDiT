#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Materialize the VDN-H3 transformer into diffusers layout for the xDiT runner.

The overlay ships a standalone materializer (kevin-mi/VDN-H3-overlay
``_overlay/materialize.py``) that folds VDN's two LoRA adapters into the base
MiniMax-H3 transformer and attaches the linear branch. The full materializer
also re-exports the VAE into SGLang layout and links Qwen3-VL components; the
xDiT runner reuses the stock diffusers base pipeline for those, so this driver
runs only the two transformer steps:

    _prefuse_transformer   14 shards with both adapters folded (W += scale*B@A),
                           + sglang_vdn_linear_branch.safetensors (800 keys)
                           + config.json (base + hybrid_attention)
                           + the safetensors index
    _write_rope_inv_freq   the RoPE buffer the diffusers export drops

Output: ``<output>/transformer/`` -- pass ``<output>`` as ``--vdn_transformer_dir``.

Needs torch, safetensors and huggingface_hub. The source checkpoints are
downloaded into the Hugging Face cache (``HF_HOME``) unless ``--source`` and
``--materialize-py`` point at local copies.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

VDN_REPO = "OpenVDN/vdn-minimax-h3"
OVERLAY_REPO = "kevin-mi/VDN-H3-overlay"
OVERLAY_REVISION = "7de18275dddfe59da36a234e222bcdd274963bc3"


def _load_materializer(path: str):
    spec = importlib.util.spec_from_file_location("vdn_h3_materialize", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_snapshot(repo: str, *, revision: str | None, allow_patterns=None) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, revision=revision, allow_patterns=allow_patterns, max_workers=8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=None,
        help=f"{VDN_REPO} snapshot dir (has h3-base/ and stage-dmd-step-250/). Default: resolve from the HF cache.",
    )
    parser.add_argument(
        "--materialize-py",
        default=None,
        help=f"Path to {OVERLAY_REPO} _overlay/materialize.py. Default: resolve from the HF cache.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output dir; the transformer is written to <output>/transformer/.",
    )
    args = parser.parse_args()

    source = args.source or _resolve_snapshot(VDN_REPO, revision=None)
    materialize_py = args.materialize_py
    if materialize_py is None:
        overlay = _resolve_snapshot(OVERLAY_REPO, revision=OVERLAY_REVISION, allow_patterns=["_overlay/*"])
        materialize_py = os.path.join(overlay, "_overlay", "materialize.py")

    for required in ("h3-base", "stage-dmd-step-250"):
        if not os.path.isdir(os.path.join(source, required)):
            print(f"ERROR: source {source!r} is missing {required}/", file=sys.stderr)
            return 2

    output = os.path.abspath(args.output)
    os.makedirs(os.path.join(output, "transformer"), exist_ok=True)

    materializer = _load_materializer(materialize_py)
    print(f"[vdn-h3] source     : {source}")
    print(f"[vdn-h3] materialize: {materialize_py}")
    print(f"[vdn-h3] output     : {output}")

    record = materializer._prefuse_transformer(source_dir=source, output_dir=output)
    materializer._write_rope_inv_freq(source_dir=os.path.join(source, "h3-base"), output_dir=output)

    print(
        f"[vdn-h3] done: folded {record['merge']['pairs_merged']} LoRA pairs into "
        f"{record['merge']['tensors_touched']} tensors; linear branch "
        f"{record['linear_branch']['tensors']} keys."
    )
    print(f"[vdn-h3] run xDiT with --vdn_transformer_dir {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

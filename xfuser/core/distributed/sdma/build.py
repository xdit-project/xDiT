"""Build the SDMA transport (transport.hip) into a shared library; needs hipcc, not a GPU.

Standard library only, so an image build can run it without importing xfuser
(whose dependencies may look for a GPU on import):

    python xfuser/core/distributed/sdma/build.py --out DIR [--offload-arch gfx942,gfx950]

The library, DIR/libxdit_sdma.so, is what XDIT_SDMA_LIB points at. Built into
this directory, transport.py loads it while it matches the source.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

SOURCE = Path(__file__).with_name("transport.hip")
LIB_NAME = "libxdit_sdma.so"
# What `build` compiles for by default.
BUILD_ARCHS = ("gfx942", "gfx950")


def source_hash() -> str:
    return hashlib.sha1(SOURCE.read_bytes()).hexdigest()


def build(out_dir, archs) -> Path:
    """Compile the transport for `archs` (e.g. ["gfx942", "gfx950"]) into out_dir/LIB_NAME."""
    hipcc = shutil.which("hipcc") or str(Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "bin" / "hipcc")
    if not Path(hipcc).exists():
        raise RuntimeError(
            "the SDMA transport needs hipcc to build. Build it where hipcc is available with `python "
            f"{Path(__file__)} --out DIR` and set XDIT_SDMA_LIB=DIR/{LIB_NAME}"
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    so = out / LIB_NAME
    tmp = so.with_suffix(f".{os.getpid()}.tmp")
    arch_flags = [f"--offload-arch={a}" for a in archs]
    cmd = [hipcc, "-O2", "-shared", "-fPIC", *arch_flags, "-o", str(tmp), str(SOURCE), "-lhsa-runtime64"]
    subprocess.run(cmd, check=True)
    tmp.rename(so)
    (out / (LIB_NAME + ".source-sha1")).write_text(source_hash() + "\n")
    return so


def main(argv=None, prog=None) -> None:
    parser = argparse.ArgumentParser(prog=prog, description=f"compile the SDMA transport into DIR/{LIB_NAME}")
    parser.add_argument("--out", required=True, metavar="DIR")
    parser.add_argument("--offload-arch", default=",".join(BUILD_ARCHS), help="comma-separated GPU architectures")
    args = parser.parse_args(argv)
    print(build(args.out, [a for a in args.offload_arch.split(",") if a]))


if __name__ == "__main__":
    main()

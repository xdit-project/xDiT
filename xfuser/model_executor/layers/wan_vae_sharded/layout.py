"""How the sharded Wan VAE decode splits latent rows over ranks (no torch / accelerator imports)."""

from typing import List, Tuple


def bands(height: int, world: int) -> List[Tuple[int, int]]:
    """(first row, rows) per rank: `height` rows split as evenly as possible, the larger bands first."""
    if world < 1 or height < world:
        raise ValueError(f"cannot split {height} latent rows over {world} ranks")
    base, extra = divmod(height, world)
    sizes = [base + (1 if r < extra else 0) for r in range(world)]
    return [(sum(sizes[:r]), sizes[r]) for r in range(world)]

import pytest
import torch

from xfuser.model_executor.layers.ltx2.na3d_eager_attn import LTX2VideoVaeEagerSdpaAttnProcessor
from xfuser.model_executor.layers.ltx2.na3d_mfma_flash import LTX2VideoVaeMfmaAttnProcessor

pytestmark = pytest.mark.rocm


def _flash_processor() -> LTX2VideoVaeMfmaAttnProcessor:
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("AITER flash-NA3D requires a ROCm GPU")
    try:
        return LTX2VideoVaeMfmaAttnProcessor()
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"AITER flash-NA3D is unavailable: {exc}")


def _attention(kernel_size: tuple[int, int, int]):
    decoder = pytest.importorskip("diffusers.models.autoencoders.ltx2_diffusion_decoder")
    torch.manual_seed(0)
    return (
        decoder.LTX2VideoVaeNeighborhoodAttention(dim=128, kernel_size=kernel_size, head_dim=64)
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )


def _fail_fallback(*args, **kwargs):
    raise AssertionError("expected the AITER kernel path, not the SDPA fallback")


@pytest.mark.parametrize(
    ("kernel_size", "grid"),
    [
        ((3, 7, 7), (5, 16, 24)),
        ((3, 5, 5), (4, 12, 16)),
        ((11, 11, 11), (11, 12, 32)),
    ],
)
@torch.no_grad()
def test_flash_na3d_matches_tiled_sdpa(kernel_size, grid):
    processor = _flash_processor()
    processor._fallback = _fail_fallback
    attn = _attention(kernel_size)
    hidden_states = torch.randn(1, *grid, 128, device="cuda", dtype=torch.bfloat16)

    output = processor(attn, hidden_states)
    reference = LTX2VideoVaeEagerSdpaAttnProcessor()(attn, hidden_states)

    assert output.shape == reference.shape
    relative_l2 = torch.linalg.vector_norm((output - reference).float()) / torch.linalg.vector_norm(reference.float())
    assert relative_l2 < 1e-2


@torch.no_grad()
def test_narrow_grid_uses_tiled_sdpa():
    # Decoder tiling can leave remnant tiles narrower than the kernel's
    # minimum width of 16; those must take the SDPA path instead of asserting.
    processor = _flash_processor()
    attn = _attention((3, 5, 5))
    hidden_states = torch.randn(1, 4, 12, 8, 128, device="cuda", dtype=torch.bfloat16)

    output = processor(attn, hidden_states)
    reference = LTX2VideoVaeEagerSdpaAttnProcessor()(attn, hidden_states)

    assert torch.equal(output, reference)

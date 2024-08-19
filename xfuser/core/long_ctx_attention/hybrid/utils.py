from yunchang.ring import (
    zigzag_ring_flash_attn_func,
    stripe_flash_attn_func,
)
from ..ring import ring_flash_attn_func

RING_IMPL_DICT = {
    "basic": ring_flash_attn_func,
    "zigzag": zigzag_ring_flash_attn_func,
    "strip": stripe_flash_attn_func,
}

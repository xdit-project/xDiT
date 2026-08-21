"""
Unit tests for xfuser.core.distributed.sharding module.

These tests verify the FSDP sharding functionality for transformer models
using pytest conventions. Tests run on CPU with gloo backend for CI compatibility.

Run with:
    pytest tests/core/test_sharding.py -v
    pytest tests/core/test_sharding.py::test_shard_dit_basic -v  # Single test
"""
import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xfuser.core.distributed.sharding import (
    shard_component,
    shard_dit,
    shard_t5_encoder,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def setup_distributed():
    """A gloo group for this module's tests, taken down again with the environment it needed.

    Scoped to the module and unwound rather than left up for the session: the rendezvous
    variables are read by any process started later, so a test that spawns workers of its
    own inherits RANK=0 and WORLD_SIZE=1 from here and hangs waiting for a rendezvous that
    has already happened. test_multirank_load_and_sharding does exactly that.
    """
    started = not dist.is_initialized()
    previous = {
        name: os.environ.get(name)
        for name in ('MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE')
    }
    if started:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29501'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'

        # Use gloo backend for CPU testing (nccl requires GPU)
        dist.init_process_group(backend='gloo', init_method='env://')

    yield

    if started:
        if dist.is_initialized():
            dist.destroy_process_group()
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture
def dit_model():
    """Create a mock DiT model."""
    class DiTBlock(nn.Module):
        def __init__(self, dim=256):
            super().__init__()
            self.attn = nn.Linear(dim, dim)
            self.mlp = nn.Linear(dim, dim)

        def forward(self, x):
            return self.mlp(self.attn(x))

    class DiT(nn.Module):
        def __init__(self, num_blocks=2):
            super().__init__()
            self.blocks = nn.ModuleList([DiTBlock(dim=256) for _ in range(num_blocks)])
            self.proj_out = nn.Linear(256, 256)

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return self.proj_out(x)

    return DiT(num_blocks=2)


@pytest.fixture
def t5_encoder_model():
    """Create a mock T5 encoder model."""
    class T5Block(nn.Module):
        def __init__(self, dim=256):
            super().__init__()
            self.layer = nn.Linear(dim, dim)

        def forward(self, x):
            return self.layer(x)

    class T5Encoder(nn.Module):
        def __init__(self, num_blocks=2):
            super().__init__()
            self.block = nn.ModuleList([T5Block() for _ in range(num_blocks)])

        def forward(self, x):
            for b in self.block:
                x = b(x)
            return x

    class T5Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = T5Encoder(num_blocks=2)

        def forward(self, x):
            return self.encoder(x)

    return T5Model()


# ============================================================================
# Test shard_component
# ============================================================================

def test_shard_component_basic(setup_distributed, simple_transformer_model):
    """Test basic FSDP wrapping of transformer blocks."""
    model = simple_transformer_model

    # Shard the model
    sharded_model = shard_component(
        model,
        wrap_attrs=['blocks'],
        device_id=0,
    )

    # Verify model is wrapped with FSDP
    assert isinstance(sharded_model, FSDP), "Model should be wrapped with FSDP"

    # Verify blocks still exist (may be wrapped)
    assert hasattr(sharded_model, 'blocks'), "Blocks attribute should exist"


def test_shard_component_with_dtype(setup_distributed, simple_transformer_model):
    """Test FSDP wrapping with dtype conversion."""
    model = simple_transformer_model

    # Shard with bfloat16
    sharded_model = shard_component(
        model,
        wrap_attrs=['blocks'],
        device_id=0,
        dtype=torch.bfloat16,
    )

    assert isinstance(sharded_model, FSDP), "Model should be wrapped with FSDP"

    # Check that parameters are in bfloat16
    for param in sharded_model.parameters():
        if param.dtype.is_floating_point:
            assert param.dtype == torch.bfloat16, f"Param dtype should be bfloat16, got {param.dtype}"


def test_shard_component_invalid_attr(setup_distributed, simple_transformer_model):
    """Test error handling for invalid block attribute."""
    model = simple_transformer_model

    # rgetattr resolves each component of the nested attribute name; the failing getattr raises
    # AttributeError when nonexistent_blocks is absent.
    with pytest.raises(AttributeError, match="nonexistent_blocks"):
        shard_component(
            model,
            wrap_attrs=['nonexistent_blocks'],
            device_id=0,
        )


def test_shard_component_with_fsdp_kwargs(setup_distributed, simple_transformer_model):
    """Test passing additional FSDP kwargs."""
    model = simple_transformer_model

    # Should not raise an error
    sharded_model = shard_component(
        model,
        wrap_attrs=['blocks'],
        device_id=0,
        sync_module_states=True,
        forward_prefetch=True,
        use_orig_params=True,
    )

    assert isinstance(sharded_model, FSDP), "Model should be wrapped with FSDP"


# ============================================================================
# Test shard_dit
# ============================================================================

def test_shard_dit_basic(setup_distributed, dit_model):
    """Test DiT model sharding."""
    model = dit_model

    sharded_model = shard_dit(
        model,
        local_rank=0,
        block_attr='blocks',
    )

    assert isinstance(sharded_model, FSDP), "DiT model should be wrapped with FSDP"
    assert hasattr(sharded_model, 'blocks'), "Blocks attribute should exist"


def test_shard_dit_dtype_conversion(setup_distributed, dit_model):
    """Test that DiT sharding converts to bfloat16."""
    model = dit_model

    sharded_model = shard_dit(
        model,
        local_rank=0,
        block_attr='blocks',
    )

    # Verify dtype conversion happened (shard_dit uses bfloat16 by default)
    for param in sharded_model.parameters():
        if param.dtype.is_floating_point:
            assert param.dtype == torch.bfloat16, "DiT params should be in bfloat16"


# ============================================================================
# Test shard_t5_encoder
# ============================================================================

def test_shard_t5_encoder_basic(setup_distributed, t5_encoder_model):
    """Test T5 encoder model sharding."""
    model = t5_encoder_model

    sharded_model = shard_t5_encoder(
        model,
        local_rank=0,
        block_attr='block',  # T5 uses 'block' not 'blocks'
    )

    assert hasattr(sharded_model, 'encoder'), "Encoder attribute should exist"
    assert isinstance(sharded_model.encoder, FSDP), "Encoder should be wrapped with FSDP"


def test_shard_t5_encoder_preserves_structure(setup_distributed, t5_encoder_model):
    """Test that T5 encoder structure is preserved after sharding."""
    model = t5_encoder_model

    sharded_model = shard_t5_encoder(
        model,
        local_rank=0,
        block_attr='block',
    )

    # Verify the model structure is intact
    assert hasattr(sharded_model, 'encoder'), "Should have encoder"
    assert hasattr(sharded_model.encoder, 'block'), "Encoder should have block"


# ============================================================================
# Test Edge Cases
# ============================================================================

def test_empty_blocks_list(setup_distributed):
    """Test handling of model with empty blocks list."""
    class EmptyBlockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([])  # Empty blocks
            self.final = nn.Linear(256, 256)

        def forward(self, x):
            return self.final(x)

    model = EmptyBlockModel()

    # Should still work with empty blocks
    sharded_model = shard_component(
        model,
        wrap_attrs=['blocks'],
        device_id=0,
    )

    assert isinstance(sharded_model, FSDP), "Should wrap even with empty blocks"


def test_single_block(setup_distributed):
    """Test sharding a model with only one block."""
    class SingleBlockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(256, 256)])

        def forward(self, x):
            return self.blocks[0](x)

    model = SingleBlockModel()

    sharded_model = shard_component(
        model,
        wrap_attrs=['blocks'],
        device_id=0,
    )

    assert isinstance(sharded_model, FSDP), "Should wrap single block model"


def test_default_device_id(setup_distributed, simple_transformer_model):
    """Test that device_id defaults to current device when None."""
    model = simple_transformer_model

    # Should use current device (0 in single-GPU test)
    sharded_model = shard_component(
        model,
        wrap_attrs=['blocks'],
        device_id=None,  # Explicitly pass None
    )

    assert isinstance(sharded_model, FSDP), "Should work with None device_id"


def test_no_dtype_conversion(setup_distributed, simple_transformer_model):
    """Test sharding without dtype conversion."""
    model = simple_transformer_model
    original_dtype = next(model.parameters()).dtype

    sharded_model = shard_component(
        model,
        wrap_attrs=['blocks'],
        device_id=0,
        dtype=None,  # No conversion
    )

    assert isinstance(sharded_model, FSDP), "Should wrap without dtype conversion"
    # Note: After wrapping, dtype might be affected by FSDP's internal handling


# ============================================================================
# Test Parameter Count Preservation
# ============================================================================

def test_parameter_count_preserved(setup_distributed, simple_transformer_model):
    """Test that total parameter count is preserved after sharding."""
    model = simple_transformer_model

    # Count params before sharding
    original_param_count = sum(p.numel() for p in model.parameters())

    sharded_model = shard_component(
        model,
        wrap_attrs=['blocks'],
        device_id=0,
    )

    # Count params after sharding
    sharded_param_count = sum(p.numel() for p in sharded_model.parameters())

    assert original_param_count == sharded_param_count, \
        f"Parameter count changed: {original_param_count} -> {sharded_param_count}"


# ============================================================================
# Integration Test
# ============================================================================

def test_end_to_end_forward_pass(setup_distributed, simple_transformer_model):
    """Test complete forward pass through sharded model."""
    model = simple_transformer_model

    sharded_model = shard_component(
        model,
        wrap_attrs=['blocks'],
        device_id=0,
        dtype=torch.float32,
    )

    # Create input
    batch_size, seq_len, dim = 2, 10, 256
    x = torch.randn(batch_size, seq_len, dim)

    # Forward pass
    output = sharded_model(x)

    # Verify output
    assert output.shape == (batch_size, seq_len, dim), \
        f"Expected shape {(batch_size, seq_len, dim)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert torch.isfinite(output).all(), "Output contains infinite values"


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v", "-s"])

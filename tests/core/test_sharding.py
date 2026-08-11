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
    has already happened. test_roadmap2_distributed does exactly that.
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


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v", "-s"])

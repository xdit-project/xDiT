"""Check upstream call signatures for Z-Image block fallbacks."""

from types import SimpleNamespace

import pytest

from xfuser.model_executor.models.transformers import transformer_z_image as z_image


@pytest.mark.parametrize("adaln_input", [None, object()])
def test_unmodulated_fallback_uses_original_signature(monkeypatch, adaln_input):
    block = SimpleNamespace(modulation=False)
    x, attn_mask, freqs_cis, output = (object() for _ in range(4))
    calls = []

    def upstream(self, x, attn_mask, freqs_cis, adaln_input):
        calls.append((self, x, attn_mask, freqs_cis, adaln_input))
        return output

    monkeypatch.setattr(z_image, "_UPSTREAM_BLOCK_FORWARD", upstream)

    result = z_image._fused_block_forward(block, x, attn_mask, freqs_cis, adaln_input)

    assert result is output
    assert calls == [(block, x, attn_mask, freqs_cis, adaln_input)]


@pytest.mark.parametrize("modulation", [False, True])
def test_per_token_fallback_passes_additional_arguments(monkeypatch, modulation):
    block = SimpleNamespace(modulation=modulation)
    x, attn_mask, freqs_cis, adaln_input, noise_mask, adaln_noisy, adaln_clean, output = (object() for _ in range(8))
    calls = []

    def upstream(self, x, attn_mask, freqs_cis, adaln_input, noise_mask, adaln_noisy, adaln_clean):
        calls.append((self, x, attn_mask, freqs_cis, adaln_input, noise_mask, adaln_noisy, adaln_clean))
        return output

    monkeypatch.setattr(z_image, "_UPSTREAM_BLOCK_FORWARD", upstream)

    result = z_image._fused_block_forward(
        block, x, attn_mask, freqs_cis, adaln_input, noise_mask, adaln_noisy, adaln_clean
    )

    assert result is output
    assert calls == [(block, x, attn_mask, freqs_cis, adaln_input, noise_mask, adaln_noisy, adaln_clean)]

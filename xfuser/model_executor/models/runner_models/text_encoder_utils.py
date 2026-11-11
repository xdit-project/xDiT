"""Text-encoder helpers shared by runners."""

import torch


def trim_text_encoder(text_encoder, last_layer: int) -> None:
    """Drop what the prompt embeddings never read from a Qwen3-VL text encoder: the
    decoder layers after ``last_layer``, the final norm and the LM head.

    For a pipeline that reads hidden states up to ``hidden_states[last_layer]``. The
    final norm only ever feeds the last hidden state and the logits are discarded, so
    those hidden states stay bit-exact while encoding takes less time and memory.
    The config keeps its original layer count.
    """
    language_model = text_encoder.model.language_model
    if len(language_model.layers) <= last_layer:
        return
    language_model.layers = language_model.layers[:last_layer]
    language_model.norm = torch.nn.Identity()
    text_encoder.lm_head = torch.nn.Identity()

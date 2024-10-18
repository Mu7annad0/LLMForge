from dataclasses import dataclass


@dataclass
class GPT_CONFIG_124M:
    vocab_size: int =  50257
    context_length: int = 1024
    embed_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate : float = 0.1
    qkv_bias : bool = False

from dataclasses import dataclass
import torch

@dataclass
class GPT_CONFIG_124M:
    vocab_size: int =  50257
    context_length: int = 1024 # max sequence length
    embed_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate : float = 0.1
    qkv_bias : bool = False


@dataclass
class Llama2_CONFIG_7B:
    vocab_size: int = 32000
    context_length: int = 4096
    embed_dim: int = 4096
    n_heads: int = 32
    n_layers: int = 32
    hidden_dim: int = 11008
    dtype = torch.bfloat16
    
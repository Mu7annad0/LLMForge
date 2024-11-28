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
class Llama31_CONFIG_8B:
    vocab_size: int = 128256
    context_length: int = 131_072
    embed_dim: int = 4096
    num_heads: int = 32
    num_kv_heads: int = 8
    n_layers: int = 32
    hidden_dim: int = 11008
    dtype = torch.bfloat16
    flash = True
    cache=True
    rope_freq = {
        "factor":0.8,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
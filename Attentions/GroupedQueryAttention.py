import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass
from Blocks.Positionals import apply_rope, precompute_rope_params


class GrupedQueryAttention(nn.Module):
    """
    This class implements Grouped Query Attention.

    Grouped Query Attention is a modification of traditional multi-head attention
    that reduces the number of key and value heads from num_heads to num_kv_heads.
    """

    def __init__(self, cfg):
        """
        Initialize the Grouped Query Attention module.

        Args:
            cfg: Configuration object containing the following attributes:
                - num_heads: Number of query heads.
                - num_kv_heads: Number of key and value heads.
                - embed_dim: Dimensionality of embeddings.
                - context_length: Maximum sequence length.
                - flash: Whether to use Flash Attention.
                - dtype: Data type for the module.
        """
        super().__init__()

        self.flash = cfg.flash
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.num_kv_groups = cfg.num_heads // cfg.num_kv_heads
        self.head_dim = cfg.embed_dim // cfg.num_heads
        self.embed_dim = cfg.embed_dim

        self.wq = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False, dtype=cfg.dtype)
        self.wk = nn.Linear(cfg.embed_dim, self.num_kv_heads * self.head_dim, bias=False, dtype=cfg.dtype)
        self.wv = nn.Linear(cfg.embed_dim, self.num_kv_heads * self.head_dim, bias=False, dtype=cfg.dtype)
        self.wo = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False, dtype=cfg.dtype)

        self.register_buffer(
            "mask", torch.triu(torch.ones(cfg.context_length, cfg.context_length), diagonal=1)
        )
        # Precompute cosine and sine values for Rotary Positional Embedding (RoPE)
        cos, sin = precompute_rope_params(cfg.embed_dim, max_seq_length=cfg.context_length)
        self.register_buffer("cos", cos.to(cfg.dtype))
        self.register_buffer("sin", sin.to(cfg.dtype))
        self.cache = None

    def forward(self, x):
        """
        Compute the Grouped Query Attention output.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seqlen, _ = x.shape

        # Project input to query, key, and value dimensions
        q = self.wq(x).view(batch_size, seqlen, self.num_heads, self.head_dim).transpose(1, 2)      # (batch_size, num_heads, seq_len, head_dim)
        k = self.wk(x).view(batch_size, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)   # (batch_size, num_kv_heads, seq_len, head_dim)
        v = self.wv(x).view(batch_size, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)   # (batch_size, num_kv_heads, seq_len, head_dim)

        # Apply Rotary Positional Embedding (RoPE) to queries and keys
        q = apply_rope(q, self.cos, self.sin)
        k = apply_rope(k, self.cos, self.sin)

        # Repeat key and value heads to match the number of query heads
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        if self.flash:
            # Compute attention using Flash Attention
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=self.mask[:seqlen, :seqlen],
                scale=self.head_dim ** 0.5,
                is_causal=True
            )
        else:
            # Compute attention using traditional matrix multiplication
            att = q @ k.transpose(2, 3)  # Shape: (batch_size, num_heads, num_tokens, num_tokens)
            mask_bool = self.mask.bool()[:seqlen, :seqlen]
            att.masked_fill(mask_bool, -torch.inf)
            att = torch.softmax(
                att / k.shape[-1]**0.5,  # Scale by sqrt(head_dim)
                dim=-1
            )
            y = (att @ v)

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(batch_size, seqlen, self.embed_dim)
        y = self.wo(y)

        return y


# @dataclass
# class config:
#     batch_size = 1
#     context_length = 8192
#     max_context_length = 3000
#     embed_dim = 4096
#     num_heads = 32
#     num_kv_heads: int = 8
#     flash = True
#     dtype = None


# cfg = config
# mha = GrupedQueryAttention(cfg)

# example_batch = torch.randn((cfg.batch_size, cfg.max_context_length, cfg.embed_dim))
# output = mha(example_batch)
# print(f"{output=}")
# print("W_key:", mha.wk.weight.shape)
# print("W_value:", mha.wv.weight.shape)
# print("W_query:", mha.wq.weight.shape)

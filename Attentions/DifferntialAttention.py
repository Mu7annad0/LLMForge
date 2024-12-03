import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from Blocks.Normalizations import RMSNormalization
from Blocks.Positionals import precompute_rope_params, apply_rope


# The Algorithm
# def DiffAttn(X, W_q, W_k, W_v, λ):
#     Q1, Q2 = split(X @ W_q)
#     K1, K2 = split(X @ W_k)
#     V = X @ W_v
#     # Qi, Ki: [b, n, d]; V: [b, n, 2d]
#     s = 1 / sqrt(d)
#     A1 = Q1 @ K1.transpose(−1, −2) ∗ s
#     A2 = Q2 @ K2.transpose(−1, −2) ∗ s
#     return (softmax(A1) − λ softmax(A2)) @ V

# def MultiHead(X, W_q, W_k, W_v, W_o, λ):
#     O = GroupNorm([DiffAttn(X, W_qi, W_ki, W_vi, λ) for i in range(h)])
#     O = O ∗ (1 − λinit)
#     return Concat(O) @ W_o


class DifferntialAttention(nn.Module):
    def __init__(self, cfg):
        """
        Differential Attention mechanism that applies a novel attention formulation
        by computing two separate attention scores (A1 and A2) and combining them
        with a learnable lambda parameter (λ)
        
        Args:
            cfg: Configuration object containing the following attributes:
                - flash: Whether to use FlashAttention for optimized computation.
                - lambda_init: Initial value for lambda (λ).
                - context_length: Maximum sequence length.
                - embed_dim: Dimensionality of embeddings.
                - num_heads: Number of attention heads.
        """
        super().__init__()
        self.flash = cfg.flash
        self.init_lambda = 0.8
        self.context_length = cfg.context_length
        self.embed_dim = cfg.embed_dim
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.embed_dim // cfg.num_heads
        self.scaler = self.head_dim ** -0.5

        # Combined projection for Q, K, V
        self.w_qkv = nn.Linear(cfg.embed_dim, 3*cfg.embed_dim, bias=False)
        self.wo = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)

        # Group Normalization for each attention head
        self.norm = nn.GroupNorm(
            num_groups=self.num_heads, 
            num_channels=self.embed_dim, 
            eps=1e-5, 
            affine=True
        )
        
        # Learnable lambda parameters for Q and K splits
        self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim).normal_(mean=0, std=0.1))

        # RoPE position encoding parameters
        cos, sin = precompute_rope_params(cfg.embed_dim, max_seq_length=cfg.context_length)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        # Causal mask for attention
        self.register_buffer(
            "mask", torch.triu(torch.ones(cfg.context_length, cfg.context_length), diagonal=1)
        )

    def forward(self, x):
        """
        Forward pass of the differential attention mechanism.
        Args:
            x: Input tensor of shape (batch_size, sequence_length, embed_dim)
            
        Returns:
            Tensor of shape (batch_size, sequence_length, embed_dim)
        """
        bsz, seqlen, embed_dim = x.shape

        # Project input to Q, K, V
        qkv = self.w_qkv(x).chunk(3, dim=-1)
        q, k, v = [x.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2) for x in qkv]
        # shape q, k, v --> (batch_size, num_heads, seq_len, head_dim)

        # Apply rotary position embeddings
        q = apply_rope(q, self.cos, self.sin)
        k = apply_rope(k, self.cos, self.sin)

        # Split Q and K for differential attention
        q1, q2 = torch.chunk(input=q, chunks=2, dim=-1)  # (batch_size, num_heads, seq_len, head_dim // 2)
        k1, k2 = torch.chunk(input=k, chunks=2, dim=-1)  # (batch_size, num_heads, seq_len, head_dim // 2)

        # Compute the differential scaling factor (λ) by combining learnable parameters and 
        # applying an exponential function for positive scaling, with an initial bias (`self.init_lambda`).
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim = -1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim = -1).float()).type_as(q2)
        lambda_ = lambda_1 - lambda_2 + self.init_lambda

        if self.flash:
            # Use Flash Attention for faster computation
            attn_mask = self.mask[:seqlen, :seqlen].unsqueeze(0).unsqueeze(1)
            a1 = F.scaled_dot_product_attention(
                q1, k1, v,
                attn_mask=attn_mask,
                scale=self.scaler,
            )
            a2 = F.scaled_dot_product_attention(
                q2, k2,v,
                attn_mask=attn_mask,
                scale=self.scaler,
            )
            out = a1 - (lambda_ * a2)  # -->(bs, num_heads, seqlen, head_dim)
        else:
            # Standard attention computation
            a1 = q1 @ k1.transpose(2, 3)
            a2 = q2 @ k2.transpose(2, 3)

            # Masking out the upper triangular part of the attention matrix
            mask_bool = self.mask.bool()[:seqlen, :seqlen].unsqueeze(0).unsqueeze(1)
            a1.masked_fill(mask_bool, -torch.inf)
            a2.masked_fill(mask_bool, -torch.inf)

            # Apply scaling and softmax
            a1 = torch.softmax((a1/self.scaler), dim=-1)
            a2 = torch.softmax((a2/self.scaler), dim=-1)

            # Compute differential attention
            a = a1 - (lambda_ * a2)
            out = a @ v  # -->(bs, num_heads, seqlen, head_dim)
        
        # Reshape for Group Normalization
        out = out.transpose(1, 2).reshape(bsz, -1, seqlen) # -->(bs, embed_dim, seqlen)

        # Apply Group Normalization
        out = self.norm(out) * (1 - self.init_lambda)

        # Reshape back to original dimensions
        out = out.reshape(bsz, seqlen, embed_dim) # -->(bs, seqlen, embed_dim)
        out = self.wo(out)

        return out
        

# torch.manual_seed(221)
# @dataclass
# class config:
#     flash = True
#     batch_size = 1
#     context_length = 512
#     embed_dim = 256
#     num_heads = 4
#     lambda_init = 0.8

# cfg = config
# mha = DifferntialAttention(cfg)

# example_batch = torch.randn((cfg.batch_size, cfg.context_length, cfg.embed_dim))
# out = mha(example_batch)

# print(f"{out.shape=}")
"""Tensor Product Attention Is All You Need"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from Blocks.Positionals import RotaryPositionEmbedding


class TensorProductAttention(nn.Module):
    """
    Implementation of Tensor Product Attention (TPA) as described in 
    "Tensor Product Attention Is All You Need" paper.
    
    TPA factorizes queries, keys, and values into low-rank tensor products,
    significantly reducing KV cache size during inference while maintaining 
    or improving performance compared to standard attention mechanisms.
    
    Args:
        embed_dim (int): Model embedding dimension
        num_heads (int): Number of attention heads
        context_length (int): Maximum sequence length for positional embeddings
        q_rank (int): Rank for query factorization
        k_rank (int): Rank for key factorization
        v_rank (int): Rank for value factorization
    """
    def __init__(self, embed_dim, num_heads, context_length, q_rank, k_rank, v_rank):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_rank = q_rank
        self.k_rank = k_rank
        self.v_rank = v_rank
        self.head_dim = embed_dim // num_heads

        # Linear projections for head-dimension factors (A factors)
        # These project input to the head dimension component
        self.w_aq = nn.Linear(embed_dim, q_rank * num_heads, bias=False)
        self.w_ak = nn.Linear(embed_dim, k_rank * num_heads, bias=False)
        self.w_av = nn.Linear(embed_dim, v_rank * num_heads, bias=False)

        # Linear projections for token-dimension factors (B factors)
        # These project input to the token dimension component
        self.w_bq = nn.Linear(embed_dim, q_rank * self.head_dim, bias=False)
        self.w_bk = nn.Linear(embed_dim, k_rank * self.head_dim, bias=False)
        self.w_bv = nn.Linear(embed_dim, v_rank * self.head_dim, bias=False)

        self.wo = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rope = RotaryPositionEmbedding(self.head_dim, context_length)

        self.q_scaler = q_rank ** -1
        self.k_scaler = k_rank ** -1
        self.v_scaler = v_rank ** -1

    def forward(self, x):
        """
        Forward pass of Tensor Product Attention
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        b, num_tokens, _ = x.shape

        # Project and reshape the A factors (head dimension components)
        # Shape: (batch_size, num_tokens, num_heads, rank)
        aq = self.w_aq(x).view(b, num_tokens, self.num_heads, self.q_rank)
        ak = self.w_ak(x).view(b, num_tokens, self.num_heads, self.k_rank)
        av = self.w_av(x).view(b, num_tokens, self.num_heads, self.v_rank)

        # Project and reshape the B factors (token dimension components)
        # Shape: (batch_size, num_tokens, rank, head_dim)
        bq = self.w_bq(x).view(b, num_tokens, self.q_rank, self.head_dim)
        bk = self.w_bk(x).view(b, num_tokens, self.k_rank, self.head_dim)
        bv = self.w_bv(x).view(b, num_tokens, self.v_rank, self.head_dim)

        bq, bk = self.rope(bq), self.rope(bk)

        # Compute tensor products and apply scaling
        # Matrix multiplication across the rank dimension, equivalent to tensor product
        # Shape: (batch_size, num_tokens, num_heads, head_dim)
        q = (aq @ bq).transpose(1, 2) * self.q_scaler
        k = (ak @ bk).transpose(1, 2) * self.k_scaler
        v = (av @ bv).transpose(1, 2) * self.v_scaler

        # Apply scaled dot-product attention with causal mask
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, num_tokens, self.embed_dim)
        y = self.wo(y)
        return y


def test_tensor_product_attention():
    """Test function"""
    torch.manual_seed(42)
    x = torch.randn(2, 512, 256)
    tpa = TensorProductAttention(
        embed_dim=256, num_heads=4, context_length=512, q_rank=6, k_rank=2, v_rank=2)
    out = tpa(x)
    print(f"{out.shape=}")

# test_tensor_product_attention()

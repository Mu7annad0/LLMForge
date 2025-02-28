"""Multi-Head Latent Attention (MLA)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from Blocks.Positionals import RotaryPositionEmbedding


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) module as used in DeepSeek-V3 architecture.
    
    This implements an efficient attention mechanism that uses latent representations
    to reduce computational complexity during inference and training. It applies
    Rotary Position Embeddings (RoPE) to a portion of the queries and keys.
    
    Args:
        embed_dim (int): Dimension of the input embeddings
        context_length (int): Maximum sequence length for positional embeddings
        num_heads (int): Number of attention heads
        rank (int): Dimension of the latent space for low-rank projections
        dropout (float): Dropout probability
        qkv (bool, optional): Whether to use bias in linear projections. Defaults to False.
    """
    def __init__(self, embed_dim, context_length, num_heads, rank, dropout, qkv=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.rank = rank  # Dimension of latent space
        self.head_dim = embed_dim // num_heads
        
        # Low-rank projections for queries, keys and values
        self.wq_down = nn.Linear(embed_dim, rank, bias=qkv)  # Projects input to latent space for queries
        self.wk_rope = nn.Linear(embed_dim, rank, bias=qkv)  # Projects input to latent space for RoPE keys
        self.wkv_down = nn.Linear(embed_dim, rank, bias=qkv)  # Shared projection for non-RoPE keys and values
        
        # Up-projections from latent space
        self.wq_rope = nn.Linear(rank, embed_dim, bias=qkv)  # Up-projects latent queries for RoPE portion
        self.wq_up = nn.Linear(rank, embed_dim, bias=qkv)    # Up-projects latent queries for non-RoPE portion
        
        self.wk_up = nn.Linear(rank, embed_dim, bias=qkv)    # Up-projects keys from latent space
        self.wv_up = nn.Linear(rank, embed_dim, bias=qkv)    # Up-projects values from latent space
        
        # Rotary position embeddings for position-aware attention
        self.rope = RotaryPositionEmbedding(self.head_dim, context_length)
        
        # Output projection
        self.wo = nn.Linear(embed_dim, embed_dim, bias=qkv)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass for the Multi-Head Latent Attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, embed_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_tokens, embed_dim)
        """
        b, num_tokens, _ = x.shape
        
        # Query processing - using both RoPE and non-RoPE parts
        # First, project to latent space
        q = self.wq_down(x)  # [b, num_tokens, rank]
        
        # Process portion with RoPE (position-aware)
        q_rope = self.wq_rope(q).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # [b, num_heads, num_tokens, head_dim]
        q_rope = self.rope(q_rope, num_tokens)  # Apply rotary position embeddings
        
        # Process portion without RoPE (content-only)
        q_nope = self.wq_up(q).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # [b, num_heads, num_tokens, head_dim]
        
        # Concatenate the RoPE and non-RoPE parts along the head_dim dimension
        q = torch.cat([q_rope, q_nope], dim=3)  # [b, num_heads, num_tokens, 2*head_dim]
        
        # Key processing - similar approach with RoPE and non-RoPE parts
        # Process portion with RoPE
        k_rope = self.wk_rope(x)  # [b, num_tokens, rank]
        k_rope = self.wk_up(k_rope).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # [b, num_heads, num_tokens, head_dim]
        k_rope = self.rope(k_rope, num_tokens)  # Apply rotary position embeddings
        
        # Process shared latent space for non-RoPE keys and values
        kv = self.wkv_down(x)  # [b, num_tokens, rank]
        
        # Process non-RoPE keys
        k_nope = self.wk_up(kv).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # [b, num_heads, num_tokens, head_dim]
        
        # Concatenate the RoPE and non-RoPE parts for keys
        k = torch.cat([k_rope, k_nope], dim=3)  # [b, num_heads, num_tokens, 2*head_dim]
        
        # Value processing (no RoPE applied to values)
        v = self.wv_up(kv).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # [b, num_heads, num_tokens, head_dim]
        
        # Compute attention using PyTorch's scaled dot-product attention
        # This performs: softmax(Q·K^T/sqrt(d_k))·V with causal masking
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p, is_causal=True)
        
        # Reshape and project output back to embedding dimension
        y = y.transpose(1, 2).contiguous().view(b, num_tokens, self.num_heads * self.head_dim)
        y = self.wo(y)  # Final output projection
        
        return y


# def test_MultiHeadLatentAttention():
#     mha = MultiHeadLatentAttention(256, 512, 4, 128, 0.1)
#     x = torch.randn(1, 512, 256)
#     out = mha(x)
#     print(f"{out.shape=}")

# test_MultiHeadLatentAttention()

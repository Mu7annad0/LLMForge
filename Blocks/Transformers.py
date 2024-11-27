import torch
import torch.nn as nn

from Attentions.CausalAttention import CausalAttention
from Attentions.GroupedQueryAttention import GrupedQueryAttention
from .FeedForwards import GPTFeedForward, LlamaFeedForward
from .Activations import GELU
from .Normalizations import LayerNormalization, RMSNormalization


class GPTTransformer(nn.Module):
    """
    A transformer block implementing the GPT2.
    
    This class represents a single transformer layer with:
    - Layer normalization
    - Causal self-attention mechanism
    - Dropout for regularization
    - Residual connections (skip connections)
    - Feed-forward neural network
    """
    def __init__(self, cfg):
        """
        Initialize the GPT Transformer block with configuration parameters.
        
        Args:
            cfg: Configuration object containing hyperparameters such as:
                - embed_dim: Embedding dimension
                - context_length: Maximum sequence length
                - drop_rate: Dropout probability
                - n_heads: Number of attention heads
                - qkv_bias: Whether to use bias in query, key, value projections
        """
        super().__init__()

        # Self-attention mechanism with causal masking
        self.attn = CausalAttention(
            embed_dim=cfg.embed_dim,
            context_length=cfg.context_length,
            dropout=cfg.drop_rate,
            num_heads=cfg.n_heads,
            qkv=cfg.qkv_bias
        )
        self.ffn = GPTFeedForward(cfg)
        self.lnorm = LayerNormalization(cfg.embed_dim)
        self.gelu = GELU()
        self.drop = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        """
        Forward pass of the transformer block.
        
        The block follows a two-stage process:
        1. Self-attention sub-layer with residual connection
        2. Feed-forward sub-layer with residual connection
        
        Each sub-layer follows the pattern:
        - Layer normalization
        - Sub-layer operation (attention or feed-forward)
        - Dropout
        - Residual connection (addition with input)
        """

        # First sub-layer: Self-Attention
        shortcut = x        # Store original input for residual connection
        x = self.lnorm(x)   # Normalize input before attention
        x = self.attn(x)    # Apply self-attention
        x = self.drop(x)    # Apply dropout for regularization
        x += shortcut       # Residual connection

        # Second sub-layer: Feed-Forward Network
        shortcut = x
        x = self.lnorm(x)
        x = self.ffn(x)     # Apply feed-forward network
        x = self.drop(x)
        x += shortcut

        return x
    

class LlamaTransformer(nn.Module):
    """
    A transformer block implementing the Llama architecture.
    
    Similar to GPT transformer, but with key differences:
    - Uses RMS (Root Mean Square) Layer Normalization instead of standard Layer Normalization
    - Implements Grouped-Query Attention instead of standard self-attention
    """
    def __init__(self, cfg):
        super().__init__()

        # RMS Layer Normalization
        self.rms = RMSNormalization(cfg.embed_dim)
        # Grouped-Query Attention mechanism
        self.attn = GrupedQueryAttention(cfg)
        self.ffn = LlamaFeedForward(cfg)

    def forward(self, x):
        """
        Forward pass of the Llama transformer block.
        
        Simplified two-stage process:
        1. Self-attention sub-layer with residual connection
        2. Feed-forward sub-layer with residual connection
        """

        # First sub-layer: Self-Attention
        shortcut = x        # Store original input for residual connection
        x = self.rms(x)     # RMS normalization before attention
        x = self.attn(x)    # Apply grouped-query attention
        x += shortcut       # Residual connection

        # Second sub-layer: Feed-Forward Network
        shortcut = x
        x = self.rms(x)
        x = self.ffn(x)     # Apply feed-forward network
        x += shortcut

        return x
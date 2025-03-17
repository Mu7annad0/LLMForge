import torch 
import torch.nn as nn
from dataclasses import dataclass

from Attentions.CausalAttention import CausalAttention
from Blocks.FeedForwards import GPTFeedForward
from Blocks.Activations import GELU


class DynamicTanh(nn.Module):
    """
    Dynamic Tanh (DyT) layer as described in the paper "Transformers without Normalization"

    DyT is a drop-in replacement for normalization layers (like LayerNorm or RMSNorm) in 
    Transformer architectures. It applies a tanh function with a learnable scalar parameter α 
    to adjust the input range, followed by an affine transformation with learnable parameters 
    γ and β (similar to those in normalization layers).
    """
    def __init__(self, dim, alpha_init = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = torch.tanh(x * self.alpha)
        return self.gamma * x + self.beta


class Transformer(nn.Module):
    """
    A simple transformer block.
    """
    def __init__(self, cfg):
        """
        Args:
            cfg: Configuration object containing hyperparameters
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
        self.dyt = DynamicTanh(cfg.embed_dim)
        self.gelu = GELU()
        self.drop = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        """
        Forward pass of the transformer block.
        """
        # First sub-layer: Self-Attention
        shortcut = x        # Store original input for residual connection
        x = self.dyt(x)   # using the dynamic tanh (DyT)
        x = self.attn(x)    # Apply self-attention
        x = self.drop(x)    # Apply dropout for regularization
        x += shortcut       # Residual connection

        # Second sub-layer: Feed-Forward Network
        shortcut = x
        x = self.dyt(x)
        x = self.ffn(x)     # Apply feed-forward network
        x = self.drop(x)
        x += shortcut

        return x


@dataclass
class TransformerConfig:
    embed_dim: int = 768
    context_length: int = 512
    drop_rate: float = 0.1
    n_heads: int = 12
    qkv_bias: bool = True

def test_transformer():
    cfg = TransformerConfig()
    transformer = Transformer(cfg)
    
    batch_size = 4
    seq_len = 256
    x = torch.randn(batch_size, seq_len, cfg.embed_dim)
    
    transformer.eval()
    try:
        with torch.no_grad():
            output = transformer(x)
        
        # Check output shape
        expected_shape = (batch_size, seq_len, cfg.embed_dim)
        actual_shape = tuple(output.shape)
        
        # Print results
        print(f"Input shape: {tuple(x.shape)}")
        print(f"Output shape: {actual_shape}")
        print(f"Expected shape: {expected_shape}")
        print(f"Shape test: {'Passed' if actual_shape == expected_shape else 'Failed'}")
        
        return actual_shape == expected_shape
    
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False

# if __name__ == "__main__":
#     success = test_transformer()
#     print(f"\nOverall test result: {'Passed' if success else 'Failed'}")

import torch.nn as nn
import torch
import torch.nn.functional as F


class CausalAttention(nn.Module):
   
    def __init__(self, embed_dim, context_length, dropout, num_heads, qkv=False):
        """Initialize the multi-head attention module.
        
        Args:
            embed_dim (int): Total dimension of the model
            context_length (int): Maximum sequence length
            dropout (float): Dropout probability
            num_heads (int): Number of attention heads
            qkv (bool): Whether to use bias in QKV projection
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads!!"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections
        self.w_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv)  # Projects input into query, key and value
        self.w = nn.Linear(embed_dim, embed_dim)  # Output projection
        self.dropout = nn.Dropout(dropout)

        # Create causal mask to ensure the model can only attend to previous tokens
        self.register_buffer(
            "mask", 
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        """Forward pass of the multi-head attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, embed_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_tokens, embed_dim)
        """
        b, num_tokens, embed_dim = x.shape

        # Project input into query, key, value vectors
        qkv = self.w_qkv(x).chunk(3, dim=-1)
        q, k, v = [x.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2) for x in qkv]
        # Shape after split: (batch_size, num_heads, num_tokens, head_dim)

        
        # att = q @ k.transpose(2, 3)  # Shape: (batch_size, num_heads, num_tokens, num_tokens)
        # mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # att.masked_fill_(mask_bool, -torch.inf)
        # att = torch.softmax(
        #     att / k.shape[-1]**0.5,  # Scale by sqrt(head_dim)
        #     dim=-1
        # )
        # att = self.dropout(att)
        # y = (att @ v)
        # # Shape after attention: (batch_size, num_tokens, num_heads, head_dim)

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p, is_causal=True)

        # Concatenate heads and project back to embed_dim
        y = y.transpose(1, 2).contiguous().view(b, num_tokens, self.embed_dim)
        y = self.w(y)  # Final output shape: (batch_size, num_tokens, embed_dim)
        
        return y
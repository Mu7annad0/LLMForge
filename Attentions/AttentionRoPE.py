import torch
import torch.nn as nn

from Blocks.Positionals import RotaryPositionalEmbedding


class MultiHeadAttentionRoPE(nn.Module):

    def __init__(self, d_in, d_out, context_length, num_heads, dtype=None, qkv=False):
        super().__init__()

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.w_q = nn.Linear(d_in, d_out, bias=qkv, dtype=dtype)
        self.w_k = nn.Linear(d_in, d_out, bias=qkv, dtype=dtype)
        self.w_v = nn.Linear(d_in, d_out, bias=qkv, dtype=dtype)
        self.w = nn.Linear(d_out, d_in, bias=qkv, dtype=dtype)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # Linear transformations and reshaping for multi-head attention
        q = self.w_q(x).view(b, num_tokens, self.num_heads, self.head_dim) # Queries
        k = self.w_k(x).view(b, num_tokens, self.num_heads, self.head_dim) # Keys
        v = self.w_v(x).view(b, num_tokens, self.num_heads, self.head_dim) # Values
        # shape now is --> [b, num_tokens, num_heads, head_dim]

        # apply rotary embedding to query and key
        q = self.rope(q)
        k = self.rope(k)
        # shape now is --> [b, num_tokens, num_heads, head_dim]

        # Transpose to get the shape --> [b, num_heads, num_tokens, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attention_scores = q @ k.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attention_scores.masked_fill_(mask_bool, -torch.inf)  # Apply mask to attention scores

        attention_weights = torch.softmax(
            attention_scores / k.shape[-1]**0.5, dim=-1
        )

        context_vector = (attention_weights @ v).transpose(1, 2)
        # Input shapes: attention_weights (b, num_heads, num_tokens, num_tokens), Values (b, num_heads, num_tokens, head_dim)
        # Output shape: (b, num_heads, num_tokens, head_dim), After transpose: (b, num_tokens, num_heads, head_dim)

        context_vector = context_vector.contiguous().view(
            b, num_tokens, self.d_out  # Reshape to (b, num_tokens, d_out)
        )
        context_vector = self.w(context_vector)
        return context_vector


# Example usage
# d_in = 512
# d_out = 512
# context_length = 1024
# num_heads = 8
# batch_size = 32
# seq_length = 100

# model = MultiHeadAttentionRoPE(d_in, d_out, context_length, num_heads)
# x = torch.randn(batch_size, seq_length, d_in)
# output = model(x)

# print(f"Input shape: {x.shape}")
# print(f"Output shape: {output.shape}")

# python -m Attentions.AttentionRoPE
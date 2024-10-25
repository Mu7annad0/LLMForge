import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv=False):
        super().__init__()
        assert(d_out % num_heads == 0), "d_out must be divisible by num_heads!!"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.w_q = nn.Linear(d_in, d_out, bias=qkv)
        self.w_k = nn.Linear(d_in, d_out, bias=qkv)
        self.w_v = nn.Linear(d_in, d_out, bias=qkv)
        self.w = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        Queries = self.w_q(x).view(b, num_tokens, self.num_heads, self.head_dim)
        Keys = self.w_k(x).view(b, num_tokens, self.num_heads, self.head_dim)
        Values = self.w_v(x).view(b, num_tokens, self.num_heads, self.head_dim)
        # shape now is (b, num_tokens, num_heads, head_dim)

        Queries = Queries.transpose(1, 2)
        Keys = Keys.transpose(1, 2)
        Values = Values.transpose(1, 2)
        # shape now is (b, num_heads, num_tokens, head_dim)

        attn_scores = Queries @ Keys.transpose(2, 3)
        # Input shapes: Queries (b, num_heads, num_tokens, head_dim), Keys (b, num_heads, head_dim, num_tokens)
        # Output shape: (b, num_heads, num_tokens, num_tokens)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(
            attn_scores / Keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ Values).transpose(1, 2)
        # Input shapes: attn_weights (b, num_heads, num_tokens, num_tokens), Values (b, num_heads, num_tokens, head_dim)
        # Output shape: (b, num_heads, num_tokens, head_dim), After transpose: (b, num_tokens, num_heads, head_dim)

        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        # shape of context_vec is (b, num_tokens, d_out)
        context_vec = self.w(context_vec)
        return context_vec
import torch
import torch.nn as nn
import math


class PositionalEncoder(nn.Module):
    def __init__(self, dim, max_seq_length):
        """
        Args:
            :param dim: The dimension of the embedding
            :param max_seq_length: The maximum sequence length.
        """
        super().__init__()

        # create a matrix of shape --> [max_length, d_model]
        pe = torch.zeros(max_seq_length, dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2)).float() * (-math.log(10000.0) / dim)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # --> [1, seq_length, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, embedding_dim]
        # Add the positional encoding to the input tensor (x)
        x = x + self.pe[:, :x.size(1)]
        return x


def precompute_rope_params(head_dim, base=10000, max_seq_length=4096, freq_config=None):
    assert head_dim % 2 == 0, "Embedding dimention must be even"

    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

    # for Llama >= 3
    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        wavelen = 2 * torch.pi / inv_freq

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama

    positions = torch.arange(max_seq_length)

    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)

    cos = torch.cos(angles)[:, :head_dim]  # Slice to match `dim`
    sin = torch.sin(angles)[:, :head_dim]  # Slice to match `dim`
    return cos, sin


def apply_rope(x, cos, sin):
    # x.shape: (bs, n_heads, seqlen, head_dim)
    bs, num_heads, seqlen, head_dim = x.shape
    
    assert head_dim % 2 == 0, "Head dimension must be even"
    
    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seqlen, :head_dim].unsqueeze(0).unsqueeze(0)
    sin = sin[:seqlen, :head_dim].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


# batch_size = 2
# num_heads = 4
# head_dim = 16

# # Instantiate RoPE parameters
# cos, sin = precompute_rope_params(
#     head_dim=head_dim,
#     base=10000,
#     max_seq_length=10000
# )

# # Dummy query and key tensors
# torch.manual_seed(123)
# queries = torch.randn(batch_size, num_heads, 10000, head_dim)
# keys = torch.randn(batch_size, num_heads, 10000, head_dim)

# # Apply rotary position embeddings
# queries_rot = apply_rope(queries, cos, sin)
# keys_rot = apply_rope(keys, cos, sin)

# print(f"{queries_rot}")
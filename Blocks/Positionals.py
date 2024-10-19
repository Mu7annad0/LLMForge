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


class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, dim, base=10000):
        """
        RotaryPositionalEmbedding encodes sequence positions using a rotary approach, enhancing position information capture, especially for longer sequences.

        Args:
            :param dim: The dimensionality of each attention head. Must be even.
            :param base: Base frequency for sinusoidal embeddings (default: 10000).
        """
        super().__init__()
        assert dim % 2 == 0, "Dimension of model must be even for rotary embeddings"
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, max_seq_length=None):
        if max_seq_length is None:
            max_seq_length = x.size(1)
            
        # Compute position encodings --> [max_seq_length, dim // 2]
        t = torch.arange(max_seq_length, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)

        # Duplicate frequencies to match the dimensionality of x --> [max_seq_length, dim]
        emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

        # Split x into two parts for even and odd indices, and embedding into sin and cosine parts
        x1, x2 = x[..., ::2], x[..., 1::2]
        cos = emb[..., None, 1::2].cos()
        sin = emb[..., None, ::2].sin()

        x_out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1) # Apply the rotary encoding
        return x_out # --> [batch_size, max_seq_length, dim]
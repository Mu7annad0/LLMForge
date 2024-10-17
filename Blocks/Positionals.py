import torch
import torch.nn as nn
import math


class PositionalEncoder(nn.Module):
    
    def __init__(self, d_model, max_seq_length):
        """
        :param d_model: the size of each embedding vector
        :param max_seq_length: the maximum length of the input sequence
        """
        super().__init__()

        # create a matrix of shape (max_length, d_model)
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)).float() * (-math.log(10000.0) / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_length, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, embedding_dim]
        # Add the positional encoding to the input tensor (x)
        x = x + self.pe[:, :x.size(1)]
        return x
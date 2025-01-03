"""Implementation of Low-Rank Adaptation (LoRA)"""
import math
import torch
from torch import nn


class LoRA(nn.Module):
    """
    LoRA (Low-Rank Adaptation) Block.

    Args:
        input_dim (int): Input dimension of the layer.
        output_dim (int): Output dimension of the layer.
        rank (int): Rank of the low-rank decomposition.
        alpha (float): Scaling factor for the LoRA weights.
    """
    def __init__(self, input_dim, output_dim, rank, alpha):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.scaling = alpha / rank

        self.lora_a = nn.Parameter(torch.zeros(input_dim, rank)) # lora_a: input_dim x rank
        self.lora_b = nn.Parameter(torch.zeros(rank, output_dim)) # lora_b: rank x output_dim

        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        """
        Forward pass for the LoRA block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Low-rank adaptation: x @ A @ B
        lora_output = x @ self.lora_a @ self.lora_b

        # Scale the output by alpha / rank
        return lora_output * self.scaling


# Example usage
# lora = LoRA(input_dim=768, output_dim=768, rank=8, alpha=16)
# x = torch.randn(32, 768)  # batch_size=32, input_dim=768
# output = lora(x)
# print(f"LoRA output shape: {output.shape}") # Shape: (32, 768)

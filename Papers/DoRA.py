"""Implementing DoRA's paper, https://arxiv.org/abs/2402.09353"""

from torch import nn
import torch.nn.functional as F
from Papers.LoRA import LoRA

class DoRA(nn.Module):
    """
    Directional Low-Rank Adaptation (DoRA) module for adapting pre-trained linear layers.

    DoRA applies a low-rank update to a given linear layer while ensuring directional consistency
    of the weight matrix. Instead of directly adding the low-rank update, the modified weights
    are normalized to maintain their directional properties and rescaled using a learnable
    parameter.

    Args:
        linear (nn.Linear): The original linear layer to apply adaptation.
        rank (int): The rank for low-rank decomposition.
        alpha (float): Scaling factor for the low-rank update.
    """
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRA(linear.in_features, linear.out_features, rank, alpha)
        self.m = nn.Parameter(
            self.linear.weight.norm(p=2, dim=0, keepdim=True))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim) after applying 
            the adapted linear transformation.
        """
        lora = self.lora.lora_a @ self.lora.lora_b

        # Add LoRA update to original weights
        numerator = self.linear.weight + lora.T 
        denominator = numerator.norm(p=2, dim=0, keepdim=True)

        # Ensure directional consistency
        directional_component = numerator / denominator

        # Rescale with learned parameter
        new_weight = self.m * directional_component

        # Apply linear transformation
        return F.linear(x, new_weight, self.linear.bias)

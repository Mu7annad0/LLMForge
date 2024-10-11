import torch.nn as nn
import torch
import math

class GELU(nn.Module):
    """
    The GELU is similar to the ReLU (Rectified Linear Unit) activation function, but it has a smoother curve. 
    The GELU function is defined as follows:

    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    This function is used to introduce non-linearity in the model, allowing it to learn more complex relationships between the inputs and outputs.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh
                          (math.sqrt(2 / math.pi) * 
                           (x + 0.044715 * x ** 3)))


import torch.nn as nn
import torch
import math

class GELU(nn.Module):
    
    def __init__(self):
        """
        The GELU is similar to the ReLU (Rectified Linear Unit) activation function, but it has a smoother curve. 
        The GELU function is defined as follows:

        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

        This function is used to introduce non-linearity in the model, allowing it to learn more complex relationships between the inputs and outputs.
        """
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh
                          (math.sqrt(2 / math.pi) * 
                           (x + 0.044715 * x ** 3)))


class SiLU(nn.Module):

    def __init__(self):
        """
        The SiLU (Sigmoid Linear Unit) activation function is a non-linear function that is used in deep learning models. 
        It is defined as follows:
        SiLU(x) = x * (1 + exp(-x))
        SiLU(x) = x * sigmoid(x)
        """
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
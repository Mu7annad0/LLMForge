import torch.nn as nn
import torch


class LayerNormalization(nn.Module):

    def __init__(self, dim, eps = 1e-5):
        """
        LayerNormalization is a technique used to normalize the inputs to a layer for each training example. 
        It normalizes the activations of each layer for each training example, 
        i.e., applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
        Args:
            dim (int): The dimensionality of the embedding space.
            eps (float): A small value added to the denominator for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.gama = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim = -1, keepdim = True, unbiased = False)
        out_norm = (x - mean) / torch.sqrt(var + self.eps)
        return (self.gama * out_norm) + self.beta
    

class RMSNormalization(nn.Module):

    def __init__(self, dim, eps=1e-5):
        """
        RMSNorm is a normalization technique that normalizes the input tensor by its root mean square (RMS) value. 
        Args:
            dim (int): The dimension of the input tensor.
            eps (float): A small value added to the denominator for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim)).float()

    def forward(self, x):
        rms = torch.sqrt(self.eps + torch.mean(x**2, dim=-1, keepdim=True))
        x_norm = x / rms
        return x_norm * self.scale
